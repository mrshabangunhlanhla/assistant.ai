# react_agent_langgraph.py
import asyncio

import re
from typing import Any, AsyncGenerator, Dict, List, Optional
import os
from collections import deque 
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool, Tool 


from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Access the environment variables (Ensure GOOGLE_API_KEY is set in .env)
api_key = os.getenv("GOOGLE_API_KEY")

ReActStreamEvent = Dict[str, str]  # {"type": "thought"|"observation"|"final_answer"|"error", "content": str}


def get_default_system_prompt(tools: List[BaseTool]) -> str:
    """ReAct (Yao et al., 2022) style system prompt, optimized for detailed reasoning."""
    tool_list = "\n".join([f"- {t.name}: {t.description}" for t in tools])

    return (
        "You are an intelligent reasoning agent that follows the ReAct format exactly.\n"
        "You can reason (Thought), act (Action), and observe (Observation).\n"
        "At each step, output **only one Thought and one Action**.\n"
        "After the tool executes, you will receive an Observation.\n"
        "Then continue reasoning, acting, and observing until you have a final answer.\n\n"
        
        "Your response MUST always contain a Thought and an Action in the exact format.\n"
        "**CRITICAL:** Once you have the final answer (either immediately or after searching), you MUST use the **Action: finish[<final answer>]** format. **DO NOT EVER STOP OUTPUTTING BEFORE THIS STEP IS COMPLETE.**\n"
        
        "Example of a simple response:\n"
        "Thought: The user is greeting me. I should respond conversationally and use the finish tool immediately.\n"
        "Action: finish[Hello! I am an intelligent reasoning agent. How can I assist you with your query today?]\n\n"
        
        "The following tools are available:\n"
        f"{tool_list}\n\n"
        "Follow this exact format:\n"
        "Thought: [Your reasoning]\n"
        "Action: [Tool name][Tool input]\n\n"
        "Do NOT output the Observation. You will receive it in the next turn.\n"
        "Do NOT repeat the same Thought/Action sequence."
    )


class ReActAgent:
    """
    A ReAct (Reasoning and Acting) agent built to enforce the strict
    Thought -> Action -> Observation loop using streaming for language model output.
    """
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        history_length: int = 3, # How many recent actions to check for loops
    ):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.history_length = history_length
        self.action_history = deque(maxlen=history_length) # Track recent actions
        self.messages: List[BaseMessage] = []
        self._tool_map = {t.name: t for t in self.tools}
        self._last_action: Optional[tuple[str, str, str]] = None  # (full_text, action_name, action_input)
        self._system_prompt = system_prompt or get_default_system_prompt(self.tools)

    def _parse_action(self, text: str) -> Optional[tuple[str, str, str]]:
        """
        Parses the LLM output for the Thought and Action.
        """
        
        # 1. Look for the perfect ReAct pattern (Thought + Action at the end)
        pattern_perfect = re.compile(
            r"Thought: (.*?)\s*Action: (.*?)\s*\[(.*?)\]\s*$",
            re.DOTALL | re.IGNORECASE
        )
        match_perfect = pattern_perfect.search(text)
        if match_perfect:
            full_text = match_perfect.group(0).strip()
            action_name = match_perfect.group(2).strip()
            action_input = match_perfect.group(3).strip()
            return (full_text, action_name, action_input)

        # 2. Aggressively look for an Action pattern anywhere, regardless of Thought formatting
        pattern_action_only = re.compile(
            r"Action: (.*?)\s*\[(.*?)\]",
            re.DOTALL | re.IGNORECASE
        )
        match_action_only = pattern_action_only.search(text)
        if match_action_only:
            action_name = match_action_only.group(1).strip()
            action_input = match_action_only.group(2).strip()
            
            # Synthesize Thought from preceding text, or create a default one
            thought_text = text[:match_action_only.start()].strip()
            if not thought_text:
                thought_text = f"Synthesized Thought: Agent found an unformatted Action intent in the raw output."

            synthetic_full_text = f"Thought: {thought_text}\nAction: {action_name}[{action_input}]"
            
            return (synthetic_full_text, action_name, action_input)

        # 3. Look for a Thought only (or partial conversational text)
        thought_pattern = re.compile(r"Thought: (.*?)\s*$", re.DOTALL | re.IGNORECASE)
        thought_match = thought_pattern.search(text)
        if thought_match:
            return (text.strip(), "", "")
        
        return None

    async def _stream_thought_action(self) -> AsyncGenerator[ReActStreamEvent, None]:
        """Streams the LLM's response until a complete Action is parsed."""
        
        prompt_messages = [SystemMessage(content=self._system_prompt)] + self.messages
        
        # 1. Stronger Instruction Injection: If a tool was just executed successfully, 
        # tell the model explicitly to use the observation content.
        if self.messages and isinstance(self.messages[-1], ToolMessage) and "SUCCESS" in self.messages[-1].content:
            prompt_messages.append(HumanMessage(content="CRITICAL: You have the successful observation. Immediately generate the final answer using the Thought: ... Action: finish[...] format. DO NOT HALLUCINATE A FAILURE. Copy the observation content *exactly* into the finish[] action."))
        
        full_response = ""
        action_found = False
        
        # Collect the full streaming output first
        async for chunk in self.llm.astream(prompt_messages):
            if chunk.content:
                full_response += chunk.content

        # Now parse the full output
        parsed_action = self._parse_action(full_response)
        
        if parsed_action and parsed_action[1]: # Action name is non-empty
            self._last_action = parsed_action
            yield {"type": "thought_action", "content": parsed_action[0]}
            action_found = True

        # --- REPAIR LOGIC: Force Finish Action, prioritizing successful Observation content ---
        # The condition checks if an action wasn't found OR if there's a successful observation.
        if not action_found and (full_response.strip() or (self.messages and isinstance(self.messages[-1], ToolMessage))):
             
            raw_output_snippet = full_response.strip()
            final_answer_content = raw_output_snippet # Default to LLM's partial output

            # *** THE CRITICAL FIX: OVERRIDE LLM OUTPUT WITH SUCCESSFUL OBSERVATION ***
            if self.messages and isinstance(self.messages[-1], ToolMessage) and "SUCCESS" in self.messages[-1].content:
                
                last_observation_content = self.messages[-1].content
                synthetic_thought = (
                    "Agent successfully executed a tool and received a SUCCESS Observation, but failed to form a complete Action or hallucinated an incorrect answer. "
                    "Forcing a 'finish' action by synthesizing the final answer from the verified Observation content."
                )
                
                # Extract the clean STDOUT content for presentation (this is key)
                stdout_match = re.search(r"\[STDOUT\]:\s*(.*)", last_observation_content, re.DOTALL)
                if stdout_match:
                    # Use the clean STDOUT content
                    final_answer_content = stdout_match.group(1).strip()
                else:
                    # Fallback to the full successful observation content
                    final_answer_content = last_observation_content.strip() 

            else:
                # Default conversational/incomplete output repair
                synthetic_thought = (
                    f"The model responded conversationally or with an incomplete format. Forcing a 'finish' action "
                    f"to deliver the response: '{raw_output_snippet[:80].replace('\n', ' ')}...'"
                )
                final_answer_content = raw_output_snippet
                
            
            synthetic_full_text = f"Thought: {synthetic_thought}\nAction: finish[{final_answer_content}]"
            
            self._last_action = (synthetic_full_text, "finish", final_answer_content)
            
            # Yield the synthetic text which was not captured in the earlier stream
            yield {"type": "thought_action", "content": synthetic_full_text}
            return


    async def _execute_tool(self, name: str, input_str: str) -> str:
        """Executes the specified tool and returns the observation."""
        tool = self._tool_map.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available tools: {list(self._tool_map.keys())}"

        try:
            observation = await tool.ainvoke(input_str)
            
            if not isinstance(observation, str):
                observation = str(observation)
                
            return observation
        except Exception as e:
            return f"Tool Execution Error for {name}: {e}"

    async def stream(self, query: str) -> AsyncGenerator[ReActStreamEvent, None]:
        """Executes the ReAct loop until a final answer is found or max iterations are reached."""
        
        self.messages.clear()
        self.messages.append(HumanMessage(content=query))
        self.action_history.clear() 

        for i in range(1, self.max_iterations + 1):
            try:
                # 1. Stream Thought and Action
                self._last_action = None
                # The _stream_thought_action now buffers and yields the final result in one go
                async for evt in self._stream_thought_action(): 
                    yield evt 
                
                if not self._last_action:
                    yield {"type": "error", "content": "LLM failed to produce a complete 'Action: [Tool name][Tool input]' after a Thought."}
                    return

                full_text, action_name, action_input = self._last_action

                # --- LOOP DETECTION LOGIC ---
                current_action_signature = (action_name.lower(), action_input)
                
                if action_name.lower() != "finish" and current_action_signature in self.action_history:
                    yield {
                        "type": "error", 
                        "content": f"Loop Detected on iteration {i}: Repetitive action '{action_name}[{action_input}]'. Terminating early to prevent infinite execution. Revise your thought process."
                    }
                    return

                self.action_history.append(current_action_signature)
                # --- END LOOP LOGIC ---

                # 2. Check for Finish Action
                if action_name.lower() == "finish":
                    self.messages.append(AIMessage(content=full_text))
                    yield {"type": "final_answer", "content": action_input}
                    return 

                # 3. Execute Tool Action
                observation = await self._execute_tool(action_name, action_input)
                
                # 4. Add Observation to Memory
                yield {"type": "observation", "content": observation}
                
                self.messages.append(AIMessage(content=full_text))
                tool_msg = ToolMessage(
                    content=observation, 
                    tool_call_id=f"call_{action_name}_{i}", 
                    name=action_name
                )
                self.messages.append(tool_msg)
                
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                return

        # --- MAX ITERATION FAILURE LOGIC ---
        yield {
            "type": "error", 
            "content": f"Maximum steps ({self.max_iterations}) reached without a final answer. Agent failed to converge to a solution. Review the agent's memory trace."
        }
        # --- END MAX ITERATION FAILURE LOGIC ---


from lib.tools import  google_search, run_shell_command, calculate, finish

# === Example Usage (Interactive CLI) ===
if __name__ == "__main__":

    tools = [google_search, calculate, run_shell_command]

    async def main():
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        agent = ReActAgent(llm=llm, tools=[*tools, finish], max_iterations=7)

        print("--- AI Agent Initiated ---")
        print(f"Max Iterations set to: {agent.max_iterations}")

        quary = ''
        while quary != "quit":
            quary = input("\n\n?: ")
            
            if quary == "quit":
                break
            
            if quary.strip():
                print("\n--- START TRACE ---")
                full_result = ""
                async for event in agent.stream(quary):
                    event_type = event["type"]
                    content = event["content"]
                    
                    if event_type == "thought_action":
                        print(f"\n{content}")
                    elif event_type == "observation":
                        print(f"Observation: {content}")
                    elif event_type == "final_answer":
                        full_result = content
                        break
                    elif event_type == "error":
                        print(f"\n[!!! ERROR: {content} !!!]")
                        break 
                print("\n--- END TRACE ---\n\n")
                if full_result:

                    print(f"AI: {full_result}")

    # Run the main async function
    try:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting agent.")