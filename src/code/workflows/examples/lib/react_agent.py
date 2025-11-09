# react_agent_langgraph.py
import asyncio
import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional
import os
import shutil
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool, StructuredTool
from langchain.tools import tool # Required for the @tool decorator

from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Access the environment variables
api_key = os.getenv("GOOGLE_API_KEY")

ReActStreamEvent = Dict[str, str]  # {"type": "thought"|"observation"|"final_answer"|"error", "content": str}


def get_default_system_prompt(tools: List[BaseTool]) -> str:
    """ReAct (Yao et al., 2022) style system prompt, optimized for detailed reasoning."""
    tool_list = "\n".join([f"- {t.name}: {t.description}" for t in tools])

    return (
        "You are an intelligent reasoning agent that follows the ReAct format exactly.\n"
        "You can reason (Thought), act (Action), and observe (Observation).\n"
        "At each step, output only one Thought and one Action.\n"
        "After the tool executes, you will receive an Observation.\n"
        "Then continue reasoning, ensuring your next **Thought** explicitly **summarizes the key takeaway from the Observation** and **outlines the next strategic step**.\n\n"
        "### Available Tools ###\n"
        f"{tool_list}\n\n"
        "### Required Format ###\n"
        "Thought 1: <your reasoning>\n"
        "Action 1: <tool_name>[{...}]\n"
        "Observation 1: <tool result>\n"
        "Thought 2: <your reasoning>\n"
        "Action 2: <next tool or finish[{\"input\": \"final answer\"}]>\n\n"
        "### Example ###\n"
        "Question: What is 7 + 9?\n"
        "Thought 1: I should use the add tool to add 7 and 9.\n"
        "Action 1: add[{\"a\": 7, \"b\": 9}]\n"
        "Observation 1: The result is 16.\n"
        "Thought 2: The observation successfully confirmed that 7 + 9 equals 16. Since the task is complete, I will now provide the final answer.\n"
        "Action 2: finish[{\"input\": \"16\"}]\n\n"
        "**CRITICAL RULE: You MUST always generate a Thought and an Action in sequence. Once the result is known, the final Action MUST use the `finish` tool.**"
    )
class ReActAgent:
    """A ReAct-style agent built for LangChain Core + LangGraph."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
    ):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt or get_default_system_prompt(tools)
        self.max_iterations = max_iterations
        self.messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        self._last_action: Optional[tuple] = None

    def clear(self):
        """Clear conversation memory, keeping the system prompt."""
        self.messages = [self.messages[0]]

    async def stream(
        self, user_input: str
    ) -> AsyncGenerator[ReActStreamEvent, None]:
        """Stream reasoning steps, tool outputs, and final answers."""
        self.messages.append(HumanMessage(content=user_input))

        for _ in range(self.max_iterations):
            try:
                
                # Stream Thought and Action
                self._last_action = None
                async for evt in self._stream_thought_action():
                    yield evt

                if not self._last_action:
                    raise ValueError("LLM did not produce an Action.")

                full_text, action_name, action_input = self._last_action

            except Exception as e:
                yield {"type": "error", "content": str(e)}
                return

            # Add LLM reasoning to memory
            self.messages.append(AIMessage(content=full_text))

            # === FINISH ACTION ===
            if action_name.lower() == "finish":
                try:
                    parsed = json.loads(action_input)
                    final_answer = parsed.get("input", action_input)
                except Exception:
                    final_answer = action_input

                # ✅ Make sure it's a string
                final_answer_str = str(final_answer)

                for ch in final_answer_str:
                    yield {"type": "final_answer", "content": ch}

                self.messages.append(AIMessage(content=final_answer_str))
                return

            # === TOOL ACTION ===
            tool = self.tools.get(action_name)
            if not tool:
                yield {"type": "error", "content": f"Unknown tool '{action_name}'"}
                return

            # Parse JSON input if possible
            try:
                parsed_input = json.loads(action_input)
            except json.JSONDecodeError:
                parsed_input = action_input

            try:
                if asyncio.iscoroutinefunction(tool.invoke):
                    observation = await tool.invoke(parsed_input)
                else:
                    observation = tool.invoke(parsed_input)
            except Exception as e:
                yield {"type": "error", "content": f"Tool error: {e}"}
                return

            # Stream Observation
            for ch in str(observation):
                yield {"type": "observation", "content": ch}

            # Add observation to context
            self.messages.append(HumanMessage(content=f"Observation: {observation}"))
            length = len(self.messages)

            print(self.messages[length-1].content)

        yield {"type": "error", "content": "Max iterations reached without finishing."}

    async def _stream_thought_action(self) -> AsyncGenerator[ReActStreamEvent, None]:
        """
        Stream the LLM reasoning (Thoughts) and capture the Action command.
        """
        full_text = ""
        # FIX: Non-greedy matching (.*?) and numbered action support (Action\s*\d*:)
        action_regex = re.compile(r"Action\s*\d*:\s*([a-zA-Z0-9_-]+)\s*\[(.*?)\]", re.DOTALL)

        async for chunk in self.llm.astream(self.messages):
            content = getattr(chunk, "content", str(chunk))
            full_text += content

            # Simplified streaming: Stream ALL LLM output as "thought"
            yield {"type": "thought", "content": content}

        # After LLM completes, parse the Action line
        # FIX: Use findall to capture ALL actions and take the LAST one for robustness.
        all_matches = action_regex.findall(full_text)
        
        if not all_matches:
            raise ValueError(f"No Action found in LLM response:\n{full_text}")

        # Extract the last valid action match
        action_name, action_input = all_matches[-1]
        self._last_action = (full_text, action_name.strip(), action_input.strip())

    async def invoke(self, user_input: str) -> AIMessage:
        """Run agent fully and return only the final answer."""
        result = []
        async for evt in self.stream(user_input):
            if evt["type"] == "final_answer":
                result.append(evt["content"])
        return AIMessage(content="".join(result))


# from tools import  google_search, run_shell_command, calculate, finish


# tools = [
#     calculate, 
#     google_search, 
#     run_shell_command
# ]

async def onAgentStream(quary, agent):

    async for evt in agent.stream(quary):
        # NOTE: Cleaned the output stream for better display
        if evt["type"] == "thought":
            print(f"Thought: {evt['content']}", end="", flush=True)
        elif evt["type"] == "observation":
            # We print Observation and ensure a newline for clarity
            print(f"\nObservation: {evt['content']}", end="", flush=True)
        elif evt["type"] == "final_answer":
            print(f"\nFinal Answer: {evt['content']}")
        elif evt["type"] == "error":
            print(f"\nError: {evt['content']}")


async def onAgentInvoke(quary, agent):
    final_answer_message = await agent.invoke(quary)
    print(f"""AI> {final_answer_message.content}""")


# # === Example Usage ===
# from models import groq, googleAI
# llm = googleAI()
# agent = ReActAgent(llm=llm, tools=[*tools, finish])
# async def main(query):
#     if(query.strip()):
#         # await onAgentStream(query, agent)
#         await onAgentInvoke(query, agent)

# if __name__ == "__main__":
#     from hf import run_loop
#     run_loop(main, "> ")
