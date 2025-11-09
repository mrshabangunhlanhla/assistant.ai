import asyncio
import re
from typing import AsyncGenerator, Dict, Generator, List, Optional, Tuple
import functools
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain.tools import tool # Needed for the example usage
from pydantic import BaseModel, Field
# Define the Stream Event type for consistent output
# Type: {"type": "thought"|"action_input"|"observation"|"final_answer"|"error", "content": str}
ReActStreamEvent = Dict[str, str] 

class ToolExecutionError(Exception):
    """Custom exception for tool execution failures."""
    pass

class ToolNotFoundError(Exception):
    """Custom exception when the LLM suggests a non-existent tool."""
    pass

# --- Agent Implementation ---

class OptimalReActAgent:
    """
    An advanced ReAct Agent combining streaming, invoking, detailed tracing,
    and robust error handling, adhering to the ReAct paper's principles.
    It provides both asynchronous (stream, async_invoke) and 
    synchronous (stream_sync, invoke) interfaces.
    
    Supports two modes:
    1. dense (default): Enforces a strict (Thought -> Action) loop[cite: 127].
    2. sparse: Allows the LLM to decide when to think or act.
    """
    # Regex to robustly parse LLM output (common LangChain format)
    ACTION_REGEX = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)
    FINAL_ANSWER_REGEX = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
    THOUGHT_REGEX = re.compile(r"Thought:\s*(.*)", re.DOTALL)
    
    # Tool used for final completion
    @tool
    def finish(self, final_answer: str) -> str:
        """
        Tool used to signal the final answer has been reached. 
        The final answer MUST be provided as the input to this tool.
        """
        return final_answer
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        max_iterations: int = 7,
        sparse_reasoning: bool = False, # <-- MODIFICATION
        system_prompt: Optional[str] = None
    ):
        """Initializes the ReAct Agent."""
        self.llm = llm
        # The agent always includes the 'finish' tool for termination
        self.tools_map = {t.name: t for t in tools + [self.finish]}
        self.max_iterations = max_iterations
        self.sparse_reasoning = sparse_reasoning # <-- MODIFICATION
        self.tool_names = list(self.tools_map.keys())
        self.system_prompt_text = system_prompt or self._get_default_system_prompt(tools)
        
    @staticmethod
    def _format_tool_list(tools: List[BaseTool]) -> str:
        """Formats the list of tools for inclusion in the system prompt."""
        return "\n".join([f"- {t.name}: {t.description}" for t in tools])

    def _get_default_system_prompt(self, tools: List[BaseTool]) -> str:
        """Generates a ReAct-style system prompt based on the reasoning mode."""
        tool_list = self._format_tool_list(tools)
        
        # <-- MODIFICATION START -->
        if self.sparse_reasoning:
            # Prompt for sparse reasoning (decision-making tasks)
            # Allows for asynchronous occurrence of thoughts and actions 
            return (
                "You are an intelligent reasoning agent.\n"
                "You can reason (Thought) or act (Action) to solve the task.\n"
                "At each step, you can choose to output a Thought to reason about the state, "
                "or an Action to interact with the environment. You can also do both.\n"
                "Thoughts can be used to decompose goals, track progress, and update your plan, "
                "but are not required on every step.\n"
                "Use the 'finish' tool with your final answer when the task is complete.\n\n"
                "Available tools:\n"
                f"{tool_list}\n"
                f"- finish: {self.finish.description}\n\n"
                "Example Formats:\n"
                "Thought: I need to [reasoning step]\n"
                "Action: tool_name[tool_input]\n\n"
                "OR\n"
                "Action: tool_name[tool_input]\n\n"
                "OR\n"
                "Thought: I need to update my plan. First, I will... [reasoning step]\n\n"
                "Observation: [Tool result]\n"
            )
        else:
            # Original prompt for dense reasoning (knowledge-intensive tasks)
            # Enforces interleaved thought-action steps 
            return (
                "You are an intelligent reasoning agent that follows the ReAct format exactly.\n"
                "Your process must alternate strictly between Thought, Action, and Observation.\n"
                "You can reason (Thought), act (Action), and observe (Observation).\n"
                "At each step, output ONLY one Thought and one Action.\n"
                "Use the 'finish' tool with your final answer when the task is complete.\n\n"
                "Available tools:\n"
                f"{tool_list}\n"
                f"- finish: {self.finish.description}\n\n"
                "Format MUST be:\n"
                "Thought: I need to [reasoning step]\n"
                "Action: tool_name[tool_input]\n\n"
                "Observation: [Tool result]\n\n"
            )
        # <-- MODIFICATION END -->
        
    def _parse_agent_output(self, llm_output: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Robustly parses the LLM output for Thought, Action Name, and Action Input.
        Returns: (Thought, Action_Name, Action_Input)
        """
        thought_match = self.THOUGHT_REGEX.search(llm_output)
        action_match = self.ACTION_REGEX.search(llm_output)
        final_answer_match = self.FINAL_ANSWER_REGEX.search(llm_output)

        thought = thought_match.group(1).strip() if thought_match else None
        
        # Check for Final Answer
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            return thought, "finish", final_answer
            
        # Check for Action
        if action_match:
            action_name = action_match.group(1).strip()
            action_input = action_match.group(2).strip()
            return thought, action_name, action_input
            
        return thought, None, None

    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Executes a tool asynchronously with error handling."""
        if tool_name not in self.tools_map:
            raise ToolNotFoundError(f"Tool '{tool_name}' is not recognized.")

        tool_func = self.tools_map[tool_name]
        
        try:
            # Check for async tool implementation
            if asyncio.iscoroutinefunction(tool_func.func):
                observation = await tool_func.func(tool_input)
            else:
                # Synchronous tool call executed in a thread pool to prevent blocking
                # We rely on the event loop management in stream_sync to ensure get_event_loop()
                # works correctly here.
                observation = await asyncio.get_event_loop().run_in_executor(
                    None, functools.partial(tool_func.func, tool_input)
                )
            
            return str(observation).strip()

        except Exception as e:
            raise ToolExecutionError(f"Error executing tool '{tool_name}' with input '{tool_input}': {type(e).__name__}: {str(e)}")


    async def stream(self, user_input: str) -> AsyncGenerator[Dict[str, str], None]:
        """
        The core ReAct loop, yielding events for streaming and tracing (Asynchronous).
        Supports both dense (Thought/Action) and sparse (Thought or Action) reasoning.
        """
        history: List[BaseMessage] = [SystemMessage(content=self.system_prompt_text), HumanMessage(content=user_input)]
        
        for i in range(1, self.max_iterations + 1):
            
            try:
                # 1. LLM Call
                yield {"type": "info", "content": f"--- Iteration {i}/{self.max_iterations} ---"}
                
                llm_response_message: AIMessage = await self.llm.ainvoke(history)
                llm_output = llm_response_message.content

                # <-- MODIFICATION START -->
                # For sparse reasoning, always add the LLM's output to history
                # This allows for "thought-only" steps
                if self.sparse_reasoning:
                    history.append(llm_response_message)
                # <-- MODIFICATION END -->
                
                # 2. Parsing LLM Output
                thought, action_name, action_input = self._parse_agent_output(llm_output)

                # 3. Output Thought (for tracing)
                if thought:
                    yield {"type": "thought", "content": thought}
                
                # 4. Handle Final Answer
                if action_name == "finish":
                    final_answer = action_input or llm_output
                    yield {"type": "final_answer", "content": final_answer}
                    return 
                
                # 5. Handle Action
                action_executed = False # Flag to track if an action was taken
                if action_name and action_input is not None:
                    
                    yield {"type": "action_input", "content": f"Action: {action_name}[{action_input}]"}
                    
                    # 6. Execute Tool
                    observation = await self._execute_tool(action_name, action_input)
                    action_executed = True
                    
                    # 7. Output Observation (for tracing)
                    yield {"type": "observation", "content": observation}
                    
                    # 8. Update History for next turn
                    # <-- MODIFICATION START -->
                    if not self.sparse_reasoning:
                        # Dense mode: add the LLM's T-A pair now
                        history.append(AIMessage(content=llm_output))
                    
                    # Both modes: add the tool observation
                    history.append(HumanMessage(content=f"Observation: {observation}"))
                    # <-- MODIFICATION END -->

                # 9. Handle invalid LLM output based on reasoning mode
                # <-- MODIFICATION START -->
                if not self.sparse_reasoning and (thought is None or not action_executed):
                    # Dense mode requires BOTH thought and action
                    error_msg = f"LLM output failed to follow DENSE ReAct format (Thought AND Action required). Output:\n{llm_output}"
                    yield {"type": "error", "content": error_msg}
                    return
                
                if self.sparse_reasoning and thought is None and not action_executed:
                    # Sparse mode requires AT LEAST a thought OR an action
                    error_msg = f"LLM output failed to parse Thought or Action. Output:\n{llm_output}"
                    yield {"type": "error", "content": error_msg}
                    return
                # <-- MODIFICATION END -->

            except ToolNotFoundError as e:
                yield {"type": "error", "content": f"Parsing Error: {str(e)}. The tool suggested does not exist."}
                return

            except ToolExecutionError as e:
                yield {"type": "error", "content": f"Tool Execution Error: {str(e)}. The agent stopped."}
                return

            except Exception as e:
                yield {"type": "error", "content": f"An unexpected error occurred during iteration {i}: {type(e).__name__}: {str(e)}"}
                return
        
        # 10. Max Iterations reached
        yield {"type": "error", "content": f"Max iterations ({self.max_iterations}) reached without finding a Final Answer. Agent stopped."}


    # --- Synchronous Methods (Implementations for LangChain compatibility and sync usage) ---

    def stream_sync(self, user_input: str) -> Generator[Dict[str, str], None, None]:
        """Synchronous wrapper for the asynchronous stream method."""
        loop = None
        newly_created = False
        
        try:
            # 1. Try to get the running loop (if nested async)
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 2. If not running, create a dedicated loop for synchronous execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop) # Set the new loop as current for this thread
            newly_created = True

        async_gen = self.stream(user_input)
        
        try:
            while True:
                try:
                    # Use run_until_complete to block until the next item is yielded
                    next_event = loop.run_until_complete(async_gen.__anext__())
                    yield next_event
                except StopAsyncIteration:
                    # Generator finished its work
                    break
                except Exception as e:
                    # Catch any exceptions during generator execution
                    yield {"type": "error", "content": f"Synchronous streaming error: {type(e).__name__}: {str(e)}"}
                    break
        finally:
            # 3. Cleanup: If we created the loop, close it and unset it
            if newly_created and loop is not None:
                loop.close()
                # Unset the loop if it's still the current one
                if asyncio.get_event_loop() is loop:
                    asyncio.set_event_loop(None)


    def invoke(self, user_input: str) -> AIMessage:
        """
        Invokes the agent to get a single, final AIMessage result (Synchronous).
        This method is required by the BaseLanguageModel interface (inherited indirectly).
        """
        result = []
        error_message = None

        # Process stream synchronously
        for evt in self.stream_sync(user_input):
            if evt["type"] == "final_answer":
                result.append(evt["content"])
            elif evt["type"] == "error":
                error_message = f"[ERROR] Agent failed: {evt['content']}"
                break
        
        if error_message:
            return AIMessage(content=error_message)
        elif result:
            return AIMessage(content="".join(result))
        else:
            return AIMessage(content="[ERROR] Agent stopped without providing a Final Answer.")
    
    async def async_invoke(self, user_input: str) -> AIMessage:
        """Asynchronous version of invoke."""
        result = []
        async for evt in self.stream(user_input):
            if evt["type"] == "final_answer":
                result.append(evt["content"])
            elif evt["type"] == "error":
                return AIMessage(content=f"[ERROR] Agent failed: {evt['content']}")
                
        if result:
            return AIMessage(content="".join(result))
        else:
            return AIMessage(content="[ERROR] Agent stopped without providing a Final Answer or an explicit error event.")