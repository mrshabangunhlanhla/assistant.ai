# import sys
# import os

# module_dir = "../lib"
# sys.path.append(module_dir)
import asyncio
from lib.models import groq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# Graph state
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

llm = groq()
# Nodes
def generate_joke(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def check_punchline(state: State):
    """Gate function to check if the joke has a punchline"""

    # Simple check - does the joke contain "?" or "!"
    if "?" in state["joke"] or "!" in state["joke"]:
        return "PASS"
    return "FAIL"


def improve_joke(state: State):
    """Second LLM call to improve the joke"""

    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: State):
    """Third LLM call for final polish"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}



graph = StateGraph(State)
graph.add_node(generate_joke)
graph.add_node(improve_joke)
graph.add_node(polish_joke)
graph.add_edge(START, "generate_joke")
graph.add_conditional_edges("generate_joke", check_punchline, {"PASS": "improve_joke", "FAIL": END})
graph.add_edge("improve_joke", "polish_joke")
graph.add_edge("polish_joke", END)
workflow = graph.compile()
async def main(query):
        state = workflow.invoke({"topic": query})
        print("Initial joke:")
        print(state["joke"])
        print("\n--- --- ---\n")
        if "improved_joke" in state:
            print("Improved joke:")
            print(state["improved_joke"])
            print("\n--- --- ---\n")

            print("Final joke:")
            print(state["final_joke"])
        else:
            print("Joke failed quality gate - no punchline detected!")
  
# === Example Usage ===
if __name__ == "__main__":
    from lib.hf import run_loop
    run_loop(main, "topic: ")
