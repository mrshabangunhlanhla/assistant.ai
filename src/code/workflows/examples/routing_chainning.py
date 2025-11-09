from langgraph.graph import StateGraph, MessagesState, START, END
from typing_extensions import Literal, TypedDict
from lib.models import groq
import asyncio
from pydantic import BaseModel, Field

class State(TypedDict): 
    input: str
    decision: str
    output: str
llm = groq()
class Router(BaseModel):
    step: Literal["joke", "story", "poem"] = Field(description="The next step")

    
def joke(state: State):
    """
    llm write a joke from the user input
    """
    input = state.get("input")
    # print(state.items())
    llm = groq()
    response = llm.invoke(["human", f"""Write a simple joke about '{input}'"""])
    print("Joke")
    return {"output": response.text}

def story(state: State):
    """
    llm write a story from the user input
    """
    input = state.get("input")
    # print(state.items())
    llm = groq()

    response = llm.invoke(["human", f"""Write a simple story about '{input}'"""])
    # print(response)
    print("Story")
    return {"output": response.text}

def poem(state: State):
    """
    llm write a poem from the user input
    """
    input = state.get("input")
    # print(state.items())
    llm = groq()
    response = llm.invoke(["human", f"""Write a simple poem about '{input}'"""])
    # print(response)
    # print("Poem")
    return {"output": response.text}


def router(state: State):
    llm = groq().with_structured_output(Router)

    response = llm.invoke(state["input"])
    step = response.step
    # print(step)
    return {"decision": step}

def handleDecision(state: State):
    decision = state["decision"]
    # print(decision)
    if "joke" in decision:
        return "joke"
    elif "poem" in decision:
        return "poem"
    else: return "story"


graph = StateGraph(State)
graph.add_node(joke)
graph.add_node(story)
graph.add_node(poem)
graph.add_node(router)
graph.add_edge(START, 'router')
graph.add_conditional_edges("router", handleDecision)

graph.add_edge("joke", END)
graph.add_edge("story", END)
graph.add_edge("story", END)

graph = graph.compile()

async def main(query):
    if(query.strip()):
        state = graph.invoke({"input": query})
        decision = state["decision"]
        output = f"""{decision[0].upper()+decision[1:]}\n{state["output"]}\n"""
        print(output)

# === Example Usage ===
if __name__ == "__main__":
    from lib.hf import run_loop
    run_loop(main, """A joke, poem or a story writer\n\nWhat do you want me to write about? """)
