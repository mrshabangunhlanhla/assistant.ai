from langgraph.graph import StateGraph, MessagesState, START, END
from typing_extensions import TypedDict
from lib.models import groq
import asyncio

class State(TypedDict): 
    topic: str
    joke: str
    story: str 
    poem: str 
    aggregated:str

    
def joke(state: State):
    """
    llm write a joke from the user topic
    """
    topic = state.get("topic")
    # print(state.items())
    llm = groq()
    response = llm.invoke(["human", f"""Write a simple joke about '{topic}'"""])
    # print(response)
    return {"joke": response.text}

def story(state: State):
    """
    llm write a story from the user topic
    """
    topic = state.get("topic")
    # print(state.items())
    llm = groq()
    response = llm.invoke(["human", f"""Write a simple story about '{topic}'"""])
    # print(response)
    return {"story": response.text}

def poem(state: State):
    """
    llm write a poem from the user topic
    """
    topic = state.get("topic")
    # print(state.items())
    llm = groq()
    response = llm.invoke(["human", f"""Write a simple poem about '{topic}'"""])
    # print(response)
    return {"poem": response.text}


def aggregate(state: State):
    joke = state["joke"]
    story = state["story"]
    poem = state["poem"]

    text = f"""Joke:\n{joke}\n\nStory:\n{story}\n\nPoem:\n{poem}"""
    return {"aggregated": text}


graph = StateGraph(State)
graph.add_node(joke)
graph.add_node(story)
graph.add_node(poem)
graph.add_node(aggregate)
graph.add_edge(START, "joke")
graph.add_edge(START, "story")
graph.add_edge(START, "poem")

graph.add_edge("joke", "aggregate")
graph.add_edge("story", "aggregate")
graph.add_edge("story", "aggregate")
graph.add_edge("aggregate", END)
graph = graph.compile()

async def main(query):
   if(query.strip()):
        state = graph.invoke({"topic": query})
        print(state["aggregated"])

# === Example Usage ===
if __name__ == "__main__":
    from lib.hf import run_loop
    run_loop(main, """A joke, poem and a story writer\n\nTopic? """)
