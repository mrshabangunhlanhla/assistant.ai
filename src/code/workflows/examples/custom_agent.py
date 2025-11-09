from lib.agents import OptimalReActAgent
from lib.models import groq, googleAI
from lib.tools import calculate, run_shell_command, google_search
import pyttsx3

# Initialize the TTS engine


tools = [calculate, run_shell_command, google_search]

agent = OptimalReActAgent(llm=groq(), tools=tools)
async def main(query: str):
    """Demonstrates asynchronous streaming."""

        
    # Set properties (optional)
    # You can change the voice, rate, or volume
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[1].id) # Change to a different voice if available
    # engine.setProperty('rate', 150) # Speed of speech (words per minute)
    # engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)

    # Convert text to speech
    # engine.say("Hello, this is a test of text-to-speech in Python.")

    # # Play the speech and wait for it to finish
    # engine.runAndWait()

    # # Stop the engine (important for releasing resources)
    # engine.stop()
    engine = pyttsx3.init()
    async for event in agent.stream(query):
        
        event_type = event["type"]
        content = event["content"]
        if event_type == "final_answer":
            engine.say(content)
            print(f"AI: {content}\n")
            engine.runAndWait()

            # Stop the engine (important for releasing resources)
            engine.stop()


        # if event_type == "thought":
        #     print(f" [THOUGHT]: {content}")
        # elif event_type == "action_input":
        #     print(f" [ACTION]: {content}")
        # elif event_type == "observation":
        #     print(f" [OBSERVATION]: {content}")
        # elif event_type == "final_answer":
        #     # print(f"AI: {content}\n")
        #     print(f"\n<<< FINAL ANSWER >>>\n{content}\n<<< /FINAL ANSWER >>>")
        # elif event_type == "error":
        #     print(f"\n[!!! AGENT ERROR !!!] {content}")
        # elif event_type == "info":
        #     print(f"\n{content}")


if __name__ == "__main__":
    from lib.hf import run_loop
    print("==========================================")
    print("              Custom Writter              ")
    print("==========================================\n")
    run_loop(main, ">> ")
