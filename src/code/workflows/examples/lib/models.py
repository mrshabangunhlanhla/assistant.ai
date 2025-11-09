from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Access the environment variables
# api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Groq LLM

def groq(model_name="llama-3.3-70b-versatile"):
    return ChatGroq(
        model_name=model_name,
        temperature=0.7
    )
def googleAI(model="gemini-2.0-flash"):
    return ChatGoogleGenerativeAI(model=model)

# print(chat_completion.choices[0].message.content)