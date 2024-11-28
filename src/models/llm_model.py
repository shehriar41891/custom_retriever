from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

def get_llm():
    # load the environement variables
    load_dotenv()
    # Retrieve OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API")
    
    print('The openai api key is ',openai_api_key)
    """
    Returns an instance of OpenAI's LLM initialized with Langchain.
    """
    if not openai_api_key:
        raise EnvironmentError("OpenAI API key is not set in your environment variables.")
    return OpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7,api_key=openai_api_key)
