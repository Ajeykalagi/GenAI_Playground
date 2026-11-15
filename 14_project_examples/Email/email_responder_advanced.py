import os
import logging
from typing import Tuple, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from config import ModelConfig
from logging_config import setup_logging
from dotenv import load_dotenv

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Set up logging configuration
setup_logging()
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------
# --------- Initialize OpenAI Model ----------
# --------------------------------------------

def initialize_openai_model() -> ChatOpenAI:
   
    model_name = ModelConfig.MODEL_NAME.value
    temperature = ModelConfig.TEMPERATURE.value

    if not model_name or not isinstance(model_name, str):
        logger.error("Invalid model name provided.")
        raise ValueError("Invalid model name provided.")
    if not (0 <= temperature <= 1):
        logger.error("Temperature must be between 0 and 1.")
        raise ValueError("Temperature must be between 0 and 1.")
    
    return ChatOpenAI(temperature=temperature, model_name=model_name)

# --------------------------------------------
# --------- Define Prompt Template ----------
# --------------------------------------------

def define_prompt_template() -> ChatPromptTemplate:
    
    return ChatPromptTemplate.from_messages([
        ("user", "You are an assistant that generates email responses. Here is the email you need to respond to:\n\n"
                 "{email_text}\n\nGenerate a {response_length} response.")
    ])

# Initialize components
llm = initialize_openai_model()  # OpenAI model instance
prompt = define_prompt_template()  # Prompt template instance
chain = prompt | llm | StrOutputParser()  # Chain setup

# --------------------------------------------
# --------- Generate Email Response ---------
# --------------------------------------------

def generate_response(email_text: str, response_length: str) -> Tuple[str, Any]:
   
    if response_length not in ['short', 'medium', 'long']:
        logger.error(f"Invalid response length provided: {response_length}")
        raise ValueError("Response length must be one of 'short', 'medium', or 'long'.")
    
    with get_openai_callback() as cb:
        response = chain.invoke({"email_text": email_text, "response_length": response_length})
    
    return response.strip(), cb

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main():
    
    try:
        email_text = input("Paste the email text:\n")  # User input for email text
        response_length = input("Select response length (short, medium, long):\n")  # User input for response length
        
        if response_length not in ['short', 'medium', 'long']:
            print("Invalid response length. Please choose from 'short', 'medium', or 'long'.")
            return
        
        response, cb = generate_response(email_text, response_length)  # Generate response
        print("\nGenerated Response:\n", response)  # Display generated response
        print("\nAPI Usage:\n", cb)  # Display API usage information
    except ValueError as ve:
        logger.error(f"Value error: {ve}")
        print(f"Value error: {ve}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
