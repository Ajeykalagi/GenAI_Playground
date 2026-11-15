# Install the necessary packages using the following commands:
# pip install langchain-openai

import os
from typing import Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

def initialize_openai_model() -> ChatOpenAI:
    """
    Initialize the OpenAI model with the API key from the environment.
    
    Returns:
        ChatOpenAI: The initialized OpenAI model.
    """
    return ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")

llm: ChatOpenAI = initialize_openai_model()

# --------------------------------------------
# --------- Define Prompt Template -----------
# --------------------------------------------

def define_prompt_template() -> ChatPromptTemplate:
    """
    Define the prompt template for generating email responses.
    
    Returns:
        ChatPromptTemplate: The defined prompt template.
    """
    return ChatPromptTemplate.from_messages([
        ("user", "You are an assistant that generates email responses. Here is the email you need to respond to:\n\n"
                 "{email_text}\n\nGenerate a {response_length} response.")
    ])

prompt: ChatPromptTemplate = define_prompt_template()

# Create the chain
chain = prompt | llm | StrOutputParser()

# --------------------------------------------
# --------- Response Generation Function -----
# --------------------------------------------

def generate_response(email_text: str, response_length: str) -> Tuple[str, Any]:
    """
    Generate a response to the provided email text using LangChain.
    
    Parameters:
        email_text (str): The email text to respond to.
        response_length (str): The length of the response ('short', 'medium', 'long').
    
    Returns:
        Tuple[str, Any]: A tuple containing the generated response (str) and OpenAI callback object.
    """
    with get_openai_callback() as cb:
        response: str = chain.invoke({"email_text": email_text, "response_length": response_length})
    return response.strip(), cb

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main() -> None:
    """
    Main function to get user input, generate an email response, and display the response.
    """
    try:
        email_text: str = input("Paste the email text:\n")
        response_length: str = input("Select response length (short, medium, long):\n")
        
        if response_length not in ['short', 'medium', 'long']:
            print("Invalid response length. Please choose from 'short', 'medium', or 'long'.")
            return
        
        response, cb = generate_response(email_text, response_length)
        print("\nGenerated Response:\n", response)
        print("\nAPI Usage:\n", cb)
    except Exception as e:
        print(f"An error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
