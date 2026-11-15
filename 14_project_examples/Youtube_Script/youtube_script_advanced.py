import os
import logging
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import pprint
from config import ModelConfig
from logging_config import setup_logging
from dotenv import load_dotenv

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
setup_logging()
logger = logging.getLogger(__name__)

# Use enums for configuration
MODEL_NAME = ModelConfig.MODEL_NAME.value
TEMPERATURE = ModelConfig.TEMPERATURE.value

def initialize_openai_model() -> ChatOpenAI:
    # Initialize the OpenAI model with configuration
    return ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME)

llm = initialize_openai_model()

# --------------------------------------------
# --------- Define Prompt Template -----------
# --------------------------------------------

def define_prompt_template() -> ChatPromptTemplate:
    # Define the prompt template for generating social media scripts
    return ChatPromptTemplate.from_messages([
        ("user", "You are a social media script writer. Write a {format} script for {content_type} about {topic}. "
                 "The title is '{title}', the category is '{category}', the target audience includes {audience}, and "
                 "the duration is {duration} seconds."),
    ])

prompt = define_prompt_template()

# Create the chain
chain = prompt | llm | StrOutputParser()

# --------------------------------------------
# --------- Script Generation Function -------
# --------------------------------------------

def generate_script(content_type: str, topic: str, title: str, category: str, audience: List[str], format: str, duration: int) -> Dict[str, Any]:
    # Generate a script for social media using LangChain
    with get_openai_callback() as cb:
        response = chain.invoke({
            "content_type": content_type,
            "topic": topic,
            "title": title,
            "category": category,
            "audience": ', '.join(audience),
            "format": format,
            "duration": duration
        })
    return response.strip(), cb

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main():
    # Main function to interact with the user and generate a social media script
    try:
        content_type = input("Select content type (YouTube or Instagram):\n").lower()
        if content_type not in ['youtube', 'instagram']:
            print("Invalid content type. Please choose 'YouTube' or 'Instagram'.")
            return

        title = input("Enter the title of the script:\n")
        category = input("Enter the category of the script:\n")
        audience = input("Enter the target audience (comma-separated):\n").split(',')
        format = input("Select format (Reel or Video):\n").lower()
        if format not in ['reel', 'video']:
            print("Invalid format. Please choose 'Reel' or 'Video'.")
            return

        duration = int(input("Enter the duration of the script in seconds:\n"))
        topic = input("Enter the topic for the script:\n")

        script, cb = generate_script(content_type, topic, title, category, audience, format, duration)
        print("\nGenerated Script:\n", script)
        print("\nAPI Usage Information:\n", cb)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
