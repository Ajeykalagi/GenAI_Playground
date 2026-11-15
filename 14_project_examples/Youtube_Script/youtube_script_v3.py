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
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel

# Set up logging configuration
setup_logging()
logger = logging.getLogger(__name__)

# Use enums for configuration
MODEL_NAME = ModelConfig.MODEL_NAME.value
TEMPERATURE = ModelConfig.TEMPERATURE.value

# Get the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found in environment variables.")
    raise ValueError("OpenAI API key not found in environment variables.")

def initialize_openai_model(api_key: str) -> ChatOpenAI:
    """
    Initialize the OpenAI model with the provided API key.

    Args:
        api_key (str): The OpenAI API key.

    Returns:
        ChatOpenAI: An instance of the OpenAI model.
    """
    return ChatOpenAI(openai_api_key=api_key, temperature=TEMPERATURE, model_name=MODEL_NAME)

llm = initialize_openai_model(OPENAI_API_KEY)

def define_prompt_template() -> ChatPromptTemplate:
    """
    Define the prompt template for generating social media scripts.

    Returns:
        ChatPromptTemplate: The defined prompt template.
    """
    return ChatPromptTemplate.from_template(
        "You are a social media script writer. Write a {format} script for {content_type} about {topic}. "
        "The title is '{title}', the category is '{category}', the target audience includes {audience}, and "
        "the duration is {duration} seconds."
    )

prompt = define_prompt_template()

class ScriptInput(BaseModel):
    content_type: str
    topic: str
    title: str
    category: str
    audience: str
    format: str
    duration: int

# Create the chain
chain = (
    RunnableParallel({
        "content_type": RunnablePassthrough(),
        "topic": RunnablePassthrough(),
        "title": RunnablePassthrough(),
        "category": RunnablePassthrough(),
        "audience": RunnablePassthrough(),
        "format": RunnablePassthrough(),
        "duration": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

def generate_script(script_input: ScriptInput) -> Dict[str, Any]:
    """
    Generate a script for social media using LangChain.

    Args:
        script_input (ScriptInput): The input parameters for the script.

    Returns:
        Dict[str, Any]: The generated script and callback information.
    """
    with get_openai_callback() as cb:
        response = chain.invoke(script_input)
    return response.strip(), cb

def main():
    """Main function to interact with the user and generate a social media script."""
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

        script_input = ScriptInput(
            content_type=content_type,
            topic=topic,
            title=title,
            category=category,
            audience=', '.join(audience),
            format=format,
            duration=duration
        )

        script, cb = generate_script(script_input)
        print("\nGenerated Script:\n")
        pprint.pprint(script)
        print("\nAPI Usage Information:\n")
        pprint.pprint(cb)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
