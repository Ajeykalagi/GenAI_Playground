# Install the necessary packages using the following commands:
# pip install langchain-openai

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

def initialize_openai_model():
    """Initialize the OpenAI model with the API key from the environment."""
    return ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")

llm = initialize_openai_model()

# --------------------------------------------
# --------- Define Prompt Template -----------
# --------------------------------------------

def define_prompt_template():
    """Define the prompt template for generating social media scripts."""
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

def generate_script(content_type, topic, title, category, audience, format, duration):
    """Generate a script for social media using LangChain."""
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

        script, cb = generate_script(content_type, topic, title, category, audience, format, duration)
        print("\nGenerated Script:\n", script)
        print("\nAPI Usage Information:\n", cb)
    except Exception as e:
        print(f"An error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
