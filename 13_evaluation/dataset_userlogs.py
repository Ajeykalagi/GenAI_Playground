# Create a new project where user questions are logged
import os

os.environ["LANGCHAIN_PROJECT"] = "DBRX"

# Load blog post
import requests
from bs4 import BeautifulSoup

# URL of the blog post to be scraped
url = "https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm"

# Send a GET request to the URL
response = requests.get(url)

# Parse the response content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Extract all paragraph texts from the blog post
text = [p.text for p in soup.find_all("p")]
full_text = "\n".join(text)

# OpenAI API
import openai
from langsmith.wrappers import wrap_openai

# Wrap the OpenAI client for LangSmith
openai_client = wrap_openai(openai.Client())

def answer_dbrx_question_oai(inputs: dict) -> dict:
    """
    Generates answers to user questions based on a provided website text using the OpenAI API.

    Parameters:
    inputs (dict): A dictionary with a single key 'question', representing the user's question as a string.

    Returns:
    dict: A dictionary with a single key 'answer', containing the generated answer as a string.
    """

    # System prompt containing the full text from the blog post
    system_msg = f"Answer user questions in 2-3 sentences about this context: \n\n\n {full_text}"

    # Prepare messages for the OpenAI API
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": inputs["question"]},
    ]

    # Call OpenAI API to get the response
    response = openai_client.chat.completions.create(
        messages=messages, model="gpt-4o-mini"
    )

    # Extract and return the generated answer
    return {"answer": response.dict()["choices"][0]["message"]["content"]}

# Example user question
answer_dbrx_question_oai({
    "question": "What are the main differences in training efficiency between MPT-7B vs DBRX?"
})

# Another example user question
answer_dbrx_question_oai({"question": "How many tokens was DBRX pre-trained on?"})
