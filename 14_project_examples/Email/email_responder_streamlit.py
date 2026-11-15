# Install the necessary packages using the following commands:
# pip install langchain-openai streamlit

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# --------------------------------------------
# --------- Set your OpenAI API key ----------
# --------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --------------------------------------------
# --------- Initialize OpenAI Model ----------
# --------------------------------------------

def initialize_openai_model(api_key):
    """
    Initialize the OpenAI model with the provided API key.
    """
    return ChatOpenAI(openai_api_key=api_key, temperature=0.5, model_name="gpt-4o-mini")

llm = initialize_openai_model(OPENAI_API_KEY)

# --------------------------------------------
# --------- Define Prompt Template -----------
# --------------------------------------------

def define_prompt_template():
    """
    Define the prompt template for generating email responses.
    """
    return ChatPromptTemplate.from_messages([
        ("user", "You are an assistant that generates email responses. Here is the email you need to respond to:\n\n"
                 "{email_text}\n\nGenerate a {response_length} response.")
    ])

prompt = define_prompt_template()

# Create the chain
chain = prompt | llm | StrOutputParser()

# --------------------------------------------
# --------- Response Generation Function -----
# --------------------------------------------

def generate_response(email_text, response_length):
    """
    Generate a response to the provided email text using LangChain.
    """
    with get_openai_callback() as cb:
        response = chain.invoke({"email_text": email_text, "response_length": response_length})
    return response.strip(), cb

# --------------------------------------------
# --------- Streamlit UI Setup ---------------
# --------------------------------------------

st.title("Email Response Generator")

email_text = st.text_area("Paste the email text:")
response_length = st.selectbox("Select response length:", ['short', 'medium', 'long'])

if st.button("Generate Response"):
    if not email_text:
        st.error("Please enter an email text to generate a response.")
    elif response_length not in ['short', 'medium', 'long']:
        st.error("Invalid response length. Please choose from 'short', 'medium', or 'long'.")
    else:
        response, cb = generate_response(email_text, response_length)
        st.subheader("Generated Response:")
        st.write(response)
        st.subheader("API Usage:")
        st.write(cb)

