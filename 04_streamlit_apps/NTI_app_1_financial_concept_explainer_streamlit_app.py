# Importing Necessary Libraries
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Step 2: Load Environment Variables
load_dotenv()

# Step 3: Initialize the Model and Parser
# Streamlit Sidebar for Model Selection
model_name = st.sidebar.selectbox("Select Model", ["gemini-1.5-pro-001", "gemini-1.5-flash"])

chat_model = ChatGoogleGenerativeAI(model=model_name)
parser = StrOutputParser()

# Step 4: Create the Streamlit Interface
st.title("Financial Assistance Chatbot")
st.write("This chatbot is designed to help you with financial assistance questions in a simple and easy-to-understand manner.")

# Step 5: Accept User Input
user_query = st.text_input("Enter your question:")

# Step 6: Process the Input and Generate a Response
if user_query:
    # Prepare the messages to be sent to the model
    messages = [
        SystemMessage(content="You are an expert in financial assistance, skilled in explaining things in a simple and easy-to-understand manner."),
        HumanMessage(content=user_query),
    ]
    # Generate the response from the model
    chat_model_response = chat_model.invoke(messages)
    # Extract and format the response
    formatted_response = parser.invoke(chat_model_response)
    # Display the response
    st.write("**Question:**", user_query)
    st.write("**Answer:**", formatted_response)

    # Step 7: Displaying Raw Responses (Optional)
    st.write("**Raw Answer:**", chat_model_response)
