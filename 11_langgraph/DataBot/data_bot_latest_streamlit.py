import os
import pandas as pd
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------
# --------- Setup and Configuration ----------

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------
# --------- Initialize OpenAI Model ----------

def initialize_openai_model():
    model_name = "gpt-4o-mini"
    temperature = 0.5
    return ChatOpenAI(temperature=temperature, model_name=model_name)

# --------------------------------------------
# --------- Read CSV Data -------------------

def read_csv_data(file_path):
    csv_data = pd.read_csv(file_path)
    return csv_data.to_string(index=False)

# --------------------------------------------
# --------- Split Text into Chunks ----------

def split_text_into_chunks(text, chunk_size=800, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

# --------------------------------------------
# --------- Create FAISS Vector Store --------

def create_vector_store(texts, embeddings_model):
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    return FAISS.from_texts(texts, embeddings)

# --------------------------------------------
# --------- RAG Chain Creation ---------------

def create_rag_chain(vector_store, llm):
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = vector_store.as_retriever()
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# --------------------------------------------
# --------- Question Answering ---------------

def perform_question_answering(rag_chain, query):
    if not query:
        raise ValueError("Query cannot be empty.")
    
    return rag_chain.invoke(query)

# --------------------------------------------
# --------- Main Function for Streamlit ------

def main():
    """Main function to interact with the user and perform question answering via Streamlit."""
    
    st.title("CSV Data Question Answering System")

    # File uploader to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read and display CSV data
        st.write("CSV Data:")
        raw_text = read_csv_data(uploaded_file)

        if not raw_text:
            st.error("No text extracted from CSV.")
            return
        
        st.text_area("Extracted Text from CSV:", raw_text, height=300)
        
        # Split text into chunks
        texts = split_text_into_chunks(raw_text)
        
        # Create vector store
        document_search = create_vector_store(texts, "text-embedding-3-small")
        
        # Initialize OpenAI model
        llm = initialize_openai_model()
        
        # Create RAG chain
        rag_chain = create_rag_chain(document_search, llm)
        
        # Input box for the user to enter a query
        user_query = st.text_input("Enter your query:")

        if user_query:
            try:
                # Perform question answering
                answer = perform_question_answering(rag_chain, user_query)
                st.write(f"Answer: {answer}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
if __name__ == "__main__":
    main()
