# Install the necessary packages using the following commands:
# pip install langchain-openai streamlit pandas
import langchain
import streamlit as st
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback

# --------------------------------------------
# --------- Set your OpenAI API key ----------
# --------------------------------------------
# TODO: create a left menu bar and have provision to enter key there. pre-populate key for runnning locally purpose.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
langchain.debug = True
# TODO: remove debug

# --------------------------------------------
# --------- Streamlit UI Setup ---------------
# --------------------------------------------

st.title("CSV Data Question Answering")

# --------------------------------------------
# --------- Initialize OpenAI Model ----------
# --------------------------------------------

def initialize_openai_model(api_key):
    return ChatOpenAI(openai_api_key=api_key, temperature=0.5, model_name="gpt-4o-mini")

llm = initialize_openai_model(OPENAI_API_KEY)

# --------------------------------------------
# --------- File Upload Handling -------------
# --------------------------------------------

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# --------------------------------------------
# --------- CSV Data Processing --------------
# --------------------------------------------

def read_csv_data(uploaded_file):
    try:
        csv_data = pd.read_csv(uploaded_file)
        return csv_data.to_string(index=False)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return ""

if uploaded_file:
    raw_text = read_csv_data(uploaded_file)

    # --------------------------------------------
    # --------- Text Splitting Function ---------
    # --------------------------------------------

    def split_text_into_chunks(text, chunk_size=800, chunk_overlap=200):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_text(text)

    texts = split_text_into_chunks(raw_text)

    # --------------------------------------------
    # --------- Vector Store Creation -----------
    # --------------------------------------------

    def create_vector_store(texts, embeddings_model, api_key):
        embeddings = OpenAIEmbeddings(model=embeddings_model, openai_api_key=api_key)
        return FAISS.from_texts(texts, embeddings)

    document_search = create_vector_store(texts, "text-embedding-3-small", OPENAI_API_KEY)

    # --------------------------------------------
    # --------- Query Input and Processing ------
    # --------------------------------------------

    query = st.text_input("Please enter your query:")

    if st.button("Answer Query"):
        # ----------------------------------------
        # --------- Question Answering -----------
        # ----------------------------------------

        def perform_question_answering(llm, document_search, query):
            chain = load_qa_chain(llm, chain_type="stuff")
            docs = document_search.similarity_search(query)
            with get_openai_callback() as cb:
                result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
            return result, cb

        result, callback_info = perform_question_answering(llm, document_search, query)

        # Display results
        st.subheader("Answer:")
        st.write(result)
        st.subheader("API Usage:")
        st.write(callback_info)
