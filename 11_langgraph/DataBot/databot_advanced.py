import os
import logging
import pandas as pd
from typing import List, Tuple, Any
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from config import ModelConfig, FileConfig
from logging_config import setup_logging
from dotenv import load_dotenv
import pprint

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
# --------- Read CSV Data -------------------
# --------------------------------------------

def read_csv_data(file_path: str) -> str:
    
    try:
        csv_data = pd.read_csv(file_path)
        return csv_data.to_string(index=False)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return ""

# --------------------------------------------
# --------- Split Text into Chunks ----------
# --------------------------------------------

def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
   
    if not text:
        logger.error("The text to split is empty.")
        raise ValueError("Text cannot be empty.")
    if chunk_size <= 0:
        logger.error("Chunk size must be positive.")
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        logger.error("Chunk overlap must be non-negative.")
        raise ValueError("chunk_overlap must be non-negative")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

# --------------------------------------------
# --------- Create FAISS Vector Store --------
# --------------------------------------------

def create_vector_store(texts: List[str], embeddings_model: str) -> FAISS:
  
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    return FAISS.from_texts(texts, embeddings)

# --------------------------------------------
# --------- Perform Question Answering -------
# --------------------------------------------

def perform_question_answering(llm: ChatOpenAI, document_search: FAISS, query: str) -> Tuple[str, Any]:
   
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = document_search.similarity_search(query)
    with get_openai_callback() as cb:
        result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return result['text'], cb

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main():
    """Main function to interact with the user and perform question answering."""
    try:
        # Define file path and read data
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            FileConfig.INPUT_DATA_FOLDER_NAME.value,
            FileConfig.FILE_NAME.value
        )
        raw_text = read_csv_data(file_path)

        # Split text into chunks
        texts = split_text_into_chunks(raw_text)

        # Create vector store
        document_search = create_vector_store(texts, "text-embedding-3-small")

        # Initialize OpenAI model
        llm = initialize_openai_model()

        # Perform question answering
        query = input("Please enter the query: ")
        result, callback_info = perform_question_answering(llm, document_search, query)

        # Output results
        pprint.pprint(result)
        pprint.pprint(callback_info)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
