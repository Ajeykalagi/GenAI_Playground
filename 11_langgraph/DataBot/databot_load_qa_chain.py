# Install the necessary packages using the following commands:
# pip install -qU langchain-openai pandas langchain-community faiss-cpu

import os
import langchain
import pandas as pd
from typing import List, Tuple, Any, Dict
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

langchain.debug = True

def initialize_openai_model() -> ChatOpenAI:
    """Initialize the OpenAI model with the API key from the environment."""
    return ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")

llm: ChatOpenAI = initialize_openai_model()

# --------------------------------------------
# --------- CSV Data Processing --------------
# --------------------------------------------

def read_csv_data(file_path: str) -> str:
    """Read CSV data from a given file path and convert to string."""
    try:
        csv_data: pd.DataFrame = pd.read_csv(file_path)
        return csv_data.to_string(index=False)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return ""

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the name of the data folder and the file
data_folder = 'input'
file_name = 'Passenger_Data_Analysis.csv'

# Construct the file path
csv_path = os.path.join(script_directory, data_folder, file_name)

# Provide the path of the CSV file
# csv_path: str = 'input/Passenger_Data_Analysis.csv'
raw_text: str = read_csv_data(csv_path)

# --------------------------------------------
# --------- Text Splitting -------------------
# --------------------------------------------

def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """Split text into manageable chunks."""
    text_splitter: CharacterTextSplitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

texts: List[str] = split_text_into_chunks(raw_text)

# --------------------------------------------
# --------- Embeddings and Vector Store ------
# --------------------------------------------

def create_vector_store(texts: List[str], embeddings_model: str) -> FAISS:
    """Create a vector store from text chunks using specified embeddings model."""
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=embeddings_model)
    return FAISS.from_texts(texts, embeddings)

document_search: FAISS = create_vector_store(texts, "text-embedding-3-small")

# --------------------------------------------
# --------- Question Answering ---------------
# --------------------------------------------

def perform_question_answering(llm: ChatOpenAI, document_search: FAISS, query: str) -> Tuple[Dict[str, Any], Any]:
    """Perform question answering based on the provided query and document search."""
    chain = load_qa_chain(llm, chain_type="stuff")
    docs: List[Dict[str, Any]] = document_search.similarity_search(query)
    with get_openai_callback() as cb:
        result: Dict[str, Any] = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return result, cb

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main() -> None:
    """Main function to interact with the user and perform question answering."""
    try:
        query: str = input("Please enter the query: ")
        result, callback_info = perform_question_answering(llm, document_search, query)
        print("### Answer:")
        print(result)
        print("### API Usage:")
        print(callback_info)
    except Exception as e:
        print(f"An error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
