import os
import logging
import pandas as pd
from typing import List, Tuple, Any
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import ModelConfig, FileConfig
from logging_config import setup_logging
from dotenv import load_dotenv
import pprint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel

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
# --------- RAG Chain Creation ---------------
# --------------------------------------------

def create_rag_chain(vector_store: FAISS, llm: ChatOpenAI) -> RunnableParallel:
   
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = vector_store.as_retriever()
    
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# --------------------------------------------
# --------- Question Answering ---------------
# --------------------------------------------

# Add typing for input
class Question(BaseModel):
    __root__: str

def perform_question_answering(rag_chain: RunnableParallel, query: str) -> Any:
   
    if not query:
        logger.error("The query is empty.")
        raise ValueError("Query cannot be empty.")
    
    chain = rag_chain.with_types(input_type=Question)
    question = Question(__root__=query)
    return chain.invoke(question.__root__)

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

        if not raw_text:
            logger.error("No text extracted from CSV.")
            return
        
        # Split text into chunks
        texts = split_text_into_chunks(raw_text)

        # Create vector store
        document_search = create_vector_store(texts, "text-embedding-3-small")
        
        # Initialize OpenAI model
        llm = initialize_openai_model()
        
        # Create RAG chain
        rag_chain = create_rag_chain(document_search, llm)
        
        # Perform question answering
        query = input("Please enter the query: ")
        answer = perform_question_answering(rag_chain, query)
        
        # Output results
        pprint.pprint(answer)
        
    except Exception as e:
        logger.error(f"An error occurred in the main execution flow: {e}")
        print(f"An error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
