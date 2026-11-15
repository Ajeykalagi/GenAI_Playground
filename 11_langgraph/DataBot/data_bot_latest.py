import os
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import pprint
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------
# --------- Initialize OpenAI Model ----------
# --------------------------------------------

def initialize_openai_model():
    model_name = "gpt-4o-mini"
    temperature = 0.5

    return ChatOpenAI(temperature=temperature, model_name=model_name)

# --------------------------------------------
# --------- Read CSV Data -------------------
# --------------------------------------------

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
# --------- Main Function --------------------

def main():
    """Main function to interact with the user and perform question answering."""
    
    # Define file path and read data
    # Default file settings
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_folder = 'input'
    default_file_name = 'Passenger_Data_Analysis.csv'
    file_path = os.path.join(script_directory, data_folder, default_file_name)


    # file_path = "input/Passenger_Data_Analysis.csv"
    raw_text = read_csv_data(file_path)

    if not raw_text:
        print("No text extracted from CSV.")
        return
    
    # Split text into chunks
    texts = split_text_into_chunks(raw_text)

    # Create vector store
    document_search = create_vector_store(texts, "text-embedding-3-small")
    
    # Initialize OpenAI model
    llm = initialize_openai_model()
    
    # Create RAG chain
    rag_chain = create_rag_chain(document_search, llm)
    
    # Continuously ask questions until the user decides to exit
    while True:
        query = input("Please enter the query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the question-answering loop. Goodbye!")
            break
        
        try:
            answer = perform_question_answering(rag_chain, query)
            pprint.pprint(answer)
        except Exception as e:
            print(f"An error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
