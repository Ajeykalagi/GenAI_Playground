import os
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pickle

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "langchain-first-project"

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

data_folder = 'processed'

# retrieve with 100
file_name = 'chunks_100_20.pkl'

# # retrieve with 500
# file_name = 'chunks_500_50.pkl'

# Construct the file path
file_path = os.path.join(script_directory, data_folder, file_name)

with open(file_path, "rb") as file:
    chunks = pickle.load(file)

# Create embeddings for the chunks using OpenAIEmbeddings
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=OpenAIEmbeddings()
)

# Create a retriever to search for relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

user_query= "what do you know of Panagiotis Soutsos"

# Retrieve documents relevant to the query
docs = retriever.invoke(user_query)

# Define the prompt template for the LLM
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Chain
chain = prompt | llm | StrOutputParser()

# Run
response = chain.invoke({"context": docs, "question": user_query})

print(response)