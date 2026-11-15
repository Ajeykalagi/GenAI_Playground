from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.callbacks import get_openai_callback

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

csv_path = 'input/Passenger_Data_Analysis.csv'

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)


def initialize_openai_model():
    """Initialize the OpenAI model with the API key from the environment."""
    return ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")


llm = initialize_openai_model()
# --------------------------------------------
# --------- CSV Data Processing --------------
# --------------------------------------------

def read_csv_data(file_path):
    """Read CSV data from a given file path and convert to string."""
    try:
        csv_data = pd.read_csv(file_path)
        return csv_data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

df = read_csv_data(csv_path)

# --------------------------------------------
# --------- Text Splitting -------------------
# --------------------------------------------

def split_text_into_chunks(text, chunk_size=800, chunk_overlap=200):
    """Split text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

raw_text = df.to_string(index=False)
texts = split_text_into_chunks(raw_text)

# --------------------------------------------
# --------- Embeddings and Vector Store ------
# --------------------------------------------

def create_vector_store(texts, embeddings_model):
    """Create a vector store from text chunks using specified embeddings model."""
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    return FAISS.from_texts(texts, embeddings)

document_search = create_vector_store(texts, "text-embedding-3-small")

# --------------------------------------------
# --------- Agent Setup ----------------------
# --------------------------------------------

class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")

TEMPLATE = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

<df>
{dhead}
</df>

You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
You also do not have use only the information here to answer questions - you can run intermediate queries to do exploratory data analysis to give you more information as needed.

You have a tool called `person_name_search` through which you can lookup a person by name and find the records corresponding to people with similar name as the query.
You should only really use this if your search term contains a persons name. Otherwise, try to solve it with code.

For example:

<question>How old is Jane?</question>
<logic>Use `person_name_search` since you can use the query `Jane`</logic>

<question>Who has id 320</question>
<logic>Use `python_repl` since even though the question is about a person, you don't know their name so you can't include it.</logic>
"""  # noqa: E501

template = TEMPLATE.format(dhead=df.head().to_markdown())

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}"),
    ]
)

repl = PythonAstREPLTool(
    locals={"df": df},
    name="python_repl",
    description="Runs code and returns the output of the final line",
    args_schema=PythonInputs,
)

retriever_tool = create_retriever_tool(
    document_search.as_retriever(), "person_name_search", "Search for a person by name"
)

tools = [repl, retriever_tool]

agent = OpenAIFunctionsAgent(
    llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"), prompt=prompt, tools=tools
)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, max_iterations=5, early_stopping_method="generate"
) | (lambda x: x["output"])

# Typing for playground inputs

class AgentInputs(BaseModel):
    input: str

agent_executor = agent_executor.with_types(input_type=AgentInputs)

# --------------------------------------------
# --------- Question Answering ---------------
# --------------------------------------------

def perform_question_answering(llm, document_search, query):
    """Perform question answering based on the provided query and document search."""
    docs = document_search.similarity_search(query)
    with get_openai_callback() as cb:
        result = agent_executor({"input": query}, return_only_outputs=True)
    return result, cb

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main():
    """Main function to interact with the user and perform question answering."""
    try:
        query = input("Please enter the query: ")
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
