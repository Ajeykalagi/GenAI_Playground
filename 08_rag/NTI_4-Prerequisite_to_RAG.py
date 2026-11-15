import os
import pprint
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the name of the data folder and the file
data_folder = 'data'
file_name = 'Olympics_2024.txt'

# Construct the file path
file_path = os.path.join(script_directory, data_folder, file_name)

# Print the file path to ensure it's correct
# print("File Path:", file_path)

# Check if the file exists
if not os.path.exists(file_path):
    print("File does not exist. Check the path and try again.")
else:
    # Load the document using TextLoader
    try:
        loader = TextLoader(file_path)
        data_docs = loader.load()
        # print(type(data_docs))
        # print(data_docs)
        # print("Data Loaded Successfully!")
    except Exception as e:
        print("An error occurred while loading the document:", e)


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(data_docs)
# pprint.pprint(chunks)

# Index
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=chunks, 
                                    embedding=OpenAIEmbeddings())


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

docs = retriever.invoke("When is 2024 Summer Olympics taking place?")

# print(len(docs))
# pprint.pprint(docs)




from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# way 1: Generation

# # Chain
# chain = prompt | llm | StrOutputParser()

# # Run
# response= chain.invoke({"context":docs,"question":"When is 2024 Summer Olympics taking place?"})

# print(response)

# way 2: Generation
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("When is 2024 Summer Olympics taking place?")

print(response)