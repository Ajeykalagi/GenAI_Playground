import os
import pprint
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

# =====================
# 1. Environment Setup
# =====================
# Setting environment variables for LangChain tracing and API keys
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "Advanced_RAG"

# =====================
# 2. File Setup
# =====================
# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the name of the data folder and the file
data_folder = 'data/olympics'
file_name = 'Olympic_History_Part_1.txt'

# Construct the file path
file_path = os.path.join(script_directory, data_folder, file_name)

# =====================
# 3. Load the Data
# =====================
loader = TextLoader(file_path)
data_docs = loader.load()

# ================================
# 4. Split the Data into Chunks
# ================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # Set a small chunk size for demonstration
    chunk_overlap=20,     # Overlap between chunks to maintain context
    length_function=len,  # Function to measure the length of the chunks
    is_separator_regex=False  # Treat the separator as a fixed string
)

# Make splits
splits = text_splitter.split_documents(data_docs)

# ===================================
# 5. Create and Index Embeddings
# ===================================
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

# ==============================================
# 6. Define Prompt Templates for Few-Shot Learning
# ==============================================
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Examples for few-shot learning
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]

# Transform examples to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["question"]
)

# Create the prompt template for generating step-back questions
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)

# =======================================================
# 7. Generate Step-Back Queries and Retrieve Information
# =======================================================
generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
user_query = "what do you know of Panagiotis Soutsos"
step_back_chain = generate_queries_step_back | retriever
# generate_queries_step_back.invoke({"question": user_query})

# ==================================
# 8. Define Response Prompt Template
# ==================================
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

# ===================================================
# 9. Create Chain for Processing and Generating Response
# ===================================================


# Retrieve context for both normal and step-back queries in one go
step_back_context = "\n\n".join([doc.page_content for doc in retriever.invoke(generate_queries_step_back.invoke({"question": user_query}))])
normal_context = "\n\n".join([doc.page_content for doc in retriever.invoke(user_query)])

# Construct the inputs dictionary
inputs = {
    "normal_context": normal_context,
    "step_back_context": step_back_context,  # This is now the resolved context
    "question": user_query
}
chain = (
    response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

response = chain.invoke(inputs)

# ======================
# 11. Print the Response
# ======================
print(response)


# chain = (
#     {
#         "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
#         "step_back_context": generate_queries_step_back | retriever,
#         "question": itemgetter("question")
#     } 
#     | response_prompt
#     | ChatOpenAI(temperature=0)
#     | StrOutputParser()
# )

# response = chain.invoke({"question": user_query})

# # ======================
# # 11. Print the Response
# # ======================
# print(response)