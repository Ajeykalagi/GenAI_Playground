from langchain_community.document_loaders import WebBaseLoader
import uuid
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
import os
import pprint

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "prj_Indexing"

# ===================================
# 1. Load the Documents
# ===================================
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())

pprint.pprint(docs)

# ===================================
# 2. Summarize the Documents
# ===================================
# WAY : 1
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOpenAI(model="gpt-4o-mini", max_retries=0)
    | StrOutputParser()
)
summaries = chain.batch(docs, {"max_concurrency": 5})

# WAY : 2
# chain = (
#     (lambda x: {"doc": x.page_content})  # Extract the content of the document using attribute access
#     | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
#     | ChatOpenAI(model="gpt-4o-mini", max_retries=0)
#     | StrOutputParser()
# )
# summaries = list(map(chain.invoke, docs))

# WAY : 3
# chain = (
#     {"doc": lambda x: x.page_content}  # Extract the content of the document
#     | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
#     | ChatOpenAI(model="gpt-4o-mini", max_retries=0)
#     | StrOutputParser()
# )
# summaries = [chain.invoke(doc) for doc in docs]

pprint.pprint(summaries)


# ===================================
# 3. Set Up Multi-Representation Indexing
# ===================================
# The vectorstore to use to index the summaries (child chunks)
vectorstore = Chroma(collection_name="summaries",
                     embedding_function=OpenAIEmbeddings())

# The storage layer for the full documents (parent documents)
store = InMemoryByteStore()
id_key = "doc_id"

# The retriever that combines the vector store and document store
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# ===================================
# 4. Link Summaries with Full Documents
# ===================================

doc_ids = [str(uuid.uuid4()) for _ in docs]

# Create Document objects for the summaries, linking them with the full documents
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# Add summaries to the vector store and full documents to the document store
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# ===================================
# 5. Query the Indexed Data
# ===================================

query = "Memory in agents"

# Perform a similarity search in the vector store to find relevant summaries
sub_docs = vectorstore.similarity_search(query, k=1)
print(sub_docs[0])

# Retrieve the full document linked to the found summary. But this is totally separate from above found summary
retrieved_docs = retriever.invoke(query, n_results=1)
print(retrieved_docs[0].page_content[0:500])
