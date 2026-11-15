from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Step 1: Prepare the Documents
docs = [
    {"text": "Cats are small, carnivorous mammals that are often kept as pets. They are known for their independence and agility."},
    {"text": "Dogs are loyal and social animals that are often kept as pets. They are known for their companionship and trainability."},
    {"text": "Birds are a group of warm-blooded vertebrates that are known for their ability to fly. They have feathers, beaks, and lay eggs."},
    {"text": "Fish are cold-blooded vertebrates that live in water. They have gills that allow them to breathe underwater."}
]

# Convert the dictionaries to Document objects
documents = [Document(page_content=d["text"]) for d in docs]

# Step 2: Create an embedding function and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Step 3: Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Retrieve the top 2 most similar documents

# 1. Using invoke(query)
query = "cat"
relevant_docs = retriever.invoke(query)
print("get_relevant_documents result:")
for doc in relevant_docs:
    print(doc.page_content)
print("\n" + "-"*40 + "\n")

# 2. Using map() to process multiple queries
queries = ["cat", "dog", "bird"]
retrieved_docs = retriever.map().invoke(queries)
print("map() result:")
for query, doc_list in zip(queries, retrieved_docs):
    print(f"Results for {query}:")
    for doc in doc_list:
        print(doc.page_content)
    print("\n" + "-"*40 + "\n")