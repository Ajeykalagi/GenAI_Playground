from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# Create an embedding function
embeddings = OpenAIEmbeddings()

# Your original documents as dictionaries
docs = [
    {"text": "Cats are small, carnivorous mammals that are often kept as pets. They are known for their independence and agility."},
    {"text": "Dogs are loyal and social animals that are often kept as pets. They are known for their companionship and trainability."},
    {"text": "Birds are a group of warm-blooded vertebrates that are known for their ability to fly. They have feathers, beaks, and lay eggs."},
    {"text": "Fish are cold-blooded vertebrates that live in water. They have gills that allow them to breathe underwater."}
]

# Convert the dictionaries to Document objects
documents = [Document(page_content=d["text"]) for d in docs]

# Create a FAISS vector store with the Document objects
vectorstore = FAISS.from_documents(documents, embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Retrieve only the top 1 most similar document

# Define a list of animal names to retrieve documents for
animal_names = ["cat", "dog", "bird"]

# Use RunnablePassthrough to create a passthrough for inputs
input_transformer = RunnablePassthrough()

# Create the chain for retrieval using method chaining
retrieval_chain = (
    input_transformer
    | retriever.map()  # No direct arguments here; the chain will pass inputs
)

# Invoke the chain with your list of animal names
docs = retrieval_chain.invoke(animal_names)

# Print the retrieved documents
for animal, doc_list in zip(animal_names, docs):
    print(f"Documents for '{animal.capitalize()}':")
    if not doc_list:
        print("  No relevant documents found.")
    else:
        for idx, d in enumerate(doc_list, start=1):
            print(f"  {idx}. {d.page_content}")
    print("\n" + "-"*40 + "\n")
