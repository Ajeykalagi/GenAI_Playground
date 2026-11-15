

import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import asyncio

# Load the HR policy PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

hr_policy_text = extract_text_from_pdf("input/HR Policy Handbook for LogiCom.pdf")

# Load OpenAI model and embeddings
llm = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

# Split text into chunks and create embeddings
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_chunks = text_splitter.split_text(hr_policy_text)
vector_store = FAISS.from_texts(text_chunks, embeddings)

# Create a Conversational Retrieval Chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store.as_retriever(), memory=memory)

# Define the chatbot logic
async def main():
    print("Welcome to the HR Bot. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("HR Bot: Goodbye!")
            break
        response = await qa_chain.ainvoke({"question": user_input})
        print(f"HR Bot: {response}")

if __name__ == "__main__":
    asyncio.run(main())
