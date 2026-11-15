import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langgraph.graph import MessageGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from functools import partial
import asyncio

# Load the HR policy PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
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

# Define the chatbot state
class HRState(BaseModel):
    input: str
    responses: list

# Define the HR agent
def handle_greeting(state: HRState):
    """Handle the greeting from the user."""
    user_input = state.input
    if 'hello' in user_input.lower():
        response = "Hello! How can I assist you today?"
    else:
        response = "Hi! How can I help you with your HR-related queries?"
    state.responses.append(response)
    return state

async def provide_hr_info(state: HRState):
    """Provide HR information based on user input."""
    user_input = state.input
    response = await qa_chain.ainvoke({"question": user_input})
    state.responses.append(response['answer'])
    return state

# Define the generic router
class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""
    choice: str = Field(description="should be one of: hr_info")

system_message = """Your job is to help as an HR representative for a company.

You should interact politely with employees to try to figure out how you can help. You can help in a few ways:

- Providing HR information: if an employee wants to know something about the HR policies, route them to `hr_info`.

If the user is asking or wants to ask about HR policies, send them to that route.
Otherwise, respond politely and let them know you can help with HR-related queries."""

def get_messages(messages):
    return [SystemMessage(content=system_message)] + messages

chain = get_messages | llm.bind_tools([Router])

# Define the Workflow
tools = [provide_hr_info]
tool_node = ToolNode(tools)

def add_name(message, name):
    """Add a name to the AIMessage."""
    if isinstance(message, AIMessage):
        _dict = message.dict()
        _dict["name"] = name
        return AIMessage(**_dict)
    else:
        return message

def _get_last_ai_message(messages):
    """Get the last AIMessage from the message list."""
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None

def _is_tool_call(msg):
    """Check if the message is a tool call."""
    return hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs

def _route(messages):
    """Route the messages to the appropriate node."""
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if not last_message.tool_calls:
            return END
        else:
            if last_message.name == "general":
                if len(last_message.tool_calls) > 1:
                    raise ValueError("Too many tools")
                return last_message.tool_calls[0]["args"]["choice"]
            else:
                return "tools"
    last_m = _get_last_ai_message(messages)
    if last_m is None:
        return "general"
    if last_m.name == "hr_info":
        return "hr_info"
    else:
        return "general"

def _filter_out_routes(messages):
    """Filter out route messages."""
    ms = []
    for m in messages:
        if _is_tool_call(m):
            if m.name == "general":
                continue
        ms.append(m)
    return ms

general_node = _filter_out_routes | chain | partial(add_name, name="general")

memory = MemorySaver()
workflow = MessageGraph()
workflow.add_node("general", general_node)
workflow.add_node("tools", tool_node)
workflow.add_node("hr_info", provide_hr_info)
nodes = {
    "general": "general",
    "hr_info": "hr_info",
    END: END,
    "tools": "tools",
}
workflow.add_conditional_edges("general", _route, nodes)
workflow.add_conditional_edges("tools", _route, nodes)
workflow.add_conditional_edges("hr_info", _route, nodes)
workflow.add_conditional_edges(START, _route, nodes)
graph = workflow.compile()

# Async Interaction Loop
async def main():
    print("Welcome to the HR Bot. Type 'bye' to exit.")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("HR Bot: Goodbye!")
            break
        state = HRState(input=user_input, responses=[])
        history.append(HumanMessage(content=user_input))
        async for output in graph.astream(history):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
        print("\n---\n")

if __name__ == "__main__":
    asyncio.run(main())
