# retrieval_agent.py

import json
import operator
from typing import Annotated, Sequence, TypedDict
import pprint
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import END, StateGraph
from langchain.tools.retriever import create_retriever_tool

# Step 1: Load Documents and Create Retriever
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=50)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

tool = create_retriever_tool(retriever, "retrieve_blog_posts", "Search and return information about Lilian Weng blog posts.")
tools = [tool]
tool_executor = ToolExecutor(tools)

# Step 2: Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Step 3: Decision Functions
def should_retrieve(state):
    print("---DECIDE TO RETRIEVE---")
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" not in last_message.additional_kwargs:
        print("---DECISION: DO NOT RETRIEVE / DONE---")
        return "end"
    else:
        print("---DECISION: RETRIEVE---")
        return "continue"

def check_relevance(state):
    print("---CHECK RELEVANCE---")
    class FunctionOutput(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    parser = PydanticOutputParser(pydantic_object=FunctionOutput)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of retrieved docs to a user question. \n 
        Here are the retrieved docs:
        \n ------- \n
        {context} 
        \n ------- \n
        Here is the user question: {question}
        If the docs contain keyword(s) in the user question, then score them as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question. \n 
        Output format instructions: \n {format_instructions}""",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini-0125-preview")
    chain = prompt | model | parser

    messages = state["messages"]
    last_message = messages[-1]
    score = chain.invoke({"question": messages[0].content, "context": last_message.content})

    if score.binary_score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "yes"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score.binary_score)
        return "no"

# Step 4: Node Functions
def call_model(state):
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini-0125-preview")
    functions = [format_tool_to_openai_function(t) for t in tools]
    model = model.bind_functions(functions)
    response = model.invoke(messages)
    return {"messages": [response]}

def call_tool(state):
    print("---EXECUTE RETRIEVAL---")
    messages = state["messages"]
    last_message = messages[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"])
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}

# Step 5: Define the Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_retrieve,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "action",
    check_relevance,
    {
        "yes": "agent",
        "no": END,
    },
)

app = workflow.compile()

# Step 6: Run the Agent

def run_agent(question):
    inputs = {
        "messages": [
            HumanMessage(
                content=question
            )
        ]
    }

    for output in app.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        run_agent(user_question)
