# Install necessary packages (run separately in your environment)
# !pip install -U langchain_community langchain-openai langchain langgraph bs4

import os
import asyncio
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from typing import List, TypedDict



# Load LCEL documentation
url = "https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# Sort and concatenate the documents
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

# Define the OpenAI prompt and data model
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n
            Here is a full set of LCEL documentation: \n ------- \n {context} \n ------- \n
            Answer the user question based on the above provided documentation. Ensure any code you provide can be executed \n
            with all required imports and variables defined. Structure your answer with a description of the code solution. \n
            Then list the imports. And finally list the functioning code block. Ensure the code is compatible with LangChain version 2. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

class Code(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-0125-preview")
code_gen_chain = code_gen_prompt | llm.with_structured_output(Code)

def parse_output(solution):
    return solution

# Optional: With re-try to correct for failure to invoke tool
code_gen_chain = code_gen_chain | parse_output

class GraphState(TypedDict):
    error: str
    messages: List
    generation: Code
    iterations: int

# Graph nodes
def generate(state: GraphState):
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    if error == "yes":
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]

    code_solution = code_gen_chain.invoke(
        {"context": concatenated_content, "messages": messages}
    )
    messages += [
        (
            "assistant",
            f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}

def code_check(state: GraphState):
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    imports = code_solution.imports
    code = code_solution.code

    try:
        exec(imports)
    except Exception as e:
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    try:
        exec(imports + "\n" + code)
    except Exception as e:
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }

def reflect(state: GraphState):
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    reflections = code_gen_chain.invoke(
        {"context": concatenated_content, "messages": messages}
    )
    messages += [("assistant", f"Here are reflections on the error: {reflections}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations}

def decide_to_finish(state: GraphState):
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        return "end"
    else:
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"

flag = "do not reflect"
max_iterations = 3

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)
workflow.add_node("check_code", code_check)
workflow.add_node("reflect", reflect)

# Build graph
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "reflect": "reflect",
        "generate": "generate",
    },
)
workflow.add_edge("reflect", "generate")
app = workflow.compile()

# Example user query
question = input("Which code do you need? : ")

# Initialize state
state = {
    "error": "no",
    "messages": [("user", question)],
    "generation": Code(prefix="", imports="", code=""),
    "iterations": 0,
}

# Invoke the app with the example query
result = app.invoke(state)

# Write the response code to a text file with better formatting
with open("code/response_code.txt", "w") as file:
    file.write("Prefix:\n")
    file.write(result["generation"].prefix + "\n\n")
    file.write("Imports:\n")
    file.write(result["generation"].imports + "\n\n")
    file.write("Code:\n")
    file.write(result["generation"].code + "\n")

print("The response code has been written to response_code.txt.")
