from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

# Load the HR policy text
with open("input/hr_policy.txt", "r", encoding="utf-8") as file:
    hr_policy_text = file.read()

class HRState(TypedDict):
    input: str
    responses: Annotated[List[str], operator.add]

graph = StateGraph(HRState)

def handle_greeting(state):
    user_input = state['input']
    if 'hello' in user_input.lower():
        response = "Hello! How can I assist you today?"
    else:
        response = "Hi! How can I help you with your HR-related queries?"
    return {'responses': [response]}

def provide_hr_info(state):
    user_input = state['input']
    if 'policy' in user_input.lower():
        response = "Sure, here is some information about our HR policies:\n" + hr_policy_text[:500] + "\n... [truncated]"
    else:
        response = "I'm sorry, I don't have information on that. Could you please clarify?"
    return {'responses': [response]}

def handle_unknown(state):
    response = "I'm sorry, I didn't understand that. Could you please clarify or ask something specific about HR policies?"
    return {'responses': [response]}

def handle_end(state):
    return {'responses': ["Goodbye!"]}

graph.add_node("greeting", handle_greeting)
graph.add_node("hr_info", provide_hr_info)
graph.add_node("unknown", handle_unknown)
graph.add_node("end", handle_end)

def should_continue(state):
    user_input = state['input']
    if 'bye' in user_input.lower():
        return 'end'
    elif 'policy' in user_input.lower():
        return 'hr_info'
    else:
        return 'unknown'

graph.add_conditional_edges(
    "greeting",
    should_continue,
    {
        "greeting": "greeting",
        "hr_info": "hr_info",
        "unknown": "unknown",
        "end": "end"
    }
)

graph.add_conditional_edges(
    "hr_info",
    should_continue,
    {
        "greeting": "greeting",
        "hr_info": "hr_info",
        "unknown": "unknown",
        "end": "end"
    }
)

graph.add_conditional_edges(
    "unknown",
    should_continue,
    {
        "greeting": "greeting",
        "hr_info": "hr_info",
        "unknown": "unknown",
        "end": "end"
    }
)

graph.set_entry_point("greeting")

app = graph.compile()

def run_hr_bot(user_input):
    state = HRState(input=user_input, responses=[])
    result = app.invoke(state)
    return result['responses']

if __name__ == "__main__":
    print("Welcome to the HR Bot. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("HR Bot: Goodbye!")
            break
        responses = run_hr_bot(user_input)
        for response in responses:
            print(f"HR Bot: {response}")
