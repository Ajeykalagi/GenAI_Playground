from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.structured_output import create_openai_fn_runnable

# Define specific functions
def get_item_info_2(item_name):
    item = fake_db["items"].get(item_name)
    if not item:
        return {"message": f"No information available for item: {item_name}"}
    return {"name": item_name, "price": item["price"], "ingredients": item["ingredients"]}

def place_order_2(item_name, quantity, address):
    if item_name not in fake_db["items"]:
        return {"message": f"We don't have {item_name}!"}
    if quantity < 1:
        return {"message": "You must order at least one item."}
    order_id = len(fake_db["orders"]) + 1
    total_price = fake_db["items"][item_name]["price"] * quantity
    order = {
        "order_id": order_id,
        "item_name": item_name,
        "quantity": quantity,
        "address": address,
        "total_price": total_price
    }
    fake_db["orders"].append(order)
    return {"message": f"Order placed successfully! Your order ID is {order_id}. Total price is ${total_price}."}

# Define the general chat function
def general_chat(query):
    return {"message": f"I'm not sure how to handle the query: '{query}'. Please ask something specific about our menu or place an order."}

# Define function metadata
functions = [
    {
        "name": "get_item_info_2",
        "description": "Get name and price of a menu item from the restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the menu item, e.g. Chop-Suey",
                },
            },
            "required": ["item_name"],
        },
    },
    {
        "name": "place_order_2",
        "description": "Place an order for a menu item from the restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the item you want to order, e.g. Chop-Suey",
                },
                "quantity": {
                    "type": "integer",
                    "description": "The number of items you want to order",
                    "minimum": 1
                },
                "address": {
                    "type": "string",
                    "description": "The address where the food should be delivered",
                },
            },
            "required": ["item_name", "quantity", "address"],
        },
    },
    {
        "name": "general_chat",
        "description": "Handle general queries not specific to the restaurant or menu",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A general query that doesn't match specific functions",
                },
            },
            "required": ["query"],
        },
    }
]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create prompt template
template = """You are an AI chatbot having a conversation 
with a human. You help in customer ordering food from a Chinese restaurant.

Human: {human_input}
AI: """

prompt = PromptTemplate(
    input_variables=["human_input"], 
    template=template
)

# Create the chain with the defined functions
chain = create_openai_fn_runnable(
    functions=functions, 
    llm=llm,
    prompt=prompt
)


def chat(query):
    # Invoke the chain with the user query
    response = chain.invoke({"human_input": query})
    print("Raw Response:", response)  # Debugging line
    
    # Check if the response contains function call metadata
    if isinstance(response, dict) and "name" in response:
        function_name = response.get("name")
        arguments = response.get("arguments", {})
        
        # Call the appropriate function based on the function name
        if function_name in ["get_item_info_2", "place_order_2", "general_chat"]:
            function_response = execute_function(function_name, **arguments)
            return function_response
        else:
            return {"message": "Function not recognized."}
    else:
        return {"message": "No function call detected or no content returned."}



def execute_function(function_name, **kwargs):
    if function_name == "get_item_info_2":
        return get_item_info_2(**kwargs)
    elif function_name == "place_order_2":
        return place_order_2(**kwargs)
    elif function_name == "general_chat":
        return general_chat(**kwargs)
    else:
        return {"message": "Function not recognized."}

# Example database
fake_db = {
    "items": {
        "Chop-Suey": {
            "name": "Chop-Suey",
            "price": 15.0,
            "ingredients": ["noodles", "vegetables", "sauce"]
        },
        "Lo-Mein": {
            "name": "Lo-Mein",
            "price": 12.0,
            "ingredients": ["noodles", "vegetables", "sauce"]
        }
    },
    "orders": []
}

# Invoke the chat function with example inputs
response1 = chat("How much does Chop-Suey cost?")
response2 = chat("I want to order two Chop-Suey to 321 Street")
response3 = chat("How old did my grandfather get?")

print(response1)
print(response2)
print(response3)
