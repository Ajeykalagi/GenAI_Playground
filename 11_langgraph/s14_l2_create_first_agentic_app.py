import openai
import re

# Initialize the OpenAI client
from openai import OpenAI
client = OpenAI()

# Example: Simple chat completion using GPT-3.5-turbo model
chat_completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello world"}]
)

# Print the response from the model
print(chat_completion.choices[0].message.content)

# Define the Agent class to manage interactions with the model
class Agent:
    def __init__(self, system=""):
        # Initialize the agent with an optional system prompt
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        # Add the user message and generate a response
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        # Generate a response using the OpenAI API
        completion = client.chat.completions.create(
                        model="gpt-4o", 
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content

# Define the system prompt for the agent
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary.

average_dog_weight:
e.g. average_dog_weight: Collie
Returns average weight of a dog when given the breed.

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look up the dog's weight using average_dog_weight.
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weighs 51 lbs.

You then output:

Answer: A Bulldog weighs 51 lbs.
""".strip()

# Define functions for the actions the agent can perform
def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name == "Scottish Terrier": 
        return "Scottish Terriers average 20 lbs"
    elif name == "Border Collie":
        return "A Border Collie's average weight is 37 lbs"
    elif name == "Toy Poodle":
        return "A Toy Poodle's average weight is 7 lbs"
    else:
        return "An average dog weighs 50 lbs"

# Map action names to corresponding functions
known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

# ------------------------------------------------
# ---------------automated agent workflow---------
# ------------------------------------------------


# Regular expression to match action commands in the agent's response
action_re = re.compile(r'^Action: (\w+): (.*)$')

# Function to handle a query with the agent, allowing multiple turns
def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        if actions:
            # If an action is detected, execute it
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            # If no actions are found, the loop ends
            return

# Example usage of the query function
question = """I have 2 dogs, a Border Collie and a Scottish Terrier. \
What is their combined weight?"""
query(question)
