# ---------USING OpenAI Python client library: Multiple Prompts----------------

from openai import OpenAI

client = OpenAI()

# Define the prompts (input)
prompts = [
    "Explain how cloud computing works? Assume that I am 12th class student with no prior technical knowledge",
    "How ants follow each other in line? Explain to me in very simple language. Not related to questions asked till date.",
    "Tell me current weather in bangalore, india?"
]

# Function to generate response from OpenAI (model)
def get_completion(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are helpful assistant, skilled in explaining things in simple and easy to understand manner."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion

# Function to extract proper response from the completion
def extract_response(completion):
    return completion.choices[0].message.content

# Generate and extract responses for all prompts (output)
responses = [get_completion(prompt) for prompt in prompts]

# Print the responses
for i, response in enumerate(responses):
    print(f"Question {i+1}: {prompts[i]}")
    print(f"Answer: {extract_response(response)}\n")