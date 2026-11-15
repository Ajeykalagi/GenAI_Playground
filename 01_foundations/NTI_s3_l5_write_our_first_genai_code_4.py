# ---------Prompt through user input-----------------

from openai import OpenAI

client = OpenAI()

# Take input from the user for the prompt
prompt = input("Please enter your question: ")

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

# Generate and extract the response for the user prompt (output)
response = get_completion(prompt)

# Print the response
print(f"Question: {prompt}")
print(f"Answer: {extract_response(response)}")