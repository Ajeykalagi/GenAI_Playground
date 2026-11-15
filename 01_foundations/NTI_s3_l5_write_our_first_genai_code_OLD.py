import os

# # --------------------------------------------
# # --------------------------------------------
# # ---------RAW API--------------------------
# # --------------------------------------------
# # --------------------------------------------

import requests

# Define the API endpoint and API key
url = "https://api.openai.com/v1/chat/completions"

api_key = os.environ["OPENAI_API_KEY"]

# Define the request headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Define the request body
data = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": "Explain how cloud computing works? Assume that I am a 12th class student with no prior technical knowledge"
        }
    ],
    "temperature": 1
}

# Send the request to the OpenAI API
response = requests.post(url, headers=headers, json=data)

# Check if the request was successful and extract the response
if response.ok:
    answer = response.json()['choices'][0]['message']['content']
    print(f"Question: {data['messages'][0]['content']}")
    print(f"Answer: {answer}")
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)


# -----------------------------------------------------------
# -----------------------------------------------------------
# ---------USING OpenAI Python client library----------------
# -----------------------------------------------------------
# -----------------------------------------------------------


# Install the necessary packages using the following commands:
# pip install openai
# pip install --upgrade pip

from openai import OpenAI

# Replace with your OpenAI API key
client = OpenAI()


# -----------------------
# -----Single prompt-----
#-------------------------

# Define the prompt (input)
prompt = "Explain how cloud computing works? Assume that I am 12th class student with no prior technical knowledge"


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
response = get_completion(prompt)

# Print the responses
print(f"Question: {prompt}")
print(f"Answer: {extract_response(response)}")

# -----------------------
# -----Multiple prompts--
#-------------------------

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

# -----------------------
# -----Prompt through user input-
#-------------------------



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