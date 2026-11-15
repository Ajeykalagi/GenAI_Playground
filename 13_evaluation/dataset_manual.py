# Install necessary packages
# !pip install -U langsmith openai ollama
# !pip install --upgrade pip

import os
import pandas as pd

# Set environment variables for LangChain
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # Enables tracing
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "langchain-first-project"

# Define QA pairs
inputs = [
    "How many tokens was DBRX pre-trained on?",
    "Is DBRX a MOE model and how many parameters does it have?",
    "How many GPUs was DBRX trained on and what was the connectivity between GPUs?",
]

outputs = [
    "DBRX was pre-trained on 12 trillion tokens of text and code data.",
    "Yes, DBRX is a fine-grained mixture-of-experts (MoE) architecture with 132B total parameters.",
    "DBRX was trained on 3072 NVIDIA H100s connected by 3.2Tbps Infiniband",
]

# Create a list of dictionaries representing the QA pairs
qa_pairs = [{"question": q, "answer": a} for q, a in zip(inputs, outputs)]

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(qa_pairs)

# Write the DataFrame to a CSV file
csv_path = "DBRX_eval.csv"
df.to_csv(csv_path, index=False)

# Print types for verification
print(type(qa_pairs))
print(type(qa_pairs[0]))

from langsmith import Client

# Initialize the LangSmith client
client = Client()

# Define dataset name
dataset_name = "DBRX"

# Create a new dataset
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="QA pairs about DBRX model.",
)

# Add the QA pairs to the dataset as examples
client.create_examples(
    inputs=[{"question": q} for q in inputs],
    outputs=[{"answer": a} for a in outputs],
    dataset_id=dataset.id,
)

# Define new QA pairs to be added
new_questions = [
    "What is the context window of DBRX Instruct?",
]

new_answers = [
    "DBRX Instruct was trained with up to a 32K token context window.",
]

# Add the new QA pairs to the dataset as examples
client.create_examples(
    inputs=[{"question": q} for q in new_questions],
    outputs=[{"answer": a} for a in new_answers],
    dataset_id=dataset.id,
)