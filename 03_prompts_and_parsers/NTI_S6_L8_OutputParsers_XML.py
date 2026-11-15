from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.output_parsers import XMLOutputParser
from langchain_openai import ChatOpenAI


# Set up a parser to handle output formatting
parser = XMLOutputParser()

# Create a prompt template for the system message
system_prompt_template: PromptTemplate = PromptTemplate(
    template="""\n {format_instructions} \n .
    Please enclose the movies in <movie></movie> tags.
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Wrap the system prompt in a SystemMessagePromptTemplate
system_message_template = SystemMessagePromptTemplate(prompt=system_prompt_template)

# Create a prompt template for the human message
user_prompt_template: PromptTemplate = PromptTemplate(
    template="Generate the shortened filmography for {actor}",
    input_variables=["actor"],
)

# Wrap the user prompt in a HumanMessagePromptTemplate
user_message_template = HumanMessagePromptTemplate(prompt=user_prompt_template)

# Combine the system and human message templates into a chat prompt template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_template, user_message_template]
)

# Format the messages by replacing the placeholders with actual values
messages = chat_prompt_template.format_messages(actor="Tom Hanks")

# Create an instance of the ChatOpenAI model
chat_model: ChatOpenAI = ChatOpenAI()

# Generate the response from the chat model using the messages
chat_model_response = chat_model.invoke(messages)

# Extract and format the response from the model's output
formatted_response = parser.invoke(chat_model_response)

# Print the formatted response type
print(f"RAW Answer: {chat_model_response}")
print(f"RAW Answer Type: {type(chat_model_response)}")
print(f"AnswerType: {type(formatted_response)}")
print(f"Answer: {formatted_response}")

for s in formatted_response['movies']:
    print(s['movie'])