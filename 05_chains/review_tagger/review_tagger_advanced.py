import os
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from config import ModelConfig  # Import model configuration from config.py
from logging_config import setup_logging  # Import logging setup from logging_config.py

# Setup logging
setup_logging()

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI LLM
def initialize_openai_model() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=ModelConfig.TEMPERATURE.value, 
        model_name=ModelConfig.MODEL_NAME.value
    )

llm = initialize_openai_model()

# --------------------------------------------
# --------- Pydantic Model for Tagging --------
# --------------------------------------------

class TaggingOutput(BaseModel):
    """Pydantic model to represent the sentiment, language, tone, and response output."""
    sentiment: str = Field(description="The sentiment of the hotel review")
    language: str = Field(description="The language of the hotel review")
    tone: str = Field(description="The tone of the hotel review")
    response: str = Field(description="The response generated based on the tone and sentiment")

# --------------------------------------------
# --------- Tagging and Response Chain Creation ------------ 
# --------------------------------------------

def get_tagging_chain() -> RunnableParallel:
    # Define the schema to ask for tone, sentiment, language, and generate a response
    schema_template = """Tag the sentiment, language, and tone for a hotel review based on the following:
    - Sentiment: Positive, Negative, Neutral, Abusive, etc.
    - Language: Chinese, Spanish, English, Hindi, Arabic, etc.
    - Tone: Formal, Informal, Sarcastic, Angry, Friendly, etc.
    
    Review: {HotelReview}
    
    Please respond in JSON format:
    {{
      "sentiment": "sentiment_value",
      "language": "language_value",
      "tone": "tone_value",
      "response": "response_value"  # Provide a response appropriate to the detected tone and sentiment
    }}
    """

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(schema_template)

    # Create a runnable chain using RunnableParallel
    chain = (
        RunnableParallel({"HotelReview": RunnablePassthrough()})
        | prompt
        | llm
    )
    
    return chain

# --------------------------------------------
# --------- Question Answering ---------------
# --------------------------------------------

class HotelReview(BaseModel):
    __root__: str

def perform_tagging(tagging_chain: RunnableParallel, review: str) -> TaggingOutput:
    # Run the tagging chain to perform sentiment, language, tone tagging, and response generation
    if not review:
        raise ValueError("Hotel review cannot be empty.")
    
    chain = tagging_chain.with_types(input_type=HotelReview)
    review_input = HotelReview(__root__=review)
    
    # Execute the chain
    response = chain.invoke(review_input.__root__)
    
    # Define the output parser for TaggingOutput model
    parser = PydanticOutputParser(pydantic_object=TaggingOutput)
    
    # Parse the response content into the Pydantic model
    return parser.parse(response.content)

# --------------------------------------------
# --------- Main Execution (No Streamlit) -----
# --------------------------------------------

if __name__ == "__main__":
    # Initialize the tagging chain
    tagging_chain = get_tagging_chain()

    # Example reviews to analyze
    reviews_input = [
        "The hotel was fantastic! The staff were friendly, and the rooms were very clean.",
        "I had a terrible experience at the hotel. The room was dirty, and the service was slow.",
        "It was an okay stay, nothing too special but not bad either.",
        "El hotel estaba limpio y bien ubicado, pero el servicio fue muy lento y el personal no parecía interesado en ayudar. No creo que vuelva a hospedarme aquí."
        "Absolutely loved it! The location was perfect and the amenities were top-notch.",
        "The hotel was noisy, and I couldn't get a good night's sleep."
    ]

    # Analyze each review
    for review in reviews_input:
        try:
            # Perform tagging and analysis
            result = perform_tagging(tagging_chain, review)

            # Print results
            print("-----Quick info of the fields------")
            print("sentiment : The sentiment of the hotel review")
            print("language : The language of the hotel review")
            print("tone : The tone of the hotel review")
            print("response : The response generated based on the tone and sentiment")
            print("------------------------------------------------------")
            print("Analysis Results for Review:")
            pprint({
                "Review": review,
                "Language": result.language,
                "Sentiment": result.sentiment,
                "Tone": result.tone,
                "Response": result.response
            })
            print("\n" + "-"*50 + "\n")  # Separator between reviews
        except Exception as e:
            print(f"An error occurred while processing the review: {review}\nError: {e}")
