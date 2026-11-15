from enum import Enum

class ModelConfig(Enum):
    """Enum to store configuration for the model."""
    MODEL_NAME = "gpt-4o-mini"   # Specify the model you want to use
    TEMPERATURE = 0.7      # Set the temperature for the LLM

class FilePaths(Enum):
    """Enum to store file paths."""
    RESUME_PATH = "input/Aditya Resume India V1.pdf"  # Default resume path
