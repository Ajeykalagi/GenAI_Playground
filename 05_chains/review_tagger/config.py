# config.py

from enum import Enum

class ModelConfig(Enum):
    MODEL_NAME = "gpt-4o-mini"  # Specify the model you want to use
    TEMPERATURE = 0.7      # Temperature for response variability
    CHUNK_SIZE = 1000      # Number of characters in each text chunk
    CHUNK_OVERLAP = 200    # Overlap between consecutive chunks
    EMBEDDING_MODEL = "text-embedding-3-small"  # Embedding model for vector store
