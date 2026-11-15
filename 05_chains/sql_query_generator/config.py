from enum import Enum

class ModelConfig(Enum):
    MODEL_NAME = "gpt-4o-mini"  # Specify the model you want to use
    TEMPERATURE = 0.2      # Lower temperature for more deterministic outputs

class DefaultConfig(Enum):
    DEFAULT_SCHEMA = "public"   # Default schema name if not provided
    DEFAULT_TABLE = "employees"  # Default table name if not provided
