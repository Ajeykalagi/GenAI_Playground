from enum import Enum

class ModelConfig(Enum):
    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.5

class FileConfig(Enum):
    FILE_NAME = "Passenger_Data_Analysis.csv"
    INPUT_DATA_FOLDER_NAME = "input"
