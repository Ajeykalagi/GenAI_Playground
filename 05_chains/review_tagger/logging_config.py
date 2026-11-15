# logging_config.py

import logging
import os

def setup_logging():
    # Create a logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logging.basicConfig(
        level=logging.DEBUG,  # Set logging level (DEBUG, INFO, WARNING, ERROR)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/app.log"),  # Write logs to a file
            logging.StreamHandler()  # Also print logs to the console
        ]
    )

    # Reduce verbosity for external libraries
    logging.getLogger('langchain').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
