# Setup Guide

This guide helps you get the repository running locally.

## 1. Prerequisites

You will typically need:

- Python 3.10 or newer
- pip
- a virtual environment tool
- optional API keys for OpenAI or other model providers

## 2. Create a virtual environment

```bash
python -m venv v_env
```

Windows:

```bash
v_env\Scripts\activate
```

macOS/Linux:

```bash
source v_env/bin/activate
```

## 3. Install dependencies

Each module may include its own `requirements.txt` file.

Example:

```bash
pip install -r 08_rag/requirements.txt
```

If a folder does not include a requirements file, install the packages required by that example directly.

## 4. Configure API keys

Some examples use model providers such as OpenAI.

Set your environment variables before running scripts:

```bash
setx OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
```

Or in a shell session:

```bash
export OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")
```

## 5. Run examples

Open a Python script in VS Code or another editor and run it with Python.

For Streamlit apps:

```bash
streamlit run app.py
```

## 6. Troubleshooting

### Import errors

Try reinstalling dependencies:

```bash
pip install -r requirements.txt
```

### Missing environment variables

Check that your API credentials are correctly set and readable in your shell.

### Package conflicts

Create a new environment and reinstall only the dependencies you need.

## 7. Recommended workflow

Use one environment for learning, then create a separate environment for larger personal projects if needed.
