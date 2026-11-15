# GenAI Playground

A hands-on repository for learning Generative AI, Large Language Models, LangChain, RAG, and AI application development.

This project is designed for anyone who wants to learn by building - whether you are a beginner exploring AI, a developer adding GenAI skills, or a builder prototyping real-world AI experiences.

## Why this repository?

The best way to learn GenAI is to experiment with real tools and real workflows.

This repo brings together practical examples for:

- prompt engineering and structured outputs
- model usage and API integration
- LangChain fundamentals
- chains, memory, and orchestration
- retrieval-augmented generation (RAG)
- advanced query strategies and agent workflows
- tracing, debugging, and evaluation
- Streamlit-based AI apps

## What you will learn

- how LLMs respond to prompts and context
- how to structure GenAI workflows
- how to build with LangChain
- how to use embeddings and vector search
- how retrieval works in real applications
- how to create multi-step AI agents and workflows
- how to debug and inspect LLM behavior
- how to turn ideas into AI-powered prototypes

## Repository structure

- `01_foundations/` - introduction to LangChain and GenAI basics
- `02_models/` - working with models and provider setups
- `03_prompts_and_parsers/` - prompts, templates, and structured output handling
- `04_streamlit_apps/` - practical UI-based AI examples
- `05_chains/` - chaining prompts and model logic together
- `06_memory/` - memory-enabled conversational patterns
- `07_prompt_engineering/` - experimentation with prompt design
- `08_rag/` - retrieval and document Q&A patterns
- `09_langsmith/` - observability, tracing, and debugging
- `10_advanced_rag/` - advanced retrieval and multi-step extraction strategies
- `11_langgraph/` - agentic flows and graph-based orchestration
- `12_callbacks/` - callbacks and execution control
- `13_evaluation/` - testing, evaluation, and iteration
- `14_project_examples/` - sample GenAI project ideas and prototypes
- `streamlit_cloud/` - deployment and hosting examples

## Recommended learning path

If you are starting from scratch, this is a practical order:

1. Start with `01_foundations/`
2. Explore `02_models/` and `03_prompts_and_parsers/`
3. Build small apps in `04_streamlit_apps/`
4. Learn chaining and flow design in `05_chains/`
5. Add interaction memory in `06_memory/`
6. Deepen your prompting skills in `07_prompt_engineering/`
7. Learn retrieval and search in `08_rag/` and `10_advanced_rag/`
8. Explore agents and orchestration in `11_langgraph/`
9. Use `09_langsmith/` and `13_evaluation/` for debugging and validation

See `docs/Learning_Path.md` for a more detailed roadmap.

## Quick start

### 1. Create a virtual environment

```bash
python -m venv v_env
```

On Windows:

```bash
v_env\Scripts\activate
```

On macOS/Linux:

```bash
source v_env/bin/activate
```

### 2. Install dependencies

Install the requirements for the module you want to run:

```bash
pip install -r <module-folder>/requirements.txt
```

### 3. Run an example

Open a Python file from any section and run it with your preferred editor or terminal.

Some examples may require:

- OpenAI API keys
- LangChain packages
- vector database support
- Streamlit
- environment variables for model access

## Practical goals of this repo

This repository is meant to help you:

- understand how LLM apps are structured
- build useful AI workflows from real examples
- test and refine prompts and retrieval behavior
- create prototypes for your own use cases
- move from tutorials to hands-on building

## Suggested project ideas

Use this repo as a foundation for building:

- private document Q&A assistants
- customer support copilots
- research and summarization assistants
- internal knowledge bots
- AI-powered web apps and dashboards
- workflow automation with LLMs

## How to use this project

Treat each folder as a building block. Copy, adapt, and combine ideas from multiple examples to create your own GenAI projects.

The best approach is:

- read the code
- modify the logic
- run experiments
- observe outputs
- keep iterating

## Learning mindset

This project is intentionally practical and iterative.

It is not about memorizing a fixed curriculum — it is about understanding patterns, testing assumptions, and building the confidence to ship AI experiences.

## Git workflow

To initialize this repository and push it to your own GitHub account:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Notes

- Some modules may require external APIs or paid model access.
- This repository is intended for learning, experimentation, and prototyping.
- You are encouraged to adapt the code for your own projects and ideas.

## Final note

The goal is simple: help you go from curiosity to building with GenAI.

Start small, stay curious, and keep experimenting.

## More resources

- [Learning path](docs/Learning_Path.md) — recommended progression for studying GenAI
- [Setup guide](docs/Setup.md) — environment setup and dependency installation
- [Project ideas](docs/ProjectIdeas.md) — starter project ideas and use cases
- [FAQ](docs/FAQ.md) — common questions and troubleshooting
