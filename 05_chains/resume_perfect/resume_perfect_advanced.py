import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from config import ModelConfig, FilePaths
from logging_config import setup_logging

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Set up logging configuration
setup_logging()
logger = logging.getLogger(__name__)

# Initialize OpenAI model using settings from config.py
llm = ChatOpenAI(temperature=ModelConfig.TEMPERATURE.value, model_name=ModelConfig.MODEL_NAME.value)

# --------------------------------------------
# --------- File Loading Functions ------------
# --------------------------------------------

def load_resume_from_pdf(file_path: str) -> str:
    """
    Load and extract text from a PDF resume file using LangChain's PyPDFLoader.
    """
    logger.info(f"Loading resume from PDF file: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Combine all text from the pages
    resume_text = " ".join([doc.page_content for doc in documents])

    logger.info("Resume text successfully extracted from PDF.")
    return resume_text

# --------------------------------------------
# --------- Job Description from Input --------
# --------------------------------------------

def get_job_description_from_input() -> str:
    """
    Take job description input from the user.
    """
    logger.info("Prompting user for job description.")
    print("\nPlease enter the job description (press Enter when done):")
    job_description = input()
    return job_description

# --------------------------------------------
# --------- Job Description Parsing ------------
# --------------------------------------------

def extract_job_keywords(job_description: str) -> str:
    """
    Extract key skills, responsibilities, and qualifications from the job description.
    """
    logger.info("Extracting key skills and qualifications from the job description.")
    prompt_template = ChatPromptTemplate.from_template(
        "Extract key skills, responsibilities, and qualifications from this job description: {job_description}"
    )

    chain = (
        RunnableParallel({"job_description": RunnablePassthrough()})
        | prompt_template
        | llm
    )

    response = chain.invoke({"job_description": job_description})
    logger.info("Key skills and qualifications extracted successfully.")
    return response.content  # Use the .content attribute to access the message content

# --------------------------------------------
# --------- Resume Gap Identification ---------
# --------------------------------------------

def identify_resume_gaps(resume: str, job_keywords: str) -> str:
    """
    Compare the resume with job requirements and list the gaps for improvement.
    """
    logger.info("Identifying gaps between the resume and the job description.")
    prompt_template = ChatPromptTemplate.from_template(
        "Compare this resume: {resume} with these job requirements: {job_keywords}."
        "List the gaps and areas of improvement to match the job better."
    )

    chain = (
        RunnableParallel({"resume": RunnablePassthrough(), "job_keywords": RunnablePassthrough()})
        | prompt_template
        | llm
    )

    response = chain.invoke({"resume": resume, "job_keywords": job_keywords})
    logger.info("Resume gaps identified successfully.")
    return response.content  # Use the .content attribute to access the message content

# --------------------------------------------
# --------- Reformat Responsibilities ---------
# --------------------------------------------

def reformat_responsibilities(resume: str) -> str:
    """
    Reformat the job responsibilities in the resume to better match the job description.
    """
    logger.info("Reformatting job responsibilities in the resume.")
    prompt_template = ChatPromptTemplate.from_template(
        "Reformat the job responsibilities in this resume to better match the job description."
        "Make them concise, clear, and action-oriented: {resume}"
    )

    chain = (
        RunnableParallel({"resume": RunnablePassthrough()})
        | prompt_template
        | llm
    )

    response = chain.invoke({"resume": resume})
    logger.info("Resume responsibilities reformatted successfully.")
    return response.content  # Use the .content attribute to access the message content

# --------------------------------------------
# --------- Main Function to Process Resume ----
# --------------------------------------------

def process_resume(file_path_resume: str) -> tuple:
    """
    Main function to load the resume, get job description from input, identify gaps, and reformat responsibilities.
    """
    logger.info("Starting the resume processing workflow.")

    # Load resume from PDF
    resume = load_resume_from_pdf(file_path_resume)
    
    # Get job description from user input
    job_description = get_job_description_from_input()

    # Extract key points from job description
    job_keywords = extract_job_keywords(job_description)
    print("\n----- Job Keywords -----")
    print(f"{job_keywords}")
    logger.info(f"Extracted job keywords: {job_keywords}")

    # Identify gaps in the resume
    gaps = identify_resume_gaps(resume, job_keywords)
    print("\n----- Areas of Improvement -----")
    print(f"{gaps}")
    logger.info(f"Identified gaps: {gaps}")

    # Reformat responsibilities in the resume
    updated_resume = reformat_responsibilities(resume)
    print("\n----- Updated Resume Responsibilities -----")
    print(f"{updated_resume}")
    logger.info("Resume responsibilities reformatted.")

    return updated_resume, gaps

# Sample usage
if __name__ == "__main__":
    # Path to the resume PDF file (can be taken from config)
    resume_path = FilePaths.RESUME_PATH.value

    logger.info("Running the resume analysis program.")
    updated_resume, suggestions = process_resume(resume_path)
    logger.info("Program finished successfully.")