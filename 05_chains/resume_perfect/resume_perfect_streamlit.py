import streamlit as st
import os
import tempfile
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from config import ModelConfig
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

def load_resume_from_pdf(uploaded_file) -> str:
    """
    Load and extract text from a PDF resume file uploaded via Streamlit.
    """
    logger.info("Saving and loading uploaded PDF file.")

    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load the saved file using PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # Combine all text from the pages
    resume_text = " ".join([doc.page_content for doc in documents])

    # Delete the temporary file after extraction
    os.remove(tmp_file_path)

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
    return st.text_area("Enter the job description")

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

    return chain.invoke({"job_description": job_description})

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

    return chain.invoke({"resume": resume, "job_keywords": job_keywords})

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

    return chain.invoke({"resume": resume})

# --------------------------------------------
# --------- Helper Function to Extract Content --------
# --------------------------------------------

def extract_content(response) -> str:
    """
    Extract and return the 'content' field from the AIMessage response.
    """
    return response.content

# --------------------------------------------
# --------- Main Function to Process Resume ----
# --------------------------------------------

def main():
    """
    Main function to process the resume and job description, identifying gaps and reformatting responsibilities.
    """
    st.title("Resume Perfect - Streamlit App")

    # Upload the resume
    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")

    if uploaded_file is not None:
        # Load resume from the uploaded PDF
        resume = load_resume_from_pdf(uploaded_file)

        # Get job description from user input
        job_description = get_job_description_from_input()

        # Process if both resume and job description are provided
        if st.button("Process Resume"):
            # Extract key points from job description
            job_keywords_response = extract_job_keywords(job_description)
            job_keywords = extract_content(job_keywords_response)
            st.markdown(f"### Job Keywords:\n\n{job_keywords}")

            # Identify gaps in the resume
            gaps_response = identify_resume_gaps(resume, job_keywords)
            gaps = extract_content(gaps_response)
            st.markdown(f"### Areas of Improvement:\n\n{gaps}")

            # Reformat responsibilities in the resume
            updated_resume_response = reformat_responsibilities(resume)
            updated_resume = extract_content(updated_resume_response)
            st.markdown(f"### Updated Resume:\n\n{updated_resume}")

# --------------------------------------------
# --------- Streamlit Entry Point -------------
# --------------------------------------------

if __name__ == "__main__":
    main()
