from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from typing import List

# Initialize OpenAI model using LangChain v2's ChatOpenAI
llm = ChatOpenAI(temperature=0.7)

# --------------------------------------------
# --------- File Loading Functions ------------ 
# --------------------------------------------

def load_resume_from_pdf(file_path: str) -> str:
    """
    Load and extract text from a PDF resume file using LangChain's PyPDFLoader.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Combine all text from the pages
    resume_text = " ".join([doc.page_content for doc in documents])

    return resume_text

# --------------------------------------------
# --------- Pydantic Model for Job Keywords ----
# --------------------------------------------

class JobKeywords(BaseModel):
    """Pydantic model to hold extracted job keywords."""
    skills: List[str] = Field(description="Extracted skills from job description")
    responsibilities: List[str] = Field(description="Extracted responsibilities from job description")
    qualifications: List[str] = Field(description="Extracted qualifications from job description")

# --------------------------------------------
# --------- Pydantic Model for Resume Gaps ----
# --------------------------------------------

class ResumeGapAnalysis(BaseModel):
    """Pydantic model to hold resume gap analysis."""
    gaps: List[str] = Field(description="Identified gaps in the resume for improvement")

# --------------------------------------------
# --------- Job Description from Input --------
# --------------------------------------------

def get_job_description_from_input() -> str:
    """
    Take job description input from the user.
    """
    print("Enter the job description (press Enter when done):")
    job_description = input()
    return job_description

# --------------------------------------------
# --------- Job Description Parsing ------------

def extract_job_keywords(job_description: str) -> JobKeywords:
    """
    Extract key skills, responsibilities, and qualifications from the job description.
    """
    prompt_template = ChatPromptTemplate.from_template(
        """Extract key skills, responsibilities, and qualifications from this job description: {job_description}.
        Please respond in the following strict JSON format:
        {{
            "skills": ["skill1", "skill2", "..."],
            "responsibilities": ["responsibility1", "responsibility2", "..."],
            "qualifications": ["qualification1", "qualification2", "..."]
        }}"""
    )

    chain = (
        RunnableParallel({"job_description": RunnablePassthrough()})
        | prompt_template
        | llm
    )

    response = chain.invoke({"job_description": job_description})
    
    # Define the output parser for JobKeywords model
    parser = PydanticOutputParser(pydantic_object=JobKeywords)
    
    # Parse the response content to the defined Pydantic model
    return parser.parse(response.content)

# --------------------------------------------
# --------- Resume Gap Identification ---------
# --------------------------------------------

def identify_resume_gaps(resume: str, job_keywords: JobKeywords) -> ResumeGapAnalysis:
    """
    Compare the resume with job requirements and list the gaps for improvement.
    """
    prompt_template = ChatPromptTemplate.from_template(
        """Compare this resume: {resume} with these job requirements: {job_keywords}. 
        List the gaps and areas of improvement to match the job better in strict JSON format:
        {{
            "gaps": ["gap1", "gap2", "..."]
        }}"""
    )

    chain = (
        RunnableParallel({"resume": RunnablePassthrough(), "job_keywords": RunnablePassthrough()})
        | prompt_template
        | llm
    )

    response = chain.invoke({"resume": resume, "job_keywords": job_keywords.dict()})

    # Define the output parser for ResumeGapAnalysis model
    parser = PydanticOutputParser(pydantic_object=ResumeGapAnalysis)
    
    # Parse the response content to the defined Pydantic model
    return parser.parse(response.content)

# --------------------------------------------
# --------- Reformat Responsibilities ---------
# --------------------------------------------

class ReformattedResume(BaseModel):
    """Pydantic model to hold reformatted resume responsibilities."""
    updated_responsibilities: List[str] = Field(description="Reformatted resume responsibilities")

def reformat_responsibilities(resume: str) -> ReformattedResume:
    """
    Reformat the job responsibilities in the resume to better match the job description.
    """
    prompt_template = ChatPromptTemplate.from_template(
        """Reformat the job responsibilities in this resume to better match the job description.
        Make them concise, clear, and action-oriented.
        Respond in strict JSON format:
        {{
            "updated_responsibilities": ["responsibility1", "responsibility2", "..."]
        }}"""
    )

    chain = (
        RunnableParallel({"resume": RunnablePassthrough()})
        | prompt_template
        | llm
    )

    response = chain.invoke({"resume": resume})
    
    # Define the output parser for ReformattedResume model
    parser = PydanticOutputParser(pydantic_object=ReformattedResume)
    
    # Parse the response content to the defined Pydantic model
    return parser.parse(response.content)

# --------------------------------------------
# --------- Main Function to Process Resume ----
# --------------------------------------------

def process_resume(file_path_resume: str) -> tuple:
    """
    Main function to load the resume, get job description from input, identify gaps, and reformat responsibilities.
    """
    # Load resume from PDF using PyPDFLoader
    resume = load_resume_from_pdf(file_path_resume)
    
    # Get job description from user input
    job_description = get_job_description_from_input()

    # Extract key points from job description
    job_keywords = extract_job_keywords(job_description)
    print("\n----- Job Keywords -----")
    print(job_keywords)

    # Identify gaps in the resume
    gaps = identify_resume_gaps(resume, job_keywords)
    print("\n----- Areas of Improvement -----")
    print(gaps)

    # Reformat responsibilities in the resume
    updated_resume = reformat_responsibilities(resume)
    print("\n----- Updated Resume Responsibilities -----")
    print(updated_resume)

    return updated_resume, gaps

# Sample usage
if __name__ == "__main__":
    # Path to the resume PDF file
    resume_path = "input/Aditya Resume India V1.pdf"

    updated_resume, suggestions = process_resume(resume_path)
