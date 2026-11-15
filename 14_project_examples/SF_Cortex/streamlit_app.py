# Install the necessary packages using the following commands:
# pip install streamlit pandas snowflake-snowpark-python

import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Get the current session credentials
session = get_active_session()

# Set pandas option for better display
pd.set_option("max_colwidth", None)

# Define the number of chunks to provide as context
num_chunks = 3

# --------------------------------------------
# --------- Functions ------------------------
# --------------------------------------------

def create_prompt(myquestion, rag):
    """Create the prompt for the model based on the question and context."""
    if rag == 1:
        cmd = """
         with results as
         (SELECT RELATIVE_PATH,
           VECTOR_COSINE_SIMILARITY(docs_chunks_table.chunk_vec,
                    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', ?)) as similarity,
           chunk
         from docs_chunks_table
         order by similarity desc
         limit ?)
         select chunk, relative_path from results 
         """
    
        df_context = session.sql(cmd, params=[myquestion, num_chunks]).to_pandas()
        context_length = len(df_context) - 1

        prompt_context = ""
        for i in range(context_length):
            prompt_context += df_context._get_value(i, 'CHUNK')

        prompt_context = prompt_context.replace("'", "")
        relative_path = df_context._get_value(0, 'RELATIVE_PATH')
    
        prompt = f"""
          'You are an expert assistant extracting information from context provided. 
           Answer the question based on the context. Be concise and do not hallucinate. 
           If you don’t have the information just say so.
          Context: {prompt_context}
          Question:  
           {myquestion} 
           Answer: '
           """
        cmd2 = f"select GET_PRESIGNED_URL(@docs, '{relative_path}', 360) as URL_LINK from directory(@docs)"
        df_url_link = session.sql(cmd2).to_pandas()
        url_link = df_url_link._get_value(0, 'URL_LINK')

    else:
        prompt = f"""
         'Question:  
           {myquestion} 
           Answer: '
           """
        url_link = "None"
        relative_path = "None"
        
    return prompt, url_link, relative_path

def complete(myquestion, model_name, rag=1):
    """Complete the question using the selected model and context."""
    prompt, url_link, relative_path = create_prompt(myquestion, rag)
    cmd = f"""
             select SNOWFLAKE.CORTEX.COMPLETE(?,?) as response
           """
    
    df_response = session.sql(cmd, params=[model_name, prompt]).collect()
    return df_response, url_link, relative_path

def display_response(question, model, rag=0):
    """Display the response to the question."""
    response, url_link, relative_path = complete(question, model, rag)
    res_text = response[0].RESPONSE
    st.markdown(res_text)
    if rag == 1:
        display_url = f"Link to [{relative_path}]({url_link}) that may be useful"
        st.markdown(display_url)

# --------------------------------------------
# --------- Streamlit App Setup --------------
# --------------------------------------------

st.title("Asking Questions to Your Own Documents with Snowflake Cortex:")
st.write("""You can ask questions and decide if you want to use your documents for context or allow the model to create its own response.""")
st.write("This is the list of documents you already have:")
docs_available = session.sql("ls @docs").collect()
list_docs = [doc["name"] for doc in docs_available]
st.dataframe(list_docs)

# Sidebar for model selection and context usage
model = st.sidebar.selectbox('Select your model:', (
    'mixtral-8x7b',
    'snowflake-arctic',
    'mistral-large',
    'llama3-8b',
    'llama3-70b',
    'reka-flash',
    'mistral-7b',
    'llama2-70b-chat',
    'gemma-7b'
))

question = st.text_input("Enter question", placeholder="Is there any special lubricant to be used with the premium bike?", label_visibility="collapsed")
rag = st.sidebar.checkbox('Use your own documents as context?')

use_rag = 1 if rag else 0

if question:
    display_response(question, model, use_rag)

# Run the Streamlit app
if __name__ == "__main__":
    st.run()
