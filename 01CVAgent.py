from dotenv import load_dotenv
load_dotenv()
import asyncio
from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import Gemini
from knowledgebase import pdf_knowledge_base

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.storage.postgres import PostgresStorage
import tempfile
import os
import PyPDF2

cvReaderAgent = Agent(
    name="CVReaderAgent",
    description="You are seasoned technology interview panel  member for hiring fresh college students!",
    model = Gemini(
        id="gemini-2.0-flash-exp",
    ),
    markdown=True,
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
    instructions=[
        "Always only search your knowledge base for resume and job description. Do not provide generic answers",
        "Job description is in jd1.pfd and resume is in Resume.pdf",
        "Highlight the name of the candidate and skills and experience in the response.",
        "For the questions to be asked in interview, you can search web to get questions and answers"
    ],
    tools=[DuckDuckGoTools()],
    read_chat_history=True,
    add_history_to_messages=True
)


print("Loading Resume knowledge base...")
cvReaderAgent.knowledge.load(recreate=False)
print("Resume Knowledge base loaded.")

# UI section
import streamlit as st

# UI Setup
st.title("💬 AI Interview Assistant")
st.caption("Powered by Gemini")

# File upload section
st.subheader("Upload Documents")
col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume (PDF)", type=['pdf'])
    if resume_file:
        # Save uploaded resume to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(resume_file.getvalue())
            resume_path = tmp_file.name

with col2:
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=['pdf'])
    if jd_file:
        # Save uploaded JD to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(jd_file.getvalue())
            jd_path = tmp_file.name

# Analysis button
if resume_file and jd_file:
    if st.button("Analyze Resume"):
        try:
            # Update knowledge base with resume
            pdf_knowledge_base.path = resume_path
            pdf_knowledge_base.load(recreate=True)
            
            # Read job description content
            with open(jd_path, 'rb') as jd:
                job_desc_text = PyPDF2.PdfReader(jd).pages[0].extract_text()
            
            # Create analysis prompt with actual job description
            analysis_prompt = f"""
            Analyze the following resume against these job description requirements:

            Job Description:
            {job_desc_text}

            Please provide:
            1. List candidate's key skills and experiences
            2. Compare with job requirements
            3. Calculate match percentage
            4. Highlight matching skills
            5. Identify gaps
            6. Provide overall recommendation
            """
            
            # Generate and display analysis
            with st.spinner("Analyzing resume..."):
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    response = cvReaderAgent.run(analysis_prompt, stream=True)
                    full_response = ""
                    
                    for _resp_chunk in response:
                        if _resp_chunk.content is not None:
                            full_response += _resp_chunk.content
                            response_placeholder.markdown(
                                full_response.replace("%", "\\%") + "▌",
                                unsafe_allow_html=True
                            )
                    
                    response_placeholder.markdown(
                        full_response.replace("%", "\\%"),
                        unsafe_allow_html=True
                    )
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                    })
                    
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
        finally:
            # Cleanup temporary files
            if 'resume_path' in locals():
                os.unlink(resume_path)
            if 'jd_path' in locals():
                os.unlink(jd_path)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input handling for follow-up questions
if prompt := st.chat_input("Ask follow-up questions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        full_response = ""
        with st.spinner("Analyzing your question..."):
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response = cvReaderAgent.run(prompt, stream=True)
                for _resp_chunk in response:
                    if _resp_chunk.content is not None:
                        full_response += _resp_chunk.content
                        response_placeholder.markdown(
                            full_response.replace("%", "\\%") + "▌",
                            unsafe_allow_html=True
                        )
                response_placeholder.markdown(
                    full_response.replace("%", "\\%"),
                    unsafe_allow_html=True
                )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
        })
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")


