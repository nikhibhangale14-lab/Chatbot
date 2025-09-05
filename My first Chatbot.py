#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import SummarizationChain

# Read LenAI credentials from Streamlit secrets
LENAI_API_KEY = st.secrets.get("LENAI_API_KEY")
LENAI_API_URL = st.secrets.get("LENAI_API_URL")

if not LENAI_API_KEY or not LENAI_API_URL:
    st.error("LenAI credentials missing. Set LENAI_API_KEY and LENAI_API_URL in Streamlit secrets.")
    st.stop()

# Set environment variables for OpenAI-compatible clients
os.environ["OPENAI_API_KEY"] = LENAI_API_KEY
os.environ["OPENAI_API_BASE"] = LENAI_API_URL

# Upload PDF files
st.header("My first Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text_page = page.extract_text()
        if text_page:
            text += text_page

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generating embedding
    embeddings = OpenAIEmbeddings()  # Will read OPENAI_API_KEY and OPENAI_API_BASE from env

    # Creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Summarization
    llm = ChatOpenAI(
        temperature=0,
        max_tokens=1000,
        model_name="GPT 4.1 Nano"  # Replace if LenAI model name differs
    )
    
    summarization_chain = SummarizationChain(llm=llm)
    summary = summarization_chain.run(input_documents=chunks)
    
    st.subheader("Document Summary")
    st.write(summary)

    # Sample questions
    st.subheader("Sample Questions You Can Ask:")
    sample_questions = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the conclusions drawn in the document?",
        "Are there any important dates mentioned?",
        "What are the main arguments presented?",
        "Who are the key figures discussed in the document?"
    ]
    for question in sample_questions:
        st.write(f"- {question}")

    # Get user question
    user_question = st.text_input("Type your question here")

    # Do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)

        # Output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)


# In[ ]:




