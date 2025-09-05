#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import requests
import json

# Function to read uploaded file into a DataFrame
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        df = None
    return df

# Function to convert DataFrame to a string for API prompt
def df_to_string(df):
    return df.to_csv(index=False)

# Function to ask question to the API
def ask_question(csv_data, question):
    url = "https://stg1.mmc-bedford-int-non-prod-ingress.mgti.mmc.com/coreapi/openai/v1/deployments/mmc-tech-gpt-35-turbo-16k-0613/chat/completions"
    prompt = (
        f"You are a helpful assistant that answers questions based on the following CSV data:\n"
        f"{csv_data}\n"
        f"Please answer the question:\n"
        f"{question}"
    )
    payload = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on provided CSV data."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1550
    })
    headers = {
        'x-api-key': 'fb2a2161-2e29-4a74-a3a2-3a3e270f7cfe1bbf4038-e134-48d4-80f9-6ba532bccd53',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()
    return response_data['choices'][0]['message']['content']

# Streamlit app
st.set_page_config(
    page_title="CSV/Excel Data Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for aesthetics
st.markdown(
"""
<style>
    body {
        background-color: #f0f2f6;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .header {
        text-align: center;
        padding: 20px;
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True
)

# Header
st.markdown('<div class="header">CSV/Excel Data Chatbot 🤖</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.success("File uploaded successfully!")
        st.subheader("Preview of your data")
        st.dataframe(df.head())

        # Convert DataFrame to string for API
        csv_data = df_to_string(df)

        # Ask questions
        question = st.text_input("Ask a question about your data:")

        if question:
            with st.spinner("Thinking..."):
                answer = ask_question(csv_data, question)
            st.markdown("### Answer")
            st.write(answer)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
else:
    st.info("Please upload a CSV or Excel file to start.")


# In[ ]:




