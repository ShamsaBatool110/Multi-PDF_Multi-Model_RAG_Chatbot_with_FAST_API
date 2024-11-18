import streamlit as st
import requests

# Define the base URL of your FastAPI server
BASE_API_URL = "http://127.0.0.1:8000"

# Function to upload PDFs
def upload_pdfs(files):
    upload_url = f"{BASE_API_URL}/upload_pdfs/"
    file_data = [("files", (file.name, file, "application/pdf")) for file in files]
    response = requests.post(upload_url, files=file_data)
    return response.json()

# Function to process PDFs
def process_pdfs(selected_pdfs, llm_choice):
    process_url = f"{BASE_API_URL}/process_pdfs/"
    data = {"llm_choice": llm_choice, "pdf_files": selected_pdfs}
    response = requests.post(process_url, data=data)
    return response.json()

# Function to ask a question
def ask_question(question, llm_choice):
    ask_url = f"{BASE_API_URL}/ask_question/"
    json_data = {"question": question, "llm_choice": llm_choice}
    response = requests.post(ask_url, json=json_data)
    return response.json()

# Streamlit UI
st.title("Multi-PDF RAG Chatbot Client")

# PDF Upload section
st.header("Upload PDFs")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)


if uploaded_files:
    if st.button("Upload PDFs"):
        upload_response = upload_pdfs(uploaded_files)
        st.success(f"Uploaded PDFs: {upload_response.get('Uploaded PDFs')}")

# LLM Selection and Process PDFs section
st.header("Process PDFs")
saved_pdfs = st.multiselect("Select previously uploaded PDFs", [file.name for file in uploaded_files])
llm_choice = st.selectbox("Select LLM", ["Mixtral", "Phi", "Llama 3.1"])

if st.button("Process PDFs"):
    if saved_pdfs and llm_choice:
        process_response = process_pdfs(saved_pdfs, llm_choice)
        st.success(f"Processed PDFs with {llm_choice}: {process_response.get('status')}")
    else:
        st.error("Please select PDFs and an LLM to process")

# Ask Question section
st.header("Ask a Question")
question = st.text_input("Type your question here")

if st.button("Ask Question"):
    if question and llm_choice:
        question_response = ask_question(question, llm_choice)
        st.subheader("Answer:")
        st.write(question_response.get("answer"))
        st.subheader("Chat History:")
        for msg in question_response.get("chat_history", []):
            st.write(msg.get("content"))
    else:
        st.error("Please enter a question and select an LLM to ask")

