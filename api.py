import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from Llama3 import llama3
from Mixtral import mixtral_llm
from Phi import phi
from Vectorstore import get_pdf_text, get_chunks, get_vectorstore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

app = FastAPI()

# Directory to store uploaded PDFs
PDF_DIR = "uploaded_pdfs"

# Ensure the directory exists
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)

# Custom prompt template
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be 
a standalone question, in its original language. Chat History: {chat_history} Follow Up Input: {question} Standalone 
question:"""

Standalone_Question_Prompt = PromptTemplate.from_template(custom_template)

# def load_saved_pdfs():
#     return [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
#
# # Save uploaded PDFs to the local directory
# def save_uploaded_pdf(file: UploadFile):
#     file_path = os.path.join(PDF_DIR, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(file.file.read())
#     return file_path

# Function to get LLM based on the choice
def get_llm(llm_choice: str):
    if llm_choice == "Llama 3.1":
        return llama3
    elif llm_choice == "Mixtral":
        return mixtral_llm
    else:
        return phi


# Function to create conversation chain
def get_conversation_chain(vectorstore, llm_choice: str):
    llm = get_llm(llm_choice)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=Standalone_Question_Prompt,
        memory=memory
    )
    return conversation_chain


# Model for the question input
class QuestionInput(BaseModel):
    question: str
    llm_choice: str


# Upload PDF endpoint
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        # save_uploaded_pdf(file)
        file_path = os.path.join(PDF_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_files.append(file_path)
    return {"Uploaded PDFs": saved_files}

# Process selected PDFs
@app.post("/process_pdfs/")
async def process_pdfs(llm_choice: str = Form(...), pdf_files: List[str] = Form(...)):
    all_selected_files = [os.path.join(PDF_DIR, pdf) for pdf in pdf_files]

    raw_text = get_pdf_text(all_selected_files)
    text_chunks = get_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)

    # Create conversation chain
    conversation_chain = get_conversation_chain(vectorstore, llm_choice)

    # Save the chain to session or globally (e.g., database)
    app.state.conversation = conversation_chain

    return {"status": "PDFs processed successfully", "llm": llm_choice}


# Ask a question endpoint
@app.post("/ask_question/")
async def ask_question(question_input: QuestionInput):
    if app.state.conversation is None:
        return JSONResponse(status_code=400, content={"error": "No conversation chain found. Process PDFs first."})

    response = app.state.conversation({'question': question_input.question})
    return {"answer": response["answer"], "chat_history": response["chat_history"]}
