from PyPDF2 import PdfReader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from langchain_chroma import Chroma
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS


# extracting text from pdf
def get_pdf_text(docs):
    documents_text = []
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        document = Document(page_content=text)
        documents_text.append(document)
    return documents_text


# converting text to chunks
def get_chunks(raw_text_documents):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_documents(raw_text_documents)
    return chunks


# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunked_text):
    # client = chromadb.Client()
    # if client.list_collections():
    #     consent_collection = client.create_collection("consent_collection")
    # else:
    #     print("Collection already exists")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    # vectordb = Chroma.from_documents(
    #     documents=chunked_text,
    #     embedding=embeddings
    # )
    vectordb = FAISS.from_documents(documents=chunked_text, embedding=embeddings)

    return vectordb
