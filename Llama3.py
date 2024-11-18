from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
os.environ.get('HUGGINGFACEHUB_API_TOKEN')

repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

llama3 = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.01,
    max_new_tokens=250
    )

