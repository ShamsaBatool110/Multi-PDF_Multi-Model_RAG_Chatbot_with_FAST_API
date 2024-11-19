from langchain_huggingface import HuggingFaceEndpoint
import os

HUGGINGFACEHUB_API_TOKEN = 'your hugginfgae access token'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

mixtral_llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.01,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
