from langchain_huggingface import HuggingFaceEndpoint
import os

HUGGINGFACEHUB_API_TOKEN = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_id = "microsoft/Phi-3.5-mini-instruct"

phi = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=250,
    temperature=0.01,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
