from dotenv import load_dotenv
import os
from huggingface_hub import login

# Load variables from .env
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

login(hf_token)