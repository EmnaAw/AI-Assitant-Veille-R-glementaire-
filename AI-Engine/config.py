import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("HF_MISTRAL")
EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"
DATA_DIR = "./data"
DB_DIR = "./db"