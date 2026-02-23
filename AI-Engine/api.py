from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import get_vector_db, hybrid_search
from app import ask_mistral

# 1. Initialize FastAPI with your new name: VeraBot
api = FastAPI(
    title="VeraBot API",
    description="Backend engine for the VeraBot Tunisian Legal Assistant",
    version="1.0.0"
)

# --- NEW: Enable CORS ---
# This allows your frontend (React, Vue, or simple HTML) to connect to this API
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Global variable for the DB
db = None

@api.on_event("startup")
def load_rag():
    global db
    print("🚀 Initializing VeraBot Knowledge Base...")
    db = get_vector_db()

# 3. Define the request structure
class LegalQuery(BaseModel):
    question: str

# 4. The main endpoint
@api.post("/ask")
def query_legal_bot(item: LegalQuery):
    if not db:
        raise HTTPException(status_code=500, detail="VeraBot is not initialized yet")
    
    print(f"📩 Received question for VeraBot: {item.question}")
    
    # Run the RAG pipeline
    docs = hybrid_search(item.question, db)
    answer = ask_mistral(item.question, docs)
    
    # Return a structured JSON response
    return {
        "bot_name": "VeraBot",
        "question": item.question,
        "answer": answer,
        "sources": list(set([d.metadata.get('source') for d in docs]))
    }

# 5. Health Check
@api.get("/")
def home():
    return {"status": "VeraBot is online and ready for legal queries"}