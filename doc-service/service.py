import os
from dotenv import load_dotenv
import google.generativeai as genai

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- NEW: Load API Key from .env file ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------------------------------------------------------------
# 1. Initialize the FastAPI application and Gemini
# --------------------------------------------------------------------------
app = FastAPI()
genai.configure(api_key=GOOGLE_API_KEY)
# This is the new, correct line
model = genai.GenerativeModel('gemini-1.5-flash-latest')
# --------------------------------------------------------------------------
# 2. Define constants
# --------------------------------------------------------------------------
DB_DIR = "db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------------------------------
# 3. Load the Database and Retriever on Startup
# --------------------------------------------------------------------------
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = db.as_retriever()
print("Database loaded and retriever is ready.")

# --------------------------------------------------------------------------
# 4. Define the data model for the incoming request
# --------------------------------------------------------------------------
class Query(BaseModel):
    question: str

# --------------------------------------------------------------------------
# 5. Create the "Expert" API endpoint
# --------------------------------------------------------------------------
@app.post("/ask")
async def ask(query: Query):
    user_question = query.question
    
    # 1. Retrieve the most relevant text chunk
    context = retriever.invoke(user_question)[0].page_content

    # 2. Create the prompt for the Gemini model
    prompt = f"""
    Based on the context below, answer the user's question.
    Context: {context}
    Question: {user_question}
    Answer:"""

    # 3. Generate the answer using the Gemini model
    response = model.generate_content(prompt)
    
    return {"answer": response.text, "source": context}