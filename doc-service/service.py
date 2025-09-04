# For our web server
from fastapi import FastAPI
from pydantic import BaseModel

# For loading the vector database and searching it
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --------------------------------------------------------------------------
# 1. Initialize the FastAPI application
# --------------------------------------------------------------------------
app = FastAPI()

# --------------------------------------------------------------------------
# 2. Define constants for our file paths and models
# --------------------------------------------------------------------------
DB_DIR = "db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------------------------------
# 3. Load the Database and Create a Retriever on Startup
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
# 5. Create the API endpoint
# --------------------------------------------------------------------------
@app.post("/ask")
async def ask(query: Query):
    """
    Receives a question, searches the database, and returns the answer.
    """
    # Get the user's question from the request
    user_question = query.question

    # Use the retriever directly to find the most relevant documents
    relevant_docs = retriever.invoke(user_question)
    
    # Extract the content from the best matching document
    answer = relevant_docs[0].page_content
    source = relevant_docs[0].metadata.get('source', 'Unknown')

    return {"answer": answer, "source": source}