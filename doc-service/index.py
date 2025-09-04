# --- Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- File Paths ---
PDF_PATH = "pdf/faq.pdf" 
DB_DIR = "db"

# --- 1. Load the PDF Document ---
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Successfully loaded {len(documents)} page(s) from the PDF.")

# --- 2. Split the Document into Chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
texts = text_splitter.split_documents(documents)
print(f"Split the document into {len(texts)} chunks.")

# --- 3. Create Embeddings ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 4. Create and Save the Vector Database --- 
db = Chroma.from_documents(texts, embeddings, persist_directory=DB_DIR)
print(f"Successfully created the vector database and saved it to '{DB_DIR}'.")
