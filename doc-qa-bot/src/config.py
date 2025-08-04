
"""
Configuration settings for Doc-QA Bot.
Loads environment variables and provides defaults.
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Directory containing documents to index
DOCS_FOLDER: str = os.getenv("DOCS_FOLDER", "data/docs")

# Supabase configuration
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "http://localhost:5432")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

# Vector store settings
TABLE_NAME: str = os.getenv("VECTOR_TABLE_NAME", "doc_qa_aiapplication")
EMBEDDING_COLUMN: str = os.getenv("EMBEDDING_COLUMN", "embedding")
CONTENT_COLUMN: str = os.getenv("CONTENT_COLUMN", "content_doc_qa_aiapplication")
ID_COLUMN: str = os.getenv("ID_COLUMN", "qa_id")
