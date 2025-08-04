from typing import List
import os
from supabase import create_client
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from config import TABLE_NAME, EMBEDDING_COLUMN, CONTENT_COLUMN, ID_COLUMN, SUPABASE_URL, SUPABASE_KEY

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_URL")

def create_vector_store(
    documents:List,
 
) -> SupabaseVectorStore:
    """
    Embed documents via Ollama and persist into Supabase pgvector table.

    Returns the initialized SupabaseVectorStore instance.
    """
    embedder = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL_EMBED"),
    )
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)


    vectordb = SupabaseVectorStore.from_documents(
        documents,
        embedding=embedder,
        client=supabase_client,
        table_name=TABLE_NAME,
        embedding_column_name=EMBEDDING_COLUMN,
        content_column_name=CONTENT_COLUMN,
        id_column_name=ID_COLUMN,
    )

    return vectordb 

