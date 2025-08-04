from typing import List
import os
from supabase import create_client
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from config import SUPABASE_URL, SUPABASE_KEY, TABLE_NAME, EMBEDDING_COLUMN, CONTENT_COLUMN, ID_COLUMN
from langchain.schema import Document

def query_docs(
query: str,
k: int = 3
) -> List[Document]:
    """
    Embed the query via Ollama, retrieve top-k similar documents from Supabase, and return them.
    """

    embedder = OllamaEmbeddings(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model=os.getenv("OLLAMA_MODEL_EMBED"),
    )
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    vectordb = SupabaseVectorStore(
        embedding=embedder,
        client=supabase_client,
        table_name=TABLE_NAME,
        embedding_column_name=EMBEDDING_COLUMN,
        content_column_name=CONTENT_COLUMN,
        id_column_name=ID_COLUMN,
    )

    docs_and_scores = vectordb.similarity_search_with_score(
        query,
        k=k
    )
    return [doc for doc, score in docs_and_scores]