import os
import typer
from loader import load_documents
from embedder import create_vector_store
from retriever import query_docs
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

doc = typer.Typer()


from dotenv import load_dotenv
load_dotenv()


DOCS_FOLDER = os.getenv("DOCS_FOLDER", "data/docs")

llm = Ollama(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model="deepseek-r1"    
)

@doc.command()
def index(
    folder: str = DOCS_FOLDER
):
    """
    Load documents from a folder, embed them, and store in Supabase.
    """
    typer.echo(f"🔍 Loading documents from {folder}")
    docs = load_documents(folder)
    typer.echo(f"📦 Loaded {len(docs)} documents.")

    typer.echo("🚀 Creating vector store in Supabase...")
    vectordb = create_vector_store(docs)
    typer.secho("✅ Indexing complete.", fg=typer.colors.GREEN)

@doc.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask your doc-qa bot."),
    k: int = typer.Option(3, help="Number of top similar docs to retrieve.")
):
    """
    Query the indexed docs and get an answer.
    """
    typer.echo(f"❓ Query: {question}")
    results = query_docs(question, k=k)
    typer.secho("📚 Top retrieved chunks:", fg=typer.colors.BLUE)
    for i, doc in enumerate(results, start=1):
        typer.echo(f"---\nChunk {i} (source: {doc.metadata.get('source')}):\n{doc.page_content[:500]}\n...")


    
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa.run(question)
    typer.secho(f"\n🤖 Answer:\n{answer}", fg=typer.colors.MAGENTA)

if __name__ == "__main__":
    doc()
