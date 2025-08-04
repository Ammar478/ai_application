from pathlib import Path
from typing import List
import magic
import pdfplumber
from langchain.schema import Document

def load_documents(folder_path: str) -> List[Document]:
    """
    Recursively load all supported documents from a directory,
    returning a list of LangChain Document objects.

    Supports:
      - PDF (.pdf)
      - Markdown / Text (.md, .txt)
    """
    documents: List[Document] = []
    root = Path(folder_path)

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            #  PDF
            pages = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages.append(text)
            content = "\n".join(pages)

        elif suffix in (".md", ".txt"):  
            content = file_path.read_text(encoding="utf-8")

        else:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={"source": str(file_path)}
            )
        )

    return documents
