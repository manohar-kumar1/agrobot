import re
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.models.config import settings
from app.utils.text_processing import clean_text


class DocumentProcessor:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_pdf(self, file_path: Path) -> List[Document]:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        for doc in documents:
            doc.page_content = self._preprocess_text(doc.page_content)
            doc.metadata["source_file"] = file_path.name

        return documents

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = clean_text(text)
        return text.strip()

    def chunk_documents(
        self, documents: List[Document], collection_type: str
    ) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["collection_type"] = collection_type
            chunk.metadata["chunk_index"] = i
            source = chunk.metadata.get("source_file", "unknown")
            page = chunk.metadata.get("page", 0)
            chunk.metadata["chunk_id"] = f"{source}_page{page}_chunk{i}"

        return chunks

    def process_pdf(self, pdf_path: Path, collection_type: str) -> List[Document]:
        documents = self.load_pdf(pdf_path)
        chunks = self.chunk_documents(documents, collection_type)
        return chunks

    def process_all_pdfs(self, pdf_dir: Path) -> Dict[str, List[Document]]:
        result = {"diseases": [], "schemes": []}

        diseases_pdf = pdf_dir / settings.diseases_pdf
        schemes_pdf = pdf_dir / settings.schemes_pdf

        if diseases_pdf.exists():
            result["diseases"] = self.process_pdf(diseases_pdf, "disease")

        if schemes_pdf.exists():
            result["schemes"] = self.process_pdf(schemes_pdf, "scheme")

        return result
