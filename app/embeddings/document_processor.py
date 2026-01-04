"""
Document processing utilities for PDF parsing and chunking.
Handles loading and preprocessing of PDF documents.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.models.config import settings
from app.utils.text_processing import clean_text


class DocumentProcessor:
    """Processes PDF documents for embedding"""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """
        Initialize document processor.

        Args:
            chunk_size: Size of text chunks (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load and parse a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            List of LangChain Document objects with page content and metadata
        """
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        # Clean text content and add source metadata
        for doc in documents:
            doc.page_content = self._preprocess_text(doc.page_content)
            doc.metadata["source_file"] = file_path.name

        return documents

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before chunking.
        Normalizes whitespace and removes PDF artifacts.

        Args:
            text: Raw text from PDF

        Returns:
            Cleaned text
        """
        # Remove multiple newlines but keep paragraph structure
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove multiple spaces
        text = re.sub(r" {2,}", " ", text)
        # Remove page numbers and headers/footers patterns
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Clean basic artifacts while keeping content
        text = clean_text(text)
        return text.strip()

    def chunk_documents(
        self, documents: List[Document], collection_type: str
    ) -> List[Document]:
        """
        Split documents into chunks with metadata.

        Args:
            documents: List of LangChain Documents
            collection_type: Type of collection ('disease' or 'scheme')

        Returns:
            List of chunked Documents with enhanced metadata
        """
        chunks = self.text_splitter.split_documents(documents)

        # Enhance metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["collection_type"] = collection_type
            chunk.metadata["chunk_index"] = i
            # Create a unique ID for the chunk
            source = chunk.metadata.get("source_file", "unknown")
            page = chunk.metadata.get("page", 0)
            chunk.metadata["chunk_id"] = f"{source}_page{page}_chunk{i}"

        return chunks

    def process_pdf(self, pdf_path: Path, collection_type: str) -> List[Document]:
        """
        Full pipeline: load PDF and chunk it.

        Args:
            pdf_path: Path to PDF file
            collection_type: Type of collection ('disease' or 'scheme')

        Returns:
            List of processed and chunked Documents
        """
        documents = self.load_pdf(pdf_path)
        chunks = self.chunk_documents(documents, collection_type)
        return chunks

    def process_all_pdfs(self, pdf_dir: Path) -> Dict[str, List[Document]]:
        """
        Process all PDFs in a directory, categorizing by filename.

        Args:
            pdf_dir: Directory containing PDF files

        Returns:
            Dict with 'diseases' and 'schemes' keys containing chunked documents
        """
        result = {"diseases": [], "schemes": []}

        diseases_pdf = pdf_dir / settings.diseases_pdf
        schemes_pdf = pdf_dir / settings.schemes_pdf

        if diseases_pdf.exists():
            result["diseases"] = self.process_pdf(diseases_pdf, "disease")

        if schemes_pdf.exists():
            result["schemes"] = self.process_pdf(schemes_pdf, "scheme")

        return result
