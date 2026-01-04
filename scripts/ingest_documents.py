import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embeddings.document_processor import DocumentProcessor
from app.embeddings.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.models.config import settings
from app.core.logging_config import setup_logging, get_logger


setup_logging(log_level="INFO")
logger = get_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Starting document ingestion for Agrobot")
    logger.info("=" * 60)

    logger.info("Initializing services...")
    embedding_service = EmbeddingService()
    doc_processor = DocumentProcessor()
    vector_store = VectorStore(embedding_service=embedding_service)

    vector_store.initialize()
    logger.info("Vector store initialized")

    project_root = Path(__file__).parent.parent
    pdf_dir = project_root / "data" / "pdfs"

    logger.info(f"Looking for PDFs in: {pdf_dir}")

    diseases_pdf = pdf_dir / settings.diseases_pdf
    schemes_pdf = pdf_dir / settings.schemes_pdf

    if not diseases_pdf.exists():
        logger.error(f"Diseases PDF not found: {diseases_pdf}")
        logger.info(f"Please download CitrusPlantPestsAndDiseases.pdf to {pdf_dir}")

    if not schemes_pdf.exists():
        logger.error(f"Schemes PDF not found: {schemes_pdf}")
        logger.info(f"Please download GovernmentSchemes.pdf to {pdf_dir}")

    if not diseases_pdf.exists() and not schemes_pdf.exists():
        logger.error("No PDF files found. Exiting.")
        return

    if diseases_pdf.exists():
        logger.info("-" * 40)
        logger.info(f"Processing: {diseases_pdf.name}")

        vector_store.delete_collection("disease")
        vector_store.initialize()

        disease_chunks = doc_processor.process_pdf(diseases_pdf, "disease")
        logger.info(f"Created {len(disease_chunks)} chunks from diseases PDF")

        if disease_chunks:
            ids = vector_store.add_documents(disease_chunks, "disease")
            logger.info(f"Added {len(ids)} documents to 'citrus_diseases' collection")

    if schemes_pdf.exists():
        logger.info("-" * 40)
        logger.info(f"Processing: {schemes_pdf.name}")

        vector_store.delete_collection("scheme")
        vector_store.initialize()

        scheme_chunks = doc_processor.process_pdf(schemes_pdf, "scheme")
        logger.info(f"Created {len(scheme_chunks)} chunks from schemes PDF")

        if scheme_chunks:
            ids = vector_store.add_documents(scheme_chunks, "scheme")
            logger.info(
                f"Added {len(ids)} documents to 'government_schemes' collection"
            )

    logger.info("-" * 40)
    logger.info("Ingestion complete! Collection statistics:")
    stats = vector_store.get_collection_stats()
    for collection, count in stats.items():
        logger.info(f"  - {collection}: {count} documents")

    logger.info("=" * 60)
    logger.info("Document ingestion completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
