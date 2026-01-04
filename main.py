"""
Entry point for running the FastAPI application.
Run with: uvicorn main:app --reload
"""

from app.main import app

__all__ = ["app"]


if __name__ == "__main__":
    main()
