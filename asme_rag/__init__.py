"""asme_rag: offline retrieval-augmented Q&A over licensed ASME PDFs.

Purpose: package root. Run with `python -m asme_rag <cmd>`.
Data flow: pdf_parser -> chunker -> storage -> rag -> cli.
How to change: add modules here; register commands in cli.py.
"""
__version__ = "0.1.0"
