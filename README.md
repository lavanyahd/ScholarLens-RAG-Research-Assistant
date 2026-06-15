# ScholarLens: Advanced Multilingual RAG Research Paper Assistant

ScholarLens is a full-stack AI research paper assistant that allows users to upload academic PDFs and ask questions about the paper. It uses Retrieval-Augmented Generation to retrieve relevant paper sections and generate source-grounded answers with page references.

## Features

- PDF upload and text extraction
- Page-wise source tracking
- Text chunking
- Multilingual embeddings
- ChromaDB vector storage
- Hybrid semantic-keyword retrieval
- Gemini-based answer generation
- Source page references
- Research paper summary generation
- Multilingual and Hinglish question support
- Gemini quota fallback handling
- Professional React chatbot interface

## Tech Stack

### Frontend
- React
- Vite
- CSS
- Axios

### Backend
- Python
- FastAPI
- PyMuPDF
- ChromaDB
- Sentence Transformers
- Gemini API

## How to Run

### Backend

```bash
cd Backend
venv\Scripts\activate
uvicorn main:app --reload