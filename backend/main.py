from pathlib import Path
import shutil
import uuid
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_utils import (
    extract_pdf_text,
    create_chunks_from_pages,
    store_chunks_in_chromadb,
    save_chunks_to_json,
    search_similar_chunks,
    rerank_chunks_with_gemini,
    hybrid_search_chunks,
    generate_answer_with_gemini,
    generate_summary_with_gemini,
    check_faithfulness_with_gemini
)


app = FastAPI(title="Advanced Multilingual RAG Research Assistant")


# -----------------------------
# CORS setup
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Upload folder
# -----------------------------
UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)


# -----------------------------
# Request models
# -----------------------------
class AskRequest(BaseModel):
    paper_id: str
    question: str
    advanced_mode: bool = False


class SummaryRequest(BaseModel):
    paper_id: str


# -----------------------------
# Home route
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Backend is running successfully",
        "project": "Advanced Multilingual RAG Research Assistant",
        "available_endpoints": [
            "/upload-pdf",
            "/search-test",
            "/ask",
            "/summary"
        ]
    }


# -----------------------------
# Upload PDF endpoint
# -----------------------------
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF, extract text, chunk it, create embeddings,
    and store chunks in ChromaDB.
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    paper_id = str(uuid.uuid4())
    saved_filename = f"{paper_id}_{file.filename}"
    file_path = UPLOAD_DIR / saved_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pages = extract_pdf_text(str(file_path))

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No readable text found. This PDF may be scanned or image-based."
            )

        chunks = create_chunks_from_pages(
            pages=pages,
            paper_id=paper_id,
            filename=file.filename
        )

        total_chunks = store_chunks_in_chromadb(chunks)
        save_chunks_to_json(chunks)
        total_characters = sum(len(page["text"]) for page in pages)

        return {
            "message": "PDF uploaded, chunked, embedded, and stored successfully",
            "paper_id": paper_id,
            "original_filename": file.filename,
            "saved_filename": saved_filename,
            "total_pages_with_text": len(pages),
            "total_characters": total_characters,
            "total_chunks_stored": total_chunks,
            "preview_chunk": chunks[0] if chunks else None
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while processing PDF: {str(e)}"
        )


# -----------------------------
# Search test endpoint
# -----------------------------
@app.get("/search-test")
def search_test(paper_id: str, question: str):
    """
    Test whether ChromaDB retrieval is working.
    This only returns retrieved chunks, not final Gemini answer.
    """

    try:
        results = hybrid_search_chunks(
            question=question,
            paper_id=paper_id,
            top_k=5
        )

        return {
            "question": question,
            "paper_id": paper_id,
            "retrieved_chunks_count": len(results),
            "retrieved_chunks": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during search: {str(e)}"
        )


# -----------------------------
# Ask question endpoint
# -----------------------------
@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Retrieve relevant chunks and generate final answer.
    Fast mode: hybrid retrieval + answer
    Advanced mode: hybrid retrieval + reranking + faithfulness check
    """

    try:
        advanced_mode = getattr(request, "advanced_mode", False)

        if advanced_mode:
            retrieved_chunks = hybrid_search_chunks(
                question=request.question,
                paper_id=request.paper_id,
                top_k=10
            )

            retrieved_chunks = rerank_chunks_with_gemini(
                question=request.question,
                chunks=retrieved_chunks,
                top_k=5
            )
        else:
            retrieved_chunks = hybrid_search_chunks(
                question=request.question,
                paper_id=request.paper_id,
                top_k=5
            )

        result = generate_answer_with_gemini(
            question=request.question,
            retrieved_chunks=retrieved_chunks
        )

        if advanced_mode:
            faithfulness_result = check_faithfulness_with_gemini(
                question=request.question,
                answer=result["answer"],
                retrieved_chunks=retrieved_chunks
            )
        else:
            faithfulness_result = {
                "status": "SKIPPED_FAST_MODE",
                "explanation": "Faithfulness check skipped in fast mode to reduce response time."
            }

        return {
            "question": request.question,
            "paper_id": request.paper_id,
            "answer": result["answer"],
            "source_pages": result["source_pages"],
            "sources": result["sources"],
            "faithfulness": faithfulness_result,
            "mode": "advanced" if advanced_mode else "fast"
        }

    except Exception as e:
        print("ERROR IN /ask:")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error while generating answer: {str(e)}"
        )
# -----------------------------
# Summary endpoint
# -----------------------------
@app.post("/summary")
def summarize_paper(request: SummaryRequest):
    """
    Generate structured summary of the uploaded paper using Gemini.
    """

    try:
        result = generate_summary_with_gemini(
            paper_id=request.paper_id
        )

        return {
            "paper_id": request.paper_id,
            "summary": result["summary"],
            "source_pages": result["source_pages"]
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while generating summary: {str(e)}"
        )