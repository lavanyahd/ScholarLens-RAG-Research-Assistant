import re
import os
from typing import List, Dict, Any

import json
from pathlib import Path
from rank_bm25 import BM25Okapi

import fitz  # PyMuPDF
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# -----------------------------
# Load multilingual embedding model
# -----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedding_model = None

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        print("Loading embedding model for the first time...")
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder="/tmp/sentence_transformers"
        )
        print("Embedding model loaded successfully.")

    return _embedding_model

# -----------------------------
# ChromaDB setup
# -----------------------------
chroma_client = chromadb.PersistentClient(path="chroma_db")
CHUNKS_FILE = Path("paper_chunks.json")

collection = chroma_client.get_or_create_collection(
    name="research_papers",
    metadata={"hnsw:space": "cosine"}
)


# -----------------------------
# Gemini client setup
# -----------------------------
def get_gemini_client():
    """
    Create Gemini client only when needed.
    This avoids crashing the full backend if API key is missing.
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it in your .env file."
        )

    return genai.Client(api_key=GEMINI_API_KEY)


# -----------------------------
# PDF text extraction
# -----------------------------
def extract_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF page by page.
    Returns list of dictionaries containing page number and text.
    """

    pages = []

    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()

        if text and text.strip():
            pages.append({
                "page_number": page_number,
                "text": text.strip()
            })

    doc.close()
    return pages


# -----------------------------
# Text chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 250) -> List[str]:
    """
    Split large text into smaller overlapping chunks.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


def create_chunks_from_pages(
    pages: List[Dict[str, Any]],
    paper_id: str,
    filename: str
) -> List[Dict[str, Any]]:
    """
    Create chunks from extracted page-wise text.
    Each chunk stores metadata: paper_id, filename, page_number, chunk_index.
    """

    all_chunks = []

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{paper_id}_page_{page_number}_chunk_{index}",
                "paper_id": paper_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_index": index,
                "text": chunk
            })

    return all_chunks


# -----------------------------
# Store chunks in ChromaDB
# -----------------------------
def store_chunks_in_chromadb(chunks: List[Dict[str, Any]]) -> int:
    """
    Convert chunks into embeddings and store them in ChromaDB.
    """

    if not chunks:
        return 0

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "paper_id": chunk["paper_id"],
            "filename": chunk["filename"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"]
        }
        for chunk in chunks
    ]

    embeddings = get_embedding_model().encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    ).tolist()

    # upsert is safer than add because it avoids duplicate ID errors
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


# -----------------------------
# Search relevant chunks
# -----------------------------
def search_similar_chunks(
    question: str,
    paper_id: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search ChromaDB and return top relevant chunks for the question.
    """
    retrieval_question = expand_question_for_retrieval(question)
    question_embedding = get_embedding_model().encode(
        retrieval_question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where={"paper_id": paper_id}
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(documents)):
        retrieved_chunks.append({
            "text": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i]
        })

    return retrieved_chunks


# -----------------------------
# Build context for Gemini
# -----------------------------
def build_context_from_chunks(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Convert retrieved chunks into a clean context string for Gemini.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        page_number = chunk["metadata"]["page_number"]
        text = chunk["text"]

        context_parts.append(
            f"[Source {index} | Page {page_number}]\n{text}"
        )

    return "\n\n".join(context_parts)

# -----------------------------
# Detect answer language
# -----------------------------

def detect_answer_language(question: str) -> str:
    """
    Detect expected answer language from the user's question.
    Handles English, Hindi, Kannada, and Hinglish.
    Also handles commands like 'summarize in Kannada'.
    """

    q = question.strip().lower()

    # If user explicitly asks answer in Kannada
    if (
        "kannada" in q
        or "in kannada" in q
        or "ಕನ್ನಡ" in q
    ):
        return "Kannada"

    # If user explicitly asks answer in Hindi
    if (
        "hindi" in q
        or "in hindi" in q
        or "हिंदी" in q
    ):
        return "Hindi"

    # Kannada script detection
    if re.search(r"[\u0C80-\u0CFF]", q):
        return "Kannada"

    # Hindi script detection
    if re.search(r"[\u0900-\u097F]", q):
        return "Hindi"

    # Roman Hindi / Hinglish keywords
    hinglish_words = {
        "kya", "hai", "hey", "matlab", "ka", "ki", "ke",
        "batao", "samjhao", "kyun", "kaise", "mein", "me"
    }

    words = set(re.findall(r"\b\w+\b", q))

    if len(words.intersection(hinglish_words)) >= 2:
        return "Hinglish"

    return "English"

# -----------------------------
# Generate answer using Gemini
# -----------------------------
def generate_answer_with_gemini(
    question: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate final answer using Gemini based only on retrieved PDF chunks.
    If Gemini quota is exceeded, return a friendly fallback response.
    """

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in the uploaded paper.",
            "sources": [],
            "source_pages": [],
            "faithfulness": "No relevant context found"
        }

    context = build_context_from_chunks(retrieved_chunks)
    answer_language = detect_answer_language(question)

    prompt = f"""
You are an advanced multilingual research paper assistant.

Your task is to answer the user's question using ONLY the provided context from the uploaded research paper.

Expected answer language: {answer_language}

Important rules:
1. Answer ONLY in the expected answer language mentioned above.
2. If the user asks "in Kannada", "Kannada", or "ಕನ್ನಡ", answer in Kannada.
3. If the user asks "in Hindi" or "Hindi", answer in Hindi.
4. If the expected answer language is Hinglish, answer in simple Hinglish.
5. Do not use outside knowledge.
6. If the context contains enough information, answer clearly.
7. If the context partially answers the question, explain using the available context.
8. Only say "The uploaded paper does not provide enough information" when the context has no useful information at all.
9. Do not create fake facts.
10. Mention page numbers naturally in the answer.

Formatting rules:
- Use clean Markdown formatting.
- Use short headings.
- Use bullet points.
- Use bold text for important terms.
- Keep paragraphs short.
- Do not write one very long paragraph.
- Make the answer look like a professional AI assistant response.

If the user asks for a summary, use this structure:

## Simple Summary

Briefly explain what the paper is about.

## Main Objective

Explain the main aim of the paper.

## Key Points

- Point 1
- Point 2
- Point 3

## Why It Matters

Explain the importance in simple words.

## Source Pages

Mention the relevant page numbers.

Context from uploaded paper:
{context}

User question:
{question}

Final answer:
"""
    sources = []

    for chunk in retrieved_chunks:
        sources.append({
            "page_number": chunk["metadata"]["page_number"],
            "filename": chunk["metadata"]["filename"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "distance": chunk.get("distance", 0)
        })

    unique_pages = sorted(
        list(set(source["page_number"] for source in sources))
    )

    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return {
            "answer": response.text,
            "sources": sources,
            "source_pages": unique_pages,
            "faithfulness": "Answer generated only from retrieved paper context"
        }

    except Exception as e:
        error_text = str(e)

        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            return {
                "answer": (
                    "Gemini API quota is currently exceeded. "
                    "The relevant paper sections were retrieved successfully, "
                    "but the AI answer could not be generated right now. "
                    "Please wait and try again later, or use a Gemini API key with more quota."
                ),
                "sources": sources,
                "source_pages": unique_pages,
                "faithfulness": "Gemini quota exceeded; answer generation skipped."
            }

        return {
            "answer": f"Gemini answer generation failed: {str(e)}",
            "sources": sources,
            "source_pages": unique_pages,
            "faithfulness": "Answer generation failed."
        }

# -----------------------------
# Optional: structured summary
# -----------------------------
def generate_summary_with_gemini(paper_id: str) -> Dict[str, Any]:
    """
    Generate a structured summary of the uploaded paper.
    Uses retrieved broad chunks from the paper.
    """

    # Use a general query to retrieve important chunks
    summary_query = (
        "abstract introduction methodology results conclusion contributions limitations"
    )

    retrieved_chunks = search_similar_chunks(
        question=summary_query,
        paper_id=paper_id,
        top_k=8
    )

    if not retrieved_chunks:
        return {
            "summary": "No relevant content found for this paper.",
            "source_pages": []
        }

    context = build_context_from_chunks(retrieved_chunks)

    prompt = f"""
You are a research paper summarization assistant.

Using ONLY the given context, create a structured summary of the paper.

Include:
1. Title or topic
2. Problem statement
3. Main objective
4. Methodology or approach
5. Key contributions
6. Results or findings
7. Limitations
8. Future scope

If any section is not available in the context, write:
"Not clearly mentioned in the retrieved context."

Context:
{context}

Structured summary:
"""

    client = get_gemini_client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    source_pages = sorted(
        list(set(chunk["metadata"]["page_number"] for chunk in retrieved_chunks))
    )

    return {
        "summary": response.text,
        "source_pages": source_pages
    }
def save_chunks_to_json(chunks):
    """
    Save chunks to a local JSON file.
    This is needed for BM25 keyword search.
    """

    existing_chunks = []

    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)

    existing_chunks.extend(chunks)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_chunks, f, ensure_ascii=False, indent=2)

    return len(existing_chunks)


def load_chunks_for_paper(paper_id: str):
    """
    Load chunks of one specific paper from JSON file.
    """

    if not CHUNKS_FILE.exists():
        return []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    paper_chunks = [
        chunk for chunk in all_chunks
        if chunk["paper_id"] == paper_id
    ]

    return paper_chunks


def tokenize_text(text: str):
    """
    Simple tokenizer for BM25.
    Converts text into lowercase word tokens.
    """

    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def bm25_search_chunks(question: str, paper_id: str, top_k: int = 10):
    """
    Perform BM25 keyword search over saved chunks.
    """

    paper_chunks = load_chunks_for_paper(paper_id)

    if not paper_chunks:
        return []

    tokenized_corpus = [
        tokenize_text(chunk["text"]) for chunk in paper_chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_question = tokenize_text(question)

    scores = bm25.get_scores(tokenized_question)

    scored_chunks = []

    for chunk, score in zip(paper_chunks, scores):
        scored_chunks.append({
            "text": chunk["text"],
            "metadata": {
                "paper_id": chunk["paper_id"],
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"]
            },
            "bm25_score": float(score)
        })

    scored_chunks = sorted(
        scored_chunks,
        key=lambda x: x["bm25_score"],
        reverse=True
    )

    return scored_chunks[:top_k]


def hybrid_search_chunks(question: str, paper_id: str, top_k: int = 5):
    """
    Advanced hybrid retrieval:
    Combines vector search and BM25 keyword search.
    """

    vector_results = search_similar_chunks(
        question=question,
        paper_id=paper_id,
        top_k=10
    )

    bm25_results = bm25_search_chunks(
        question=question,
        paper_id=paper_id,
        top_k=10
    )

    combined = {}

    # Add vector results
    for rank, item in enumerate(vector_results):
        key = (
            item["metadata"]["page_number"],
            item["metadata"]["chunk_index"]
        )

        # Lower vector distance is better, so convert to score
        vector_score = 1 / (1 + item["distance"])

        combined[key] = {
            "text": item["text"],
            "metadata": item["metadata"],
            "vector_distance": item["distance"],
            "bm25_score": 0,
            "hybrid_score": vector_score + (1 / (rank + 1))
        }

    # Add BM25 results
    for rank, item in enumerate(bm25_results):
        key = (
            item["metadata"]["page_number"],
            item["metadata"]["chunk_index"]
        )

        bm25_score = item["bm25_score"]

        if key in combined:
            combined[key]["bm25_score"] = bm25_score
            combined[key]["hybrid_score"] += bm25_score + (1 / (rank + 1))
        else:
            combined[key] = {
                "text": item["text"],
                "metadata": item["metadata"],
                "vector_distance": None,
                "bm25_score": bm25_score,
                "hybrid_score": bm25_score + (1 / (rank + 1))
            }

    final_results = list(combined.values())

    final_results = sorted(
        final_results,
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    # Convert vector_distance to distance field for Gemini source output compatibility
    cleaned_results = []

    for item in final_results[:top_k]:
        cleaned_results.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "distance": item["vector_distance"] if item["vector_distance"] is not None else 0,
            "bm25_score": item["bm25_score"],
            "hybrid_score": item["hybrid_score"]
        })

    return cleaned_results
def expand_question_for_retrieval(question: str) -> str:
    """
    Expands short or Hinglish questions so retrieval becomes better.
    """

    q = question.strip()
    lower_q = q.lower()

    if "rag" in lower_q:
        return (
            q + " Retrieval-Augmented Generation meaning definition "
            "abstract introduction large language models"
        )

    return q

def rerank_chunks_with_gemini(question: str, chunks, top_k: int = 5):
    """
    Rerank retrieved chunks using Gemini.
    Gemini selects the most relevant chunks for the user's question.
    """

    if not chunks:
        return []

    client = get_gemini_client()

    chunk_texts = []

    for i, chunk in enumerate(chunks):
        page_number = chunk["metadata"]["page_number"]
        text = chunk["text"][:800]

        chunk_texts.append(
            f"Chunk {i}\nPage: {page_number}\nText: {text}"
        )

    chunks_for_prompt = "\n\n".join(chunk_texts)

    prompt = f"""
You are a reranking assistant for a RAG system.

The user question may be in English, Hindi, Hinglish, Kannada, or another language.

Your task:
Select the most relevant chunks that directly help answer the user's question.

User question:
{question}

Retrieved chunks:
{chunks_for_prompt}

Return ONLY the chunk numbers of the best {top_k} chunks in order of relevance.
Return as comma-separated numbers only.

Example:
0,2,4
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        selected_indexes = []

        for item in text.replace("\n", ",").split(","):
            item = item.strip()

            if item.isdigit():
                index = int(item)

                if 0 <= index < len(chunks):
                    selected_indexes.append(index)

        reranked_chunks = []

        for index in selected_indexes:
            if chunks[index] not in reranked_chunks:
                reranked_chunks.append(chunks[index])

        if reranked_chunks:
            return reranked_chunks[:top_k]

        return chunks[:top_k]

    except Exception:
        return chunks[:top_k]
    
def check_faithfulness_with_gemini(
    question: str,
    answer: str,
    retrieved_chunks
):
    """
    Check whether the generated answer is supported by the retrieved PDF context.
    """

    if not retrieved_chunks:
        return {
            "status": "NOT_SUPPORTED",
            "explanation": "No retrieved context was available to verify the answer."
        }

    context = build_context_from_chunks(retrieved_chunks)

    prompt = f"""
You are a strict faithfulness evaluator for a RAG system.

Your task is to check whether the generated answer is supported by the provided research paper context.

Rules:
1. Use ONLY the provided context.
2. Do not use outside knowledge.
3. Check whether the answer is factually supported by the context.
4. Return one of these labels:
   - SUPPORTED
   - PARTIALLY_SUPPORTED
   - NOT_SUPPORTED
5. Give a short explanation.

Context:
{context}

User question:
{question}

Generated answer:
{answer}

Return your response in this exact format:

Status: SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED
Explanation: short explanation here
"""

    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        status = "PARTIALLY_SUPPORTED"

        if "NOT_SUPPORTED" in text:
            status = "NOT_SUPPORTED"
        elif "SUPPORTED" in text and "PARTIALLY_SUPPORTED" not in text:
            status = "SUPPORTED"
        elif "PARTIALLY_SUPPORTED" in text:
            status = "PARTIALLY_SUPPORTED"

        return {
            "status": status,
            "explanation": text
        }

    except Exception as e:
        return {
            "status": "CHECK_FAILED",
            "explanation": f"Faithfulness check failed: {str(e)}"
        }    