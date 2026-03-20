"""
RAG-based PDF Document Q&A Chatbot
=====================================
End-to-end Retrieval Augmented Generation (RAG) pipeline
that answers questions from uploaded PDF documents.

Flow:
1. Upload PDF → Extract text → Split into chunks
2. Generate embeddings using HuggingFace
3. Store embeddings in FAISS vector database
4. User asks question → Embed question → Search FAISS
5. Retrieved chunks + question → LLM → Answer

Tech Stack:
- LangChain — RAG pipeline orchestration
- HuggingFace — sentence-transformers embeddings (free)
- FAISS — vector database for similarity search
- flan-t5-base — free LLM for answer generation
- FastAPI — REST API backend
- PyPDF — PDF text extraction

Author: Gourav Yadav
"""

import os
import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline as hf_pipeline
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL        = "google/flan-t5-base"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
TOP_K            = 3
UPLOAD_DIR       = "uploaded_pdfs"
VECTORSTORE_DIR  = "vectorstore"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline for PDF Q&A.
    - Loads and indexes PDF documents
    - Answers questions using retrieved context + LLM
    """

    def __init__(self):
        self.vectorstore = None
        self.qa_chain    = None
        self._load_models()

    def _load_models(self):
        """Load embedding model and LLM once at startup."""
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        logger.info("Loading LLM...")
        device = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            max_new_tokens=256,
            device=device
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Models loaded.")

    def index_pdf(self, pdf_path: str) -> int:
        """Load PDF, chunk, embed, store in FAISS. Returns chunk count."""
        logger.info(f"Indexing: {pdf_path}")

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs   = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} pages")

        # Store in FAISS
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            new_store = FAISS.from_documents(chunks, self.embeddings)
            self.vectorstore.merge_from(new_store)

        self.vectorstore.save_local(VECTORSTORE_DIR)
        self._build_chain()
        return len(chunks)

    def _build_chain(self):
        """Build LangChain RetrievalQA chain with custom prompt."""
        prompt_template = """Use the context below to answer the question.
If the answer is not in the context, say "I could not find this in the documents."

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": TOP_K}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def load_existing(self) -> bool:
        """Load previously saved vectorstore from disk."""
        vs_path = Path(VECTORSTORE_DIR)
        if vs_path.exists() and any(vs_path.iterdir()):
            self.vectorstore = FAISS.load_local(
                VECTORSTORE_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self._build_chain()
            logger.info("Loaded existing vectorstore.")
            return True
        return False

    def answer(self, question: str) -> dict:
        """Answer a question using RAG pipeline."""
        if self.qa_chain is None:
            raise ValueError("No documents indexed. Upload a PDF first.")

        result  = self.qa_chain({"query": question})
        sources = [
            {
                "page": doc.metadata.get("page", "unknown"),
                "preview": doc.page_content[:150] + "..."
            }
            for doc in result.get("source_documents", [])
        ]

        return {
            "question": question,
            "answer":   result["result"],
            "sources":  sources
        }


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG PDF Q&A Chatbot",
    description="Upload PDFs and ask questions using RAG + HuggingFace",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()


@app.on_event("startup")
async def startup():
    loaded = rag.load_existing()
    if loaded:
        logger.info("Existing vectorstore loaded.")
    else:
        logger.info("No vectorstore found. Upload a PDF to get started.")


# ── Schemas ───────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {"question": "What is the main topic of this document?"}
        }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status":  "running",
        "message": "RAG PDF Chatbot API",
        "docs":    "Visit /docs to test the API"
    }

@app.get("/health")
async def health():
    return {
        "status":          "healthy",
        "models_loaded":   rag.embeddings is not None,
        "vectorstore_ready": rag.vectorstore is not None,
        "qa_chain_ready":  rag.qa_chain is not None
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        chunks = rag.index_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message":        "PDF uploaded and indexed successfully.",
        "filename":       file.filename,
        "chunks_indexed": chunks
    }

@app.post("/ask")
async def ask(request: QuestionRequest):
    """Ask a question about uploaded PDFs."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag.answer(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)