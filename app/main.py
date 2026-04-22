from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pdf_utils import extract_text_from_pdf, chunk_text
from app.rag import get_embedding, retrieve_relevant_chunks, generate_answer_with_euron
import os
from rich import print
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import pathlib

app = FastAPI(
    title="RAG-based PDF QA System",
    description="Ask questions about the supplied PDF document.",
    version="1.0"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

PDF_PATH = "document.pdf"

chunks = []
chunk_embeddings = []

class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    """
    On application startup:
    - Check if the PDF file exists
    - Extract text from PDF
    - Chunk the text
    - Generate embeddings for each chunk
    """
    global chunks, chunk_embeddings
    if not os.path.exists(PDF_PATH):
        print(f"[bold red]ERROR:[/bold red] PDF file '{PDF_PATH}' not found.")
        raise FileNotFoundError(f"PDF file '{PDF_PATH}' not found. Please add it to the project root.")
    
    print(f"[bold blue]Extracting text from PDF: {PDF_PATH}[/bold blue]")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"[bold blue]Chunking text...[/bold blue]")
    chunks = chunk_text(text)
    print(f"[bold blue]Generating embeddings for chunks...[/bold blue]")
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
    print(f"[bold green]Setup complete. Ready to answer questions![/bold green]")
    
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = pathlib.Path("app/static/index.html")
    return FileResponse(index_path)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    relevant_chunks = retrieve_relevant_chunks(question, chunks, chunk_embeddings)
    try:
        answer = generate_answer_with_euron(relevant_chunks, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = {
        "question": question,
        "answer": answer
    }
    return response