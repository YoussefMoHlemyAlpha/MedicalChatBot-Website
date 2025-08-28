from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from utilis import fetch_dataset, build_or_load_vector_store, get_retrieval_chain

app = FastAPI()

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (JS, CSS) from src/
app.mount("/static", StaticFiles(directory="src"), name="static")

# Root route → serve index.html
@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("src/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Load dataset + vector store
results = fetch_dataset()
vector_store = build_or_load_vector_store(results)
qa_chain = get_retrieval_chain(vector_store, temperature=0.2)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    try:
        response = qa_chain.invoke({"query": query.question})
        return {
            "answer": response["result"],
            "sources": [doc.metadata for doc in response["source_documents"]]
        }
    except Exception as e:
        return {"error": str(e)}
