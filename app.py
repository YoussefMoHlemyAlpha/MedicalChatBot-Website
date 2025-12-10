from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from utilis import fetch_dataset, build_or_load_vector_store, get_retrieval_chain
import logging
import concurrent.futures

# Configure logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f"Received query: {query.question}")
        
        # Run chain in a separate thread with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(qa_chain.invoke, {"query": query.question})
            try:
                response = future.result(timeout=300) # 300 seconds timeout
            except concurrent.futures.TimeoutError:
                logger.error("Request timed out waiting for LLM response.")
                return {"error": "Response timed out. The model is taking too long to generate an answer."}

        logger.info("Chain invocation successful.")
        logger.info(f"Response result: {response.get('result', '')[:100]}...") # Log first 100 chars
        return {
            "answer": response["result"],
            "sources": [doc.metadata for doc in response["source_documents"]]
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        str_error = str(e)
        if "No connection could be made" in str_error or "10061" in str_error or "Connection refused" in str_error:
             return {"error": "Authentication/Connection Error. Is the Ollama (LLM) server running? Please start it to use the chatbot."}
        return {"error": str(e)}
