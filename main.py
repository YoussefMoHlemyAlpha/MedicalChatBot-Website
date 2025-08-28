import os
import requests
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms.base import LLM
import httpx
from bs4 import BeautifulSoup

app = FastAPI(title="Medical ChatBot", description="AI-powered medical information chatbot")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for the vector store and embeddings
vector_store = None
embeddings = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []

class OllamaLLM(LLM):
    """Custom Ollama LLM wrapper for LangChain"""
    
    model: str = "llama2"
    base_url: str = "http://localhost:11434"
    
    def _call(self, prompt: str, stop: List[str] = None, **kwargs) -> str:
        """Call the Ollama API"""
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 512
                    }
                },
                timeout=60.0
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return "I apologize, but I'm having trouble processing your request right now."
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return "I apologize, but I'm currently unavailable. Please try again later."
    
    @property
    def _llm_type(self) -> str:
        return "ollama"

def fetch_medline_data():
    """Fetch medical data from MedlinePlus API"""
    medical_topics = [
        "diabetes", "cancer", "fever", "influenza", "depression", 
        "anemia", "obesity", "hypertension", "asthma", "arthritis"
    ]
    
    documents = []
    
    for topic in medical_topics:
        try:
            # MedlinePlus API endpoint
            url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={topic}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Parse the XML response
                soup = BeautifulSoup(response.content, 'xml')
                
                for document in soup.find_all('document'):
                    title = document.find('content', {'name': 'title'})
                    summary = document.find('content', {'name': 'FullSummary'})
                    
                    if title and summary:
                        content = f"Title: {title.text}\n\nSummary: {summary.text}"
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": "MedlinePlus", "topic": topic}
                        ))
                        
        except Exception as e:
            print(f"Error fetching data for {topic}: {e}")
            # Add fallback content for each topic
            fallback_content = get_fallback_content(topic)
            documents.append(Document(
                page_content=fallback_content,
                metadata={"source": "Fallback", "topic": topic}
            ))
    
    return documents

def get_fallback_content(topic: str) -> str:
    """Provide fallback medical content when API is unavailable"""
    fallback_data = {
        "diabetes": "Diabetes is a chronic condition that affects how your body processes glucose (sugar). There are two main types: Type 1 (autoimmune) and Type 2 (insulin resistance). Management includes blood sugar monitoring, healthy diet, regular exercise, and medication as prescribed by healthcare providers.",
        "cancer": "Cancer is a disease where cells grow uncontrollably and can spread to other parts of the body. There are many types of cancer, each requiring specific treatment approaches including surgery, chemotherapy, radiation, or immunotherapy. Early detection through regular screening is crucial.",
        "fever": "Fever is a temporary increase in body temperature, often due to illness. Normal body temperature is around 98.6°F (37°C). Fever helps fight infections but should be monitored, especially in children and elderly. Seek medical attention for high fever or fever with severe symptoms.",
        "influenza": "Influenza (flu) is a contagious respiratory illness caused by flu viruses. Symptoms include fever, cough, body aches, and fatigue. Annual vaccination is recommended. Treatment may include antiviral medications if started early.",
        "depression": "Depression is a mood disorder characterized by persistent sadness, loss of interest, and other symptoms that interfere with daily life. It's a treatable condition with therapy, medication, or combination approaches. Professional help is important for proper diagnosis and treatment.",
        "anemia": "Anemia occurs when you don't have enough healthy red blood cells or hemoglobin. Common symptoms include fatigue, weakness, and pale skin. Causes include iron deficiency, chronic diseases, or genetic factors. Treatment depends on the underlying cause.",
        "obesity": "Obesity is a complex condition involving excessive body fat that increases health risks. It's measured using BMI (Body Mass Index). Management includes healthy eating, physical activity, behavior changes, and sometimes medical intervention.",
        "hypertension": "Hypertension (high blood pressure) is when blood pressure readings are consistently above normal (140/90 mmHg). Often called 'silent killer' as it may have no symptoms. Management includes lifestyle changes and medications as prescribed.",
        "asthma": "Asthma is a chronic respiratory condition causing breathing difficulties due to airway inflammation and narrowing. Symptoms include wheezing, coughing, and shortness of breath. Management includes avoiding triggers and using prescribed medications.",
        "arthritis": "Arthritis involves joint inflammation causing pain and stiffness. Common types include osteoarthritis (wear and tear) and rheumatoid arthritis (autoimmune). Treatment focuses on pain management, maintaining joint function, and slowing progression."
    }
    
    return f"Title: {topic.capitalize()}\n\nSummary: {fallback_data.get(topic, 'General medical information about ' + topic + '. Please consult with healthcare providers for specific medical advice.')}"

def initialize_vector_store():
    """Initialize the FAISS vector store with medical data"""
    global vector_store, embeddings
    
    try:
        print("Fetching medical data...")
        documents = fetch_medline_data()
        
        print("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        print(f"Creating vector store with {len(split_docs)} document chunks...")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        print("Vector store initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise e

def get_relevant_context(query: str, k: int = 3) -> List[Document]:
    """Retrieve relevant documents for the query"""
    if vector_store is None:
        return []
    
    try:
        docs = vector_store.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

def generate_response(query: str, context_docs: List[Document]) -> str:
    """Generate response using Ollama LLM"""
    try:
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt
        prompt = f"""You are a helpful medical information assistant. Based on the following medical information, please answer the user's question. Always remind users to consult with healthcare providers for personal medical concerns.

Medical Information:
{context}

User Question: {query}

Please provide a helpful and accurate response based on the medical information provided. If the information is not sufficient to answer the question, please say so and recommend consulting a healthcare provider.

Response:"""

        # Initialize Ollama LLM
        llm = OllamaLLM()
        
        # Generate response
        response = llm._call(prompt)
        
        # Add disclaimer
        disclaimer = "\n\n⚠️ **Important**: This information is for educational purposes only. Please consult with a licensed healthcare provider for personal medical concerns, diagnosis, or treatment decisions."
        
        return response + disclaimer
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm currently experiencing technical difficulties. Please try again later or consult with a healthcare provider for medical information."

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    try:
        initialize_vector_store()
    except Exception as e:
        print(f"Failed to initialize application: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat messages and return responses"""
    try:
        query = request.message.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get relevant context
        context_docs = get_relevant_context(query)
        
        # Generate response
        response = generate_response(query, context_docs)
        
        # Extract sources
        sources = [doc.metadata.get("topic", "Medical Database") for doc in context_docs]
        sources = list(set(sources))  # Remove duplicates
        
        return ChatResponse(response=response, sources=sources)
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "vector_store_ready": vector_store is not None}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
