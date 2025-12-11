import os
import re
import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Bio import Entrez
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
Entrez.email = "ahmed.agamy393@gmail.com"
DOC_PATH = "medical_rules.txt"
PERSIST_DIR = "./chroma_db"
MEDLINEPLUS_SEARCH = "https://wsearch.nlm.nih.gov/ws/query"

app = FastAPI(title="Medical Chatbot API")

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MEDICINE DB  ---
MEDICINE_PRICES = {
    "Panadol Extra": 12.00,
    "Aspirin 81mg": 5.50,
    "Telfast": 15.00,
    "Metformin 500mg": 8.00,
    "Insulin Pen": 45.00,
    "Tamoxifen 20mg": 25.00,
    "Imatinib 400mg": 100.00,
    "Orlistat 120mg": 30.00,
    "Prozac 20mg": 18.00,
    "Zoloft 50mg": 20.00
}

# --- DATA MODELS ---
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "guest"

class BotResponse(BaseModel):
    answer: str
    source: str = None
    action: dict = None  # New field for frontend actions

# --- MEDLINEPLUS FETCHER ---
def fetch_medlineplus_summary(url):
    """Extract the main readable summary from a MedlinePlus page"""
    try:
        headers = {'User-Agent': 'MedicalChatbot/1.0 (+your.email@example.com)'}
        r = requests.get(url, headers=headers, timeout=10)
        html = r.text

        # Look for the main summary section
        start = html.find('<h2>Summary</h2>')
        if start == -1:
            start = html.find('id="topic-summary"')
        if start == -1:
            return None

        # Extract text until next major section
        end = html.find('<h2>', start + 1)
        if end == -1:
            end = html.find('</section>', start)
        if end == -1:
            end = len(html)

        text = html[start:end]
        # Strip HTML tags
        text = re.sub('<.*?>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:1200] + ("..." if len(text) > 1200 else "")
    except:
        return None

def get_medlineplus_info(query, language="en"):
    """Get clean, patient-friendly info from MedlinePlus"""
    params = {
        "db": "healthTopics",
        "term": query,
        "rettype": "brief",
        "language": language
    }
    try:
        response = requests.get(MEDLINEPLUS_SEARCH, params=params, timeout=10)
        if response.status_code != 200:
            return None

        root = ET.fromstring(response.content)
        for content in root.findall(".//content"):
            title = content.get("title", "").strip()
            url = content.get("url", "").strip()
            if not url:
                continue

            # Fetch full page summary
            summary = fetch_medlineplus_summary(url)
            if query.lower() in title.lower() or (summary and query.lower() in summary.lower()):
                return {
                    "title": title,
                    "url": url,
                    "summary": summary or "No detailed summary available."
                }
        return None
    except:
        return None

# --- MEDICAL INFO HELPER ---
def get_medical_info(question):
    # 1. Try MedlinePlus first
    info = get_medlineplus_info(question)
    if info and info["summary"]:
        return (
            f"**{info['title']}**\n\n"
            f"{info['summary']}\n\n"
            f"Source: [MedlinePlus]({info['url']})"
        )

    # 2. Fallback: PubMed abstract
    try:
        handle = Entrez.esearch(db="pubmed", term=question, retmax=1, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        if record["IdList"]:
            pmid = record["IdList"][0]
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
            abstract = handle.read().strip()
            handle.close()
            return (
                f"**Latest Research Summary**\n\n"
                f"{abstract}\n\n"
                f"Source: [PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
            )
    except Exception as e:
        print(f"PubMed error: {e}")
        pass

    return "I couldn't find reliable, plain-language information on this topic right now."

# --- INITIALIZATION ---
print("Initializing ChromaDB and LLM...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

if not os.path.exists(PERSIST_DIR):
    print("First time setup: Indexing your rules document...")
    if os.path.exists(DOC_PATH):
        loader = TextLoader(DOC_PATH, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        print("Indexing complete! Knowledge base ready.")
    else:
        print(f"Warning: {DOC_PATH} not found. Starting with empty knowledge base.")
        # Create an empty vectorstore if doc missing (avoids crash, but no rules)
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    print("Loading existing knowledge base...")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM & Chain
llm = ChatOllama(model="llama3.2", temperature=0.3)

prompt_template = """
You are a kind, professional medical information assistant.
Your job is to give clear, accurate, general health education — never diagnose or treat.

Always follow these strict rules from your documentation:
{rules_context}

Here is reliable medical information:
{medical_info}

User question: {question}

Instructions:
- Answer in warm, easy-to-understand language.
- Keep it concise (2–4 short paragraphs max).
- If greeting only (hi, hello), reply warmly and invite a question.
- ALWAYS end with: "This is not medical advice; please consult a doctor."

Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

chain = (
    {
        "rules_context": lambda x: format_docs(retriever.invoke(x["question"])),
        "medical_info": lambda x: get_medical_info(x["question"]),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("Backend ready!")

# --- AGENT INTEGRATION ---
from agent import MedicalAgent

agent = MedicalAgent()
sessions = {}  # In-memory session store: user_id -> state_dict

# --- API ENDPOINTS ---

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Mount static files (assets like css, js, images) at /static
app.mount("/static", StaticFiles(directory="src"), name="static")

# Mount images directly at /images to match new index.html paths
app.mount("/images", StaticFiles(directory="src/images"), name="images")

@app.get("/")
def read_root():
    return FileResponse('src/index.html')

@app.post("/ask", response_model=BotResponse)
def ask_question(request: QuestionRequest):
    try:
        user_input = request.question.strip()
        user_id = request.user_id
        
        if not user_input:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        print(f"Received question: {user_input} from {user_id}")
        
        # Initialize session if needed
        if user_id not in sessions:
            sessions[user_id] = {"state": "NORMAL"}
            
        session = sessions[user_id]
        state = session.get("state", "NORMAL")
        
        response_text = ""
        
        # --- STATE MACHINE ---
        
        if state == "NORMAL":
            # 1. Try Agent Prediction
            prediction = agent.process_request(user_input)
            intent = prediction["intent"]
            medicine = prediction["medicine"]
            quantity = prediction["quantity"]
            confidence = prediction["confidence"]
            
            if medicine: # Only if a medicine was actually found
                # Ask for confirmation
                sessions[user_id]["state"] = "CONFIRMATION"
                sessions[user_id]["last_prompt"] = user_input
                sessions[user_id]["last_prediction"] = prediction
                
                response_text = (
                    f"I detected that you want to **{intent.replace('_', ' ')}**.\n"
                    f"Item: **{medicine}**, Quantity: **{quantity}**.\n\n"
                    f"Is this correct? (Type **OK** or **Wrong**)"
                )
                return BotResponse(answer=response_text)
            
            else:
                # Fallback to RAG if no medicine found (likely general question)
                response_text = chain.invoke({"question": user_input})
                return BotResponse(answer=response_text)

        elif state == "CONFIRMATION":
            clean_input = user_input.lower().strip()
            
            if clean_input in ["ok", "yes", "confirm", "sure"]:
                # Execute Action
                last_pred = session.get("last_prediction", {})
                intent = last_pred.get("intent", "unknown")
                medicine = last_pred.get("medicine", "unknown")
                quantity = last_pred.get("quantity", 1)
                
                # Normalize medicine name to match DB if possible (fuzzy match or direct map)
                price = 0.0
                matched_name = medicine
                
                # Simple fuzzy matching (substring check)
                for db_name, db_price in MEDICINE_PRICES.items():
                    if medicine.lower() in db_name.lower():
                        matched_name = db_name
                        price = db_price
                        break
                
                # Determine action type based on intent
                action_type = "add_to_cart"
                response_msg = f"Great! I've added {quantity}x {matched_name} to your cart."
                
                if "buy" in intent.lower():
                    action_type = "buy_now"
                    response_msg = f"Great! Proceeding to checkout with {quantity}x {matched_name}."
                
                action_payload = {
                    "type": action_type,
                    "medicine": matched_name,
                    "price": price,
                    "quantity": quantity
                }
                
                # Reset
                sessions[user_id]["state"] = "NORMAL"
                return BotResponse(answer=response_msg, action=action_payload)
                
            elif clean_input in ["wrong", "no", "incorrect"]:
                # Ask for correction
                sessions[user_id]["state"] = "CORRECTION"
                response_text = (
                    "I apologize. Please enter the correct action and prompt in this format:\n"
                    "**[Action] [Prompt]**\n"
                    "Example: `buy_medicine I want 2 panadol`"
                )
                return BotResponse(answer=response_text)
            
            else:
                return BotResponse(answer="Please type **OK** or **Wrong**.")

        elif state == "CORRECTION":
            # Parse correction
            # Expected: "action prompt"
            parts = user_input.split(' ', 1)
            if len(parts) == 2:
                action, prompt = parts
                action = action.lower()
                
                if action in ["buy_medicine", "add_to_cart"]:
                    # Save dataset
                    agent.save_new_data(prompt, action)
                    response_text = "Thank you. I have saved this correction for future training."
                else:
                    response_text = "Invalid action. Please start with `buy_medicine` or `add_to_cart`."
            else:
                response_text = "Invalid format. Please use **[Action] [Prompt]**."
            
            sessions[user_id]["state"] = "NORMAL"
            return BotResponse(answer=response_text)

        # Default fallback (shouldn't reach here easily)
        return BotResponse(answer="I am confused. resetting.", source="Internal")

    except Exception as e:
        print(f"Error processing request: {e}")
        # Reset session on error to avoid stuck state
        if 'user_id' in locals() and user_id in sessions:
             sessions[user_id]["state"] = "NORMAL"
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
