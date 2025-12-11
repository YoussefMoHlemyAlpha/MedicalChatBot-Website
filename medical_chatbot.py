import os
import re
import requests
import xml.etree.ElementTree as ET
from Bio import Entrez
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# CONFIGURATION 
Entrez.email = "ahmed.agamy393@gmail.com"
DOC_PATH = "medical_rules.txt"
PERSIST_DIR = "./chroma_db"
MEDLINEPLUS_SEARCH = "https://wsearch.nlm.nih.gov/ws/query"

# MEDLINEPLUS FETCHER
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
        text = re.sub('\s+', ' ', text).strip()
        return text[:1200] + ("..." if len(text) > 1200 else "")
    except:
        return None
    
# MEDICAL INFO (MedlinePlus → PubMed fallback)
def get_medical_info(question):
    # 1. Try MedlinePlus first (best for patients)
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
    except:
        pass

    return "I couldn't find reliable, plain-language information on this topic right now."

# LOAD & INDEX RULES DOCUMENT
if not os.path.exists(PERSIST_DIR):
    print("First time setup: Indexing your rules document...")
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("Indexing complete! Knowledge base ready.")
else:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM & PROMPT
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

# CHAT LOOP
def chat():
    print("\n🤖 Medical Information Assistant is ready!")
    print("   General health info only • Not a doctor")
    print("   Type 'bye' or 'exit' to quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["bye", "exit", "quit", "goodbye"]:
                print("Chatbot: Take care and stay healthy! 👋")
                break
            if not user_input:
                continue

            print("Chatbot: Thinking...", end="\r")
            response = chain.invoke({"question": user_input})
            print(" " * 50, end="\r")  # clear "Thinking..."
            print(f"Chatbot: {response}")
            print("—" * 60)

        except KeyboardInterrupt:
            print("\n \nChatbot: Take care! 👋")
            break
        except Exception as e:
            print(f"Oops, something went wrong: {e}")

if __name__ == "__main__":
    chat()
