# utils.py

import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

FAISS_PATH = "vector_store"  # folder to save FAISS index


# ----------------------------
# 1. Fetch MedlinePlus API
# ----------------------------
def fetch_dataset():
    terms = ["diabetes", "Cancer", "Fever", "Influenza", "Depression", "Anemia", "Obesity"]
    url = "https://wsearch.nlm.nih.gov/ws/query"
    results = {}

    for term in terms:
        params = {"db": "healthTopics", "term": term}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results[term] = response.text  # XML response
        else:
            results[term] = "Error in fetching data"

    return results


# ----------------------------
# 2. Parse XML to text
# ----------------------------
def parsing(xml_text: str):
    soup = BeautifulSoup(xml_text, "xml")
    docs = []
    for doc in soup.find_all("document"):
        title = doc.find("content", {"name": "title"})
        snippet = doc.find("content", {"name": "snippet"})
        title_text = title.text if title else ""
        snippet_text = snippet.text if snippet else ""
        docs.append(f"{title_text}\n{snippet_text}")
    return docs


# ----------------------------
# 3. Chunking
# ----------------------------
def chunking(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.create_documents(docs)
    return chunks


# ----------------------------
# 4. Build or Load Vector Store
# ----------------------------
def build_or_load_vector_store(results):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(FAISS_PATH, embed, allow_dangerous_deserialization=True)

    all_docs = []
    for term, xml_text in results.items():
        if not xml_text.startswith("Error"):
            parsed = parsing(xml_text)
            all_docs.extend(parsed)

    if not all_docs:
        print("No documents parsed. Aborting.")
        return None

    chunks = chunking(all_docs)
    vs = FAISS.from_documents(chunks, embedding=embed)
    vs.save_local(FAISS_PATH)
    return vs


# ----------------------------
# 5. LLM (Ollama)
# ----------------------------
def get_llm(temperature: float = 0.2):
    return Ollama(model="llama2", temperature=temperature)


# ----------------------------
# 6. Retrieval Chain
# ----------------------------
def get_retrieval_chain(vector_store, temperature: float):
    prompt_template = """
You are a helpful and knowledgeable **Medical AI Assistant**.
You provide clear, factual, and reliable medical information based on the provided context.


Always encourage users to consult a licensed healthcare provider for personal concerns.

Guidelines:
- Use only the provided context to answer.
- If possible, include common symptoms, causes, and treatments.
- If the context is insufficient, say so and provide safe, general information.
- Provide answers in clear, simple, and concise language.
- When appropriate, organize answers in bullet points (symptoms, causes, treatments).
- Avoid speculation or personal medical advice.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    llm = get_llm(temperature=temperature)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 8}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain
