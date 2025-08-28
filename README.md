# Medical ChatBot

A web-based **Medical ChatBot** powered by **FastAPI**, **LangChain**, and **Ollama LLM**, providing accurate and up-to-date medical information. The chatbot retrieves information from **MedlinePlus** datasets and answers user queries in clear and concise language.

---

## Features

- Ask questions about medical conditions such as diabetes, cancer, fever, influenza, depression, anemia, and obesity.
- Retrieval-based responses using FAISS vector store and embeddings.
- Clean, responsive frontend with a chat interface.
- Real-time chatbot interaction with loading feedback.
- Sections for **About Us** and **Services**.
- Encourages users to consult licensed healthcare providers for personal concerns.

---

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **LLM**: Ollama (`llama2` model)
- **Vector Store**: FAISS
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Data Source**: MedlinePlus API
- **Python Libraries**: 
  - `requests`
  - `BeautifulSoup4`
  - `langchain`
  - `fastapi`
  - `pydantic`

---

## Project Structure

 |── src/
│ ├── index.html # Frontend HTML
│ ├── style.css # Frontend CSS
│ └── script.js # Frontend JS
├── utils.py # Utility functions for data fetching, parsing, vector store, and LLM
├── app.py # FastAPI backend
└── vector_store/ # FAISS vector store saved locally



---

## Setup Instructions

1. **Clone the repository**

2. **pip install -r requirements.txt**

3. **uvicorn main:app --reload**

---
I can also generate a ready-to-use **`requirements.txt`** for this project if you want.  

Do you want me to create that as well?




