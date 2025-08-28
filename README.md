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
## Usage

-Type a medical question in the input field.

-Click Send or press Enter.

-Wait for the chatbot to retrieve information and display the answer with sources.

-Explore the About Us and Services sections for additional information.

## Notes

-The chatbot is retrieval-based, providing information only from MedlinePlus.

-Always consult a licensed healthcare provider for personal medical advice.

-The FAISS vector store will be built on first run and stored in the vector_store/ directory for faster queries next time.



