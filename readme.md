# Retrieval-Augmented Generation (RAG) System

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using open-source tools with local vector storage. It enhances LLM capabilities by retrieving relevant information from external documents before generating responses.

---

## 📌 Project Features

- ✅ Document loading from PDF, DOCX, and TXT files.
- ✅ Text chunking with overlap to prevent information loss.
- ✅ Embedding generation using Sentence Transformers.
- ✅ Local vector store using FAISS for efficient similarity search.
- ✅ Advanced retrieval strategy using Maximum Marginal Relevance (MMR).
- ✅ LLM integration using NGU API (and optional GROQ API).
- ✅ Prompt engineering for optimal LLM performance.

---

## 📁 Project Structure

RAG/
├── main.py
├── document_loader.py
├── splitter.py
├── embedding_store.py
├── llm_client.py
├── .env (add your API keys here)
├── requirements.txt
└── documents/ (place your test documents here)

---

## 🚀 Setup Instructions

1. Clone the repository:
```bash
git clone <your-repo-url>
cd RAG
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
Run the project:

bash
Copy
Edit
python main.py
📖 LLM Configurations Tested
NGU API: qwen2.5-coder:7b

Optional: GROQ API with llama-3.3-70b-versatile

📈 Evaluation & Experiments
Compared Basic Similarity Search vs. MMR Retrieval.

MMR Retrieval provided more diverse and relevant document chunks.

Final answers generated using NGU API showed high relevance to user queries.

🛠️ Challenges Faced
Resolved API integration issues by creating a flexible llm_client.py supporting multiple providers.

Managed FAISS index storage and ensured consistent metadata mapping.

Handled document splitting to avoid information loss at chunk boundaries.

Improved prompt engineering to enhance LLM response quality.

📌 Notes
For privacy, the .env file should not be committed with actual keys. A sample .env is provided.

Ensure the documents/ folder contains relevant documents for retrieval.
