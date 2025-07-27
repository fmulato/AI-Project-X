# 🧠 CV Analyzer AI

A lightweight web app that allows users to **upload a PDF résumé (CV)** and ask context-aware questions in **English or Portuguese** based on the document’s content.

Ideal for **recruiters, HR professionals, or personal use**, especially in the tech industry.

---

## ✨ Features

- 📄 Upload PDF CVs
- 🔍 Intelligent text chunking and vector search using `sentence-transformers` + FAISS
- 🤖 Answer generation using free models via [OpenRouter.ai](https://openrouter.ai)
- 🌐 Multilingual support: choose between English or Portuguese
- ⚡ No paid OpenAI API required
- 🧪 Simple Flask interface (MVP)
- 🗂️ Files temporarily stored in `uploaded_files/` (optional)

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
