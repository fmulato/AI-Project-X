import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Carrega variáveis ambiente
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = Flask(__name__)
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Vetores e textos
index = None
text_chunks = []

# Carrega o modelo de embeddings localmente (carregar só uma vez)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def split_text(text, chunk_size=500, overlap=50):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []

    for para in paragraphs:
        if len(" ".join(current_chunk + [para]).split()) < chunk_size:
            current_chunk.append(para)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def embed_texts(chunks):
    filtered_chunks = [c.strip() for c in chunks if isinstance(c, str) and c.strip()]
    if not filtered_chunks:
        raise ValueError("Lista de chunks vazia ou inválida para embedding.")
    
    print(f"Gerando embeddings localmente para {len(filtered_chunks)} chunks")
    embeddings = embedder.encode(filtered_chunks, convert_to_numpy=True)
    return embeddings.astype("float32")

def load_pdf_text(filepath):
    doc = fitz.open(filepath)
    return "\n".join([page.get_text() for page in doc])

def prepare_index(text):
    global index, text_chunks
    text_chunks = split_text(text)
    embeddings = embed_texts(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

@app.route("/", methods=["GET", "POST"])
def index_page():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():

    file = request.files['pdf']
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])

    global index, text_chunks
    new_chunks = split_text(text)
    new_embeddings = embed_texts(new_chunks)

    if index is None:
        index = faiss.IndexFlatL2(new_embeddings.shape[1])
        index.add(new_embeddings)
    else:
        index.add(new_embeddings)

    text_chunks.extend(new_chunks)

    return jsonify({"message": f"Arquivo '{file.filename}' carregado e vetorizado com sucesso."})

@app.route("/ask", methods=["POST"])
def ask_question():
    query = request.json.get("question")
    language = request.json.get("language", "pt")

    print(f"Idioma selecionado: {language}")

    query_vec = embed_texts([query])
    D, I = index.search(query_vec, k=3)
    context = "\n".join([text_chunks[i] for i in I[0]])

    if language == "en":
        prompt = f"""
You are a recruiter specialized in IT jobs. Based solely on the following content, answer the question in English.  
If the answer is not found in the content, say you do not have enough information.

Content:
{context}

Question: {query}
Answer in English:
"""
    else:
        prompt = f"""
Você é um recrutador especializado em vagas na área de Tecnologia da Informação. Baseie sua resposta apenas no conteúdo abaixo.  
Se a resposta não estiver no conteúdo, diga que não há informação suficiente.

Conteúdo:
{context}

Pergunta: {query}
Resposta:
"""

    #print(f"Prompt gerado: {prompt}")

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [
                {"role": "system", "content": "Always reply ONLY in English." if language == "en" else "Sempre responda SOMENTE em português."},
                {"role": "user", "content": prompt.strip()}
                            ]
            }

        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"[Erro ao consultar modelo do OpenRouter: {e}]"

    return jsonify({"answer": answer.strip()})

if __name__ == "__main__":
    app.run(debug=True)
