# bot.py
import os
import sqlite3
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from ollama import Client
import re

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
VECTOR_DIR = Path("vector_store")
DB_PATH = Path("cache.db")

# --- Load FAISS + chunks ---
print("Loading FAISS index and chunks...")
index = faiss.read_index(str(VECTOR_DIR / "faiss.index"))
with open(VECTOR_DIR / "chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# embedding model for similarity
sbert = SentenceTransformer("all-mpnet-base-v2")

# SQLite cache
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS cache (
    qnorm TEXT PRIMARY KEY,
    question TEXT,
    answer TEXT,
    created_at INTEGER
)
""")
conn.commit()

# Ollama client
ollama = Client()  # connects to local Ollama daemon

# --- Sinhala-aware tokenizer ---
SINDHI_RE = re.compile(r'[\u0D80-\u0DFF]+|[A-Za-z0-9]+|[^\s\w]', flags=re.UNICODE)

def sinhala_tokenize(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return SINDHI_RE.findall(text)

def normalize_question(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\u0D80-\u0DFF\s]', '', text)
    return text

# --- Retrieval function ---
def retrieve_top_k(question, k=4, embed_model=sbert):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

# --- RAG answer generation ---
def generate_answer(question, top_chunks, short=True):
    context_text = "\n\n---\n\n".join(top_chunks)
    answer_style = "Give a short (1-2 lines) student-friendly answer in Sinhala and/or English. Use Sinhala when short phrase exists, otherwise use English. Keep it simple."
    if not short:
        answer_style = "Provide a clear explanation in Sinhala and English. Use examples if needed."
    prompt = f"""
You are ICT Guru, an assistant for Sri Lankan A/L ICT students. Use the provided CONTEXT to answer the user's question. If the context is limited, answer based on the context and say 'From syllabus' or 'Based on syllabus'. Keep answers short (unless asked for detailed).

CONTEXT:
{context_text}

QUESTION:
{question}

INSTRUCTIONS:
{answer_style}

Answer:
"""
    resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not text:
        text = "\n".join([c[:400] for c in top_chunks])
    return text.strip()

# --- Cache functions ---
def check_cache(qnorm):
    cur.execute("SELECT answer FROM cache WHERE qnorm = ?", (qnorm,))
    r = cur.fetchone()
    return r[0] if r else None

def write_cache(qnorm, question, answer):
    ts = int(time.time())
    cur.execute("REPLACE INTO cache (qnorm, question, answer, created_at) VALUES (?, ?, ?, ?)",
                (qnorm, question, answer, ts))
    conn.commit()

# --- Telegram handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hi! I'm ICT Guru — your offline A/L ICT Helper. Ask me any ICT question in Sinhala or English.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ask about ICT topics (Data, Hardware, Networks, Programming, etc.). Try: 'දත්ත කියන්නේ මොකක්ද?' or 'What is CPU?'")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_q = update.message.text
    qnorm = normalize_question(user_q)

    cached = check_cache(qnorm)
    if cached:
        await update.message.reply_text(f"🔁 Cached answer:\n{cached}")
        return

    top = retrieve_top_k(user_q, k=4)
    if not top:
        await update.message.reply_text("Sorry, I couldn't find relevant syllabus text. Try rephrasing.")
        return

    try:
        answer = generate_answer(user_q, top, short=True)
    except Exception as e:
        answer = top[0][:800]
        print("LLM error:", e)

    try:
        write_cache(qnorm, user_q, answer)
    except Exception as e:
        print("Cache write error:", e)

    await update.message.reply_text(answer)

def main():
    if TELEGRAM_BOT_TOKEN is None:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in .env")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Starting ICT Guru bot...")
    app.run_polling()

if __name__ == "__main__":
    main()
