# 🇱🇰 ICT Guru — Sri Lankan A/L ICT Helper Bot

Offline Telegram bot that answers A/L ICT questions in Sinhala + English using your syllabus PDF, Ollama LLM, FAISS RAG, and SQLite cache.

---

## 🧠 Features
- Uses your A/L ICT syllabus as knowledge base
- Runs fully offline on DigitalOcean VPS
- Sinhala + English question understanding
- RAG summarization & cached responses
- No API keys, no external cloud

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/al_ict_bot.git
cd al_ict_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
