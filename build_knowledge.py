# build_knowledge.py
import os
import json
from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

VECTOR_DIR = Path("vector_store")
VECTOR_DIR.mkdir(exist_ok=True)

PDF_PATH = Path("al_ict.pdf")
CHUNK_SIZE = 500

reader = PdfReader(str(PDF_PATH))
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
print(f"Total chunks: {len(chunks)}")

sbert = SentenceTransformer("all-mpnet-base-v2")
embeddings = sbert.encode(chunks, convert_to_numpy=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, str(VECTOR_DIR / "faiss.index"))

with open(VECTOR_DIR / "chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("FAISS index and chunks saved to vector_store/")
