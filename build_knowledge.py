# build_knowledge.py
import os, json, re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, fitz
from tqdm import tqdm

PDF_PATH = Path("al_ict.pdf")
OUT_DIR = Path("vector_store")
OUT_DIR.mkdir(exist_ok=True)
EMB_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE, OVERLAP = 900, 128

def extract_text(path):
    doc = fitz.open(path)
    return "\n".join([p.get_text("text") for p in doc])

def chunk(text):
    chunks, start = [], 0
    text = re.sub(r'\n{2,}', '\n', text)
    while start < len(text):
        end = start + CHUNK_SIZE
        if end > len(text): end = len(text)
        cut = text.rfind(". ", start, end)
        if cut == -1: cut = end
        chunks.append(text[start:cut].strip())
        start = cut - OVERLAP
    return [c for c in chunks if len(c) > 40]

def build_index(chunks):
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)
    faiss.write_index(idx, str(OUT_DIR/"faiss.index"))
    json.dump(chunks, open(OUT_DIR/"chunks.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("✅ Index built with", len(chunks), "chunks.")

if __name__ == "__main__":
    txt = extract_text(PDF_PATH)
    ch = chunk(txt)
    build_index(ch)
