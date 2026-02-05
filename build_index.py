import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
BASE_DATA_DIR = "knowledge/3_index_ready"
INDEX_DIR = "index"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 50   # ochrana proti šumu

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LAYERS = ["raw", "synth", "meta"]

# ---------- SETUP ----------
os.makedirs(INDEX_DIR, exist_ok=True)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("▶ Using LOCAL embeddings:", EMBEDDING_MODEL_NAME)

# ---------- HELPERS ----------
def chunk_text(text: str, size: int, overlap: int):
    """
    Epistemicky bezpečné chunkování:
    - primárně po prázdných řádcích (respektuje ručně psané vrstvy)
    - fallback po znacích pro dlouhé odstavce
    - žádná interpretace obsahu
    """
    paragraphs = [
        p.strip() for p in text.split("\n\n")
        if len(p.strip()) >= MIN_CHUNK_LENGTH
    ]

    chunks = []

    for p in paragraphs:
        if len(p) <= size:
            chunks.append(p)
        else:
            i = 0
            while i < len(p):
                part = p[i:i + size].strip()
                if len(part) >= MIN_CHUNK_LENGTH:
                    chunks.append(part)
                i += size - overlap

    return chunks


def embed(texts):
    vectors = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.array(vectors, dtype="float32")


# ---------- LOAD DATA (STRICT LAYERS) ----------
documents = []

for layer in LAYERS:
    layer_dir = os.path.join(BASE_DATA_DIR, layer)

    if not os.path.isdir(layer_dir):
        raise RuntimeError(f"❌ Chybí vrstva: {layer_dir}")

    for file in os.listdir(layer_dir):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(layer_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

            if not text:
                print(f"⚠️  Prázdný soubor přeskočen: {layer}/{file}")
                continue

            documents.append({
                "source": f"{layer}/{file}",
                "text": text,
                "layer": layer
            })

if not documents:
    raise RuntimeError("❌ Nebyla nalezena žádná indexovatelná data.")

print(f"▶ Loaded {len(documents)} documents (layered)")


# ---------- CHUNK ----------
chunks = []

for doc in documents:
    parts = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)

    for part in parts:
        chunks.append({
            "text": part,
            "source": doc["source"],
            "layer": doc["layer"]
        })

if not chunks:
    raise RuntimeError("❌ Po chunkování nevznikl žádný validní chunk.")

print(f"▶ Created {len(chunks)} text chunks")


# ---------- EMBED ----------
vectors = embed([c["text"] for c in chunks])
dim = vectors.shape[1]

print(f"▶ Embedding dimension: {dim}")


# ---------- FAISS ----------
index = faiss.IndexFlatL2(dim)
index.add(vectors)

faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))


# ---------- SAVE CHUNKS ----------
with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ HOTOVO: {len(chunks)} chunků uloženo do FAISS (s vrstvami).")
