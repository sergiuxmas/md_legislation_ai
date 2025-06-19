import os
import sys
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# === CONFIGURATION ===

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
input_dir = os.path.join(project_root, "output")

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "Constitutia_2024_ro_chunks_e5_multilingual"

# === INITIALIZATION ===

print(f"🧠 Loading embedding model: {EMBEDDING_MODEL_ID}")
model = SentenceTransformer(EMBEDDING_MODEL_ID)

print("🔗 Connecting to Chroma...")
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# === LOAD CHUNKS FILE ===

if len(sys.argv) < 2:
    print("❌ Usage: python embed_chunks.py <chunks_filename.json>")
    sys.exit(1)

filename = sys.argv[1]
json_path = os.path.join(input_dir, filename)

if not os.path.isfile(json_path):
    print(f"❌ File not found: {json_path}")
    sys.exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"📄 Loaded {len(chunks)} chunks from {filename}")

# === CREATE OR REUSE COLLECTION ===

existing_collections = [col.name for col in chroma_client.list_collections()]
if COLLECTION_NAME in existing_collections:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"📚 Reusing collection: {COLLECTION_NAME}")
else:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    print(f"🆕 Created collection: {COLLECTION_NAME}")

# === FILTER ALREADY ADDED CHUNKS ===

existing_ids = set()
try:
    result = collection.get(include=[])
    existing_ids = set(result["ids"])
except:
    pass

new_chunks = []
new_ids = []
e5_encoded_chunks = []

for i, chunk in enumerate(chunks):
    chunk_id = f"chunk_{i}"
    if chunk_id not in existing_ids:
        new_chunks.append(chunk)
        new_ids.append(chunk_id)
        e5_encoded_chunks.append(f"passage: {chunk.strip()}")

if not new_chunks:
    print("⚠️ All chunks are already embedded.")
    sys.exit(0)

print(f"🔄 Generating embeddings for {len(new_chunks)} new chunks...")
embeddings = model.encode(e5_encoded_chunks, convert_to_numpy=True, show_progress_bar=True)

collection.add(
    documents=new_chunks,
    embeddings=embeddings.tolist(),
    ids=new_ids
)

print(f"✅ Added {len(new_chunks)} chunks to collection '{COLLECTION_NAME}'")
