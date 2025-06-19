import os
import sys
import json
from sentence_transformers import SentenceTransformer
import chromadb

# === CONFIGURATION ===

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
input_dir = os.path.join(project_root, "output")

model_id = "BAAI/bge-m3"

# === ASK FOR CHUNK INPUT FILE ===

filename = input("Enter the chunks JSON file name (e.g. constitutia_rm_chunks.json): ").strip()
json_path = os.path.join(input_dir, filename)

if not os.path.isfile(json_path):
    print(f"❌ File not found: {json_path}")
    sys.exit(1)

# === EXTRACT FILE NAME AND SET COLLECTION NAME ===

base_name = os.path.splitext(os.path.basename(filename))[0]
collection_name = f"{base_name}_m3"

# === CONNECT TO CHROMA SERVER ===

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
print(f"🌐 Connected to Chroma Server at http://localhost:8000")

# === LOAD CHUNKS ===

with open(json_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"📄 Loaded {len(chunks)} chunks from {filename}")

# === LOAD EMBEDDING MODEL ===

print(f"🧠 Loading embedding model: {model_id} ...")
model = SentenceTransformer(model_id)

# === GET OR CREATE COLLECTION ===

collection_names = [col.name for col in chroma_client.list_collections()]
if collection_name in collection_names:
    collection = chroma_client.get_collection(name=collection_name)
    print(f"📚 Reusing existing collection: {collection_name}")
else:
    collection = chroma_client.create_collection(name=collection_name)
    print(f"🆕 Created new collection: {collection_name}")

# === CHECK EXISTING IDS ===

existing_ids = set()
try:
    result = collection.get(include=[])
    existing_ids = set(result["ids"])
except:
    pass

# === PREPARE NEW CHUNKS ===

new_docs = []
new_ids = []
passage_inputs = []

for i, chunk in enumerate(chunks):
    chunk_id = f"chunk_{i}"
    if chunk_id not in existing_ids:
        new_docs.append(chunk)
        new_ids.append(chunk_id)
        passage_inputs.append(f"passage: {chunk}")

if not new_docs:
    print("⚠️ No new chunks to add. Collection is already up to date.")
    sys.exit(0)

print(f"🔄 Generating embeddings for {len(new_docs)} new chunks...")
embeddings = model.encode(passage_inputs, convert_to_numpy=True, show_progress_bar=True)

collection.add(
    documents=new_docs,
    embeddings=embeddings.tolist(),
    ids=new_ids
)

print(f"✅ Added {len(new_docs)} new chunks to collection '{collection_name}' using model '{model_id}'")
