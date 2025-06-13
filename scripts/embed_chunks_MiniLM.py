import os
import sys
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# === CONFIGURATION ===
# ✅ Use full absolute paths for Windows
raw_project_root = r"C:\Users\screciun\PycharmProjects\am_internship-2024\md_legislation_ai_project"
project_root = os.path.abspath(raw_project_root)
db_path = os.path.join(project_root, "output", "chroma_db")
input_dir = os.path.join(project_root, "output")

# ✅ Ensure Chroma DB directory exists
os.makedirs(db_path, exist_ok=True)
print(f"📁 RESOLVED PATH TO DB: {db_path}")

# === INPUT VALIDATION ===
if len(sys.argv) < 2:
    print("❌ Please provide the chunks JSON file name (e.g. constitutia_rm_chunks.json)")
    sys.exit(1)

filename = sys.argv[1]
json_path = os.path.join(input_dir, filename)

if not os.path.isfile(json_path):
    print(f"❌ File not found: {json_path}")
    sys.exit(1)

base_name = os.path.splitext(filename)[0]  # collection name = base filename

# === LOAD CHUNKS ===
with open(json_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"📄 Loaded {len(chunks)} chunks from {filename}")

# === LOAD EMBEDDING MODEL ===
print("🧠 Loading embedding model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# === CONNECT TO CHROMA ===
print("🔌 Connecting to Chroma...")
chroma_client = chromadb.Client(Settings(
    persist_directory=db_path,
    anonymized_telemetry=False
))

# 🔧 Engine Check
engine_type = chroma_client.__class__.__name__
print("🔧 Chroma client engine type:", engine_type)
if "Ephemeral" in engine_type:
    print("❌ Chroma is running in-memory. Persistence is NOT working.")
    print("👉 Try: pip install duckdb chromadb --upgrade")
    sys.exit(1)

# === GET OR CREATE COLLECTION ===
collection_names = [col.name for col in chroma_client.list_collections()]
if base_name in collection_names:
    collection = chroma_client.get_collection(name=base_name)
    print(f"📚 Reusing existing collection: {base_name}")
else:
    collection = chroma_client.create_collection(name=base_name)
    print(f"🆕 Created new collection: {base_name}")

# === CHECK EXISTING CHUNK IDs ===
existing_ids = set()
try:
    result = collection.get(include=[])
    existing_ids = set(result["ids"])
except Exception as e:
    print("⚠️ Could not fetch existing IDs:", str(e))

# === FILTER AND EMBED NEW CHUNKS ===
new_docs = []
new_ids = []

for i, chunk in enumerate(chunks):
    chunk_id = f"chunk_{i}"
    if chunk_id not in existing_ids:
        new_docs.append(chunk)
        new_ids.append(chunk_id)

if not new_docs:
    print("⚠️ No new chunks to add. Collection is up to date.")
    sys.exit(0)

print(f"🔄 Generating embeddings for {len(new_docs)} new chunks...")
embeddings = model.encode(new_docs, convert_to_numpy=True, show_progress_bar=True)

# === ADD TO COLLECTION ===
collection.add(
    documents=new_docs,
    embeddings=embeddings.tolist(),
    ids=new_ids
)

print(f"✅ Added {len(new_docs)} new chunks to collection '{base_name}'")
print(f"📌 To verify, run: python list_collections.py")

