import chromadb
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===

collection_name = "Constitutia_2024_ro_chunks_e5"  # MUST MATCH the embed script
model_id = "intfloat/e5-large-v2"

# === CONNECT TO CHROMA SERVER ===

client = chromadb.HttpClient(host="localhost", port=8000)
print(f"🌐 Connected to Chroma Server - collection: {collection_name}")

# === LOAD EMBEDDING MODEL ===

print(f"🧠 Loading embedding model: {model_id} ...")
model = SentenceTransformer(model_id)

# === GET COLLECTION ===

collection = client.get_collection(name=collection_name)

# === GET USER QUERY ===

query = input("❓ Introdu o întrebare în limba română:\n> ").strip()
if not query:
    print("⚠️ Query was empty.")
    exit()

# Format query for E5 model
formatted_query = f"query: {query}"
query_embedding = model.encode(formatted_query)

# === SEARCH CHROMA ===

try:
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
except Exception as e:
    print(f"❌ Query failed: {str(e)}")
    exit()

# === DISPLAY RESULTS ===

docs = results.get("documents", [[]])[0]
if not docs:
    print("⚠️ No results found.")
else:
    print("\n📄 Cele mai relevante articole găsite:")
    for i, doc in enumerate(docs):
        print(f"\n🔹 Rezultatul {i + 1}:\n{doc}")
