import chromadb

# === Connect to Chroma Server (running via Docker) ===
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
print("🌐 Connected to Chroma Server at http://localhost:8000")

# === List available collections ===
collections = chroma_client.list_collections()

if not collections:
    print("⚠️ No collections found on Chroma Server.")
else:
    print("📚 Available collections:")
    for col in collections:
        print(f" - {col.name}")
