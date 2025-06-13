import os
import sys
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Step 1: Load collection
collection_name = "Constitutia_2024_ro_chunks"

chroma_client = chromadb.Client(Settings(
    persist_directory="../output/chroma_db"
))

collection = chroma_client.get_collection(name=collection_name)

# Step 2: Load same embedding model used for indexing
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Step 3: Accept a question from the user (or hardcode for testing)
if len(sys.argv) >= 2:
    query = " ".join(sys.argv[1:])
else:
    query = input("🔎 Introdu o întrebare legală în limba română:\n> ")

# Step 4: Embed the query
query_embedding = model.encode(query)

# Step 5: Search
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

# Step 6: Show results
print("\n📄 Cele mai relevante articole găsite:")
for i, doc in enumerate(results['documents'][0]):
    print(f"\n📌 Rezultatul {i + 1}:\n{doc}")
