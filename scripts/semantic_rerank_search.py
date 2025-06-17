from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import numpy as np

# === CONFIGURATION ===
# embedding_model_id = "intfloat/e5-large-v2"
embedding_model_id = "BAAI/bge-m3"
collection_name = "Constitutia_2024_ro_chunks_e5"
chroma_host = "localhost"
chroma_port = 8000
top_k = 20   # number of initial Chroma results
final_k = 5  # top N after reranking

# === INIT MODELS ===
print("🧠 Loading embedding model...")
embed_model = SentenceTransformer(embedding_model_id)

print("🔗 Connecting to Chroma Server...")
client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
collection = client.get_collection(name=collection_name)

# === GET USER QUERY ===
user_question = input("❓ Introdu o întrebare legală în română:\n> ").strip()
if not user_question:
    print("⚠️ Întrebarea este goală.")
    exit()

# Format query for E5
query = f"query: {user_question}"
query_emb = embed_model.encode([query])

# === INITIAL CHROMA SEARCH ===
results = collection.query(
    query_embeddings=query_emb.tolist(),
    n_results=top_k
)
retrieved_docs = results["documents"][0]

# === EMBED CHUNKS FOR RERANKING ===
doc_embeddings = embed_model.encode([f"passage: {doc}" for doc in retrieved_docs])

# === RERANK BASED ON COSINE SIMILARITY ===
similarities = cosine_similarity(query_emb, doc_embeddings)[0]
top_indices = np.argsort(similarities)[::-1][:final_k]
top_chunks = [retrieved_docs[i] for i in top_indices]

# === DISPLAY FINAL RESULTS ===
print(f"\n📚 Top {final_k} articole relevante după rerank:\n")
for i, chunk in enumerate(top_chunks, 1):
    print(f"🔹 {i}. {chunk.strip()}\n")
