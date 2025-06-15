import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# === CONFIGURATION ===

# Chroma
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "Constitutia_2024_ro_chunks_e5"

# Embeddings
EMBEDDING_MODEL = "intfloat/e5-large-v2"

# LLaMA model path (RoMistral GGUF file)
LLAMA_MODEL_PATH = r"C:\llama\models\RoMistral-7b-Instruct.Q4_K_S.gguf"

# Inference params
N_RESULTS = 5
MAX_TOKENS = 1024
N_CTX = 2048
N_THREADS = 6
N_GPU_LAYERS = 30  # Adjust for your GPU


# === INIT MODELS ===

print("🧠 Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("📚 Connecting to Chroma...")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(name=COLLECTION_NAME)

print("🦙 Loading RoMistral (LLaMA)...")
llm = Llama(
    model_path=LLAMA_MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS
)

# === GET USER QUESTION ===

question = input("❓ Introdu o întrebare legală în limba română:\n> ").strip()
if not question:
    print("⚠️ Întrebarea este goală.")
    exit()

# Embed the query using E5 format
formatted_query = f"query: {question}"
query_embedding = embed_model.encode(formatted_query)

# === CHROMA SEARCH ===

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=N_RESULTS
)

chunks = results.get("documents", [[]])[0]

if not chunks:
    print("⚠️ Nu s-au găsit articole relevante.")
    exit()

# === FORMAT PROMPT FOR LLaMA ===

prompt = f"### Întrebare:\n{question}\n\n### Context:\n"
for i, chunk in enumerate(chunks):
    prompt += f"{i+1}. {chunk.strip()}\n"

prompt += "\n### Răspuns:\n"

# === RUN LLaMA GENERATION ===

print("\n🧠 Generare răspuns... (acesta poate dura câteva secunde)\n")
response = llm(prompt, max_tokens=MAX_TOKENS, stop=["###"])

# === OUTPUT ===

print("📌 Răspuns generat:\n")
print(response["choices"][0]["text"].strip())
