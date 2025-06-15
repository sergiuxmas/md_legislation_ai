import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# === CONFIGURATION ===
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "Constitutia_2024_ro_chunks_e5"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
LLAMA_MODEL_PATH = r"C:\llama\models\RoMistral-7b-Instruct.Q4_K_S.gguf"

N_RESULTS = 3
MAX_TOKENS = 1024
N_CTX = 2048
N_THREADS = 6
N_GPU_LAYERS = 30

# === INITIALIZATION ===
print("🔧 Initializing models and Chroma connection...")

embed_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

llm = Llama(
    model_path=LLAMA_MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS
)

def generate_response(question):
    if not question.strip():
        return "⚠️ Te rog introdu o întrebare."

    # Step 1: Embed the query
    formatted_query = f"query: {question}"
    query_embedding = embed_model.encode(formatted_query)

    # Step 2: Retrieve from Chroma
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=N_RESULTS)
    chunks = results.get("documents", [[]])[0]

    if not chunks:
        return "⚠️ Nu s-au găsit articole relevante."

    # Step 3: Format prompt
    prompt = f"### Întrebare:\n{question}\n\n### Context:\n"
    for i, chunk in enumerate(chunks):
        prompt += f"{i+1}. {chunk.strip()}\n"
    prompt += "\n### Răspuns:\n"

    # Step 4: Generate response
    response = llm(prompt, max_tokens=MAX_TOKENS, stop=["###"])
    return response["choices"][0]["text"].strip()

# === GRADIO UI ===
demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=3, label="Întrebare juridică (în limba română)"),
    outputs=gr.Textbox(label="Răspuns generat de RoMistral"),
    title="Asistent Juridic - Constituția RM",
    description="Întreabă în limba română. Sistemul va căuta în Constituția Republicii Moldova și va genera un răspuns."
)

if __name__ == "__main__":
    demo.launch()
