from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import chromadb
import numpy as np
import gradio as gr

# === CONFIGURATION ===
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "Constitutia_2024_ro_chunks_e5"
# EMBEDDING_MODEL = "intfloat/e5-large-v2"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLAMA_MODEL_PATH = r"C:\llama\models\RoMistral-7b-Instruct.Q4_K_S.gguf"

# Retrieval settings
TOP_K_INITIAL = 20
TOP_K_FINAL = 6

# LLaMA generation settings
MAX_TOKENS = 2048
N_CTX = 4096
N_THREADS = 6
N_GPU_LAYERS = 40

# === INITIALIZATION ===
print("🧠 Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("🔗 Connecting to Chroma...")
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

print("🦙 Loading LLaMA model (RoMistral)...")
llm = Llama(
    model_path=LLAMA_MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS
)

# === RAG + RERANK FUNCTION ===
def answer_question(question):
    if not question.strip():
        return "⚠️ Te rog introdu o întrebare."

    # Step 1: Embed the query
    query_embedding = embed_model.encode([question])

    # Step 2: Retrieve from Chroma
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K_INITIAL)
    retrieved_docs = results["documents"][0]

    if not retrieved_docs:
        return "⚠️ Nu s-au găsit articole relevante."

    # Step 3: Rerank
    doc_embeddings = embed_model.encode(retrieved_docs)
    sims = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:TOP_K_FINAL]
    top_chunks = [retrieved_docs[i] for i in top_indices]

    # Step 3: Format prompt
    context = "\n".join([f"{i + 1}. {chunk.strip()}" for i, chunk in enumerate(top_chunks)])
    prompt = f"""### Întrebare:
    {question}

    ### Context:
    {context}

    ### Răspuns:
    """

    # Step 5: Generate
    output = llm(prompt, max_tokens=MAX_TOKENS, stop=["###"])
    answer = output["choices"][0]["text"].strip()

    return answer, context, prompt

# === GRADIO UI ===
with gr.Blocks() as demo:
    gr.Markdown("# 🧑‍⚖️ Asistent Juridic RAG – Constituția RM 🇲🇩")
    qbox = gr.Textbox(label="Întrebare legală (română)")
    submit = gr.Button("🔍 Caută și generează răspuns")
    abox = gr.Textbox(label="🧠 Răspuns generat", lines=6)
    cbox = gr.Textbox(label="📚 Context din Constituție (reranked)", lines=6)
    pbox = gr.Textbox(label="📄 Prompt LLaMA", visible=False)

    submit.click(fn=answer_question, inputs=qbox, outputs=[abox, cbox, pbox])

demo.launch()
