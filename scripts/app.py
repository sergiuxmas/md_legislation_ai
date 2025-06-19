import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import numpy as np
import gradio as gr
import time
from typing import Tuple, List, Dict, Any

# === CONFIGURATION ===

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "Constitutia_2024_ro_chunks_e5_multilingual"

LLAMA_MODELS = {
    "RoMistral-7B": r"C:\llama\models\RoMistral-7B-Instruct.Q4_K_S.gguf",
    "RoLLaMA3.1-8B": r"C:\llama\models\RoLlama3.1-8b-Instruct.Q4_K_M.gguf"
}

TOP_K_INITIAL = 20
TOP_K_FINAL = 6
MAX_TOKENS = 2048
N_CTX = 4096
N_THREADS = 6
N_GPU_LAYERS = 40

# === INITIALIZATION ===

print("🧠 Loading embedding model:", EMBEDDING_MODEL_NAME)
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("🔗 Connecting to Chroma...")
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# === Load models on demand and cache them ===
loaded_llms: Dict[str, Llama] = {}


def get_llm(model_name: str) -> Llama:
    if model_name not in loaded_llms:
        print(f"🦙 Loading model: {model_name}")
        loaded_llms[model_name] = Llama(
            model_path=LLAMA_MODELS[model_name],
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS
        )
    return loaded_llms[model_name]


# === COMBINED PIPELINE ===

def full_pipeline(question: str, model_name: str) -> Tuple[str, str, str, str, str]:
    if not question.strip():
        return "", "⚠️ Întrebarea este goală.", "", "", ""

    t0_search = time.time()

    formatted_query = f"query: {question}"
    query_emb = embed_model.encode([formatted_query])
    results = collection.query(query_embeddings=query_emb.tolist(), n_results=TOP_K_INITIAL)

    docs = results.get("documents", [[]])[0]
    doc_embs = embed_model.encode([f"passage: {doc}" for doc in docs])
    sims = cosine_similarity(query_emb, doc_embs)[0]
    top_indices = np.argsort(sims)[::-1][:TOP_K_FINAL]
    top_chunks = [docs[i] for i in top_indices]

    context = "\n".join([f"{i + 1}. {chunk.strip()}" for i, chunk in enumerate(top_chunks)])
    prompt = f"""### Întrebare:
{question}

### Context:
{context}

### Răspuns:
"""

    search_time = f"{(time.time() - t0_search):.2f} secunde"

    # === LLaMA Inference ===
    t0_llm = time.time()
    llm = get_llm(model_name)
    output = llm(prompt, max_tokens=MAX_TOKENS, stop=["###"])
    answer = output["choices"][0]["text"].strip()
    llm_time = f"{(time.time() - t0_llm):.2f} secunde"

    return context, answer, prompt, search_time, llm_time


# === GRADIO UI ===

with gr.Blocks() as demo:
    gr.Markdown("## 🧑‍⚖️ Asistent Juridic RAG – Constituția RM 🇲🇩")

    qbox = gr.Textbox(label="Întrebare legală (română)")
    model_selector = gr.Dropdown(
        choices=list(LLAMA_MODELS.keys()),
        value="RoMistral-7B",
        label="Alege modelul LLaMA"
    )
    submit = gr.Button("🔍 Caută și răspunde")
    answer_box = gr.Textbox(label="🧠 Răspuns generat", lines=3)
    llm_time_box = gr.Textbox(label="⏱️ Timp generare", interactive=False)
    prompt_box = gr.Textbox(label="📄 Prompt LLaMA", visible=False)
    context_box = gr.Textbox(label="📚 Context extras", lines=6)
    search_time_box = gr.Textbox(label="⏱️ Timp căutare", interactive=False)

    submit.click(
        fn=full_pipeline,
        inputs=[qbox, model_selector],
        outputs=[context_box, answer_box, prompt_box, search_time_box, llm_time_box]
    )

demo.launch()
