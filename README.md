# md_legislation_ai

### 🖥️ Pasul 3a: Interfață Web (Gradio)

- ✅ Fisier: `app.py`
- ✅ Interfață Gradio care acceptă întrebări și returnează răspunsuri legale în limba română
- ✅ Rulat local la: http://localhost:7860

---

## 🚀 Cum rulezi proiectul

### 1. Pornește Chroma Server
```bash
docker-compose up -d
```

## Local environment
 - Legal area: Constitutia Republicii Moldova RO, EN
 - LLM model: RoMistral-7b-Instruct
 - Run: python, llama-cpp-python interface
 - Database: ChromaDB


## Production environment
 - LLM model: RoLlama3.1-8b-Instruct

## RAG Script logic
```bash
 1. Accept a question in Romanian
 2. Search Chroma for top 3 relevant legal chunks
 3. Format them as context
 4. Build a prompt like:
    ### Întrebare:
    [question]

    ### Context:
    1. [article 1]
    2. [article 2]
    3. [article 3]

    ### Răspuns:
```
---

## 🧩 Rezumatul etapelor proiectului

### ✅ Pasul 1: Prelucrarea Constituției și căutare semantică
- 📄 Încărcare fișier PDF: Constituția Republicii Moldova
- ✂️ Fragmentare în articole folosind LangChain
- 🔍 Generare embeddings cu `intfloat/e5-large-v2`
- 🗃️ Salvare vectori într-o colecție Chroma: `Constitutia_2024_ro_chunks_e5`
- ✅ Testare cu interogări în limba română folosind `test_search.py`

### ✅ Pasul 2: Integrarea LLM local (RAG complet)
- 🧠 Model ales: `RoMistral-7B-Instruct.Q4_K_M.gguf` (optimizat pentru română)
- ⚙️ Inference local cu `llama-cpp-python`
- 🔗 Integrare cu Chroma: întrebări → context legal → generare răspuns
- 🧪 Script `rag_local.py`: funcțional, testat, produce răspunsuri coerente

### ✅ Pasul 3a: Interfață web (UI)
- 🖥️ UI construit cu Gradio (`app.py`)
- 🎛️ Interfață prietenoasă pentru întrebări legale în română
- ✅ Local, offline, fără API extern
- ✅ Complet funcțional: întrebare → căutare → generare → afișare răspuns

### 🔜 Pasul 3b: Input vocal
- 🎤 Planificat: integrare Whisper sau Whisper.cpp
- Scop: utilizatorul poate pune întrebări prin microfon

### 🔜 Pasul 4: Extindere corpus legislativ
- 📚 Codul Penal, Codul Civil, Monitorul Oficial etc.
- Va fi vectorizat și integrat în colecții Chroma separate

### 🔜 Pasul 5: Migrare pe VM avansat
- ☁️ Mutarea sistemului pe o mașină virtuală performantă
- Scop: testare, demo public, scalabilitate

---
## Prompt example
```bash
A)  Întrebare
Contextul întrebărilor se referă la Republica Moldova.
Poate Președintele țării să-și depună demisia din propria inițiativă?
Răspunsul să conțină referința la articolul din legislație.

B) Cerințe față de răspuns:
1) Să includă referințele la legislație.
2) Verificare 
  - Să existe referințele la articolele din legislație.
  - Se afișează doar articolele relevante la întrebare.

C) Totalizare
 - La sfîrșit se adaogă o generalizare, luînd în considerare întrebarea adresată și informația găsită.
```

## Notes
 - Chroma locks the embedding dimension at collection creation — and you can't mix or overwrite them.
 - If is used multiple users/sessions
```bash
with gr.Blocks() as demo:
    model_state = gr.State(embed_model)
    llm_state = gr.State(llm)
    
    submit.click(fn=answer_question, inputs=[question, model_state, llm_state], ...)
#   //But for now — global initialization at the top is enough for single-user local use.
```
 - Integrate FAISS with GPU acceleration
```bash
| Feature              | FAISS (GPU)       |
| -------------------- | ----------------- |
| Vector search speed  | ⚡ **Much faster** |
| Million-scale search | ✅ Easily scalable |
| GPU support          | ✅ CUDA-enabled    |
| Offline-compatible   | ✅ 100% local      |
```
