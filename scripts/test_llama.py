from llama_cpp import Llama

# Path to your downloaded model
model_path = r"C:\llama\models\RoMistral-7b-Instruct.Q4_K_S.gguf"

# Load the model
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=30,  # Use 0 if CPU-only
    verbose=True
)

# Romanian legal prompt (Alpaca-style formatting)
prompt = """
### Întrebare:
Care este rolul Curții Constituționale în Republica Moldova?

### Răspuns:
"""

# Run the model
output = llm(prompt, max_tokens=256, stop=["###"])
print("\n🧠 Răspuns generat:\n", output["choices"][0]["text"])
