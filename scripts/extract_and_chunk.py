import fitz  # PyMuPDF
import os
import sys
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Check if a filename was provided
if len(sys.argv) < 2:
    print("❌ Please provide the PDF filename as an argument (e.g. constitutia_rm.pdf)")
    sys.exit(1)

filename = sys.argv[1]
pdf_path = os.path.join("..", "data", filename)

# Check if the file exists
if not os.path.isfile(pdf_path):
    print(f"❌ File not found: {pdf_path}")
    sys.exit(1)

# Extract base name (without extension) for naming output
base_name = os.path.splitext(filename)[0]

# Step 1: Extract text from PDF
doc = fitz.open(pdf_path)
full_text = ""
for page in doc:
    full_text += page.get_text()
doc.close()

# Step 2: Chunk text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(full_text)

# Step 3: Save to JSON
output_dir = os.path.join("..", "output")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, f"{base_name}_chunks.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Done. {len(chunks)} chunks saved to {output_path}")
