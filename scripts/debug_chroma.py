import os
import chromadb
from chromadb.config import Settings

db_path = r"C:\Users\screciun\PycharmProjects\am_internship-2024\md_legislation_ai_project\output\chroma_db"
db_path = os.path.abspath(db_path)
print("📁 RESOLVED PATH TO DB:", db_path)

client = chromadb.Client(Settings(
    persist_directory=db_path,
    anonymized_telemetry=False
))

print("🔧 Chroma engine:", client.__class__.__name__)

collections = client.list_collections()
if not collections:
    print("⚠️ Still no collections found.")
else:
    print("📚 Found collections:")
    for c in collections:
        print(f" - {c.name}")
