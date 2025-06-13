import chromadb

# Connect to Chroma Server (running in Docker)
client = chromadb.HttpClient(host="localhost", port=8000)

# List collections first
collections = client.list_collections()
if not collections:
    print("⚠️ No collections found.")
    exit()

print("📚 Available collections:")
for col in collections:
    print(f" - {col.name}")

# Ask user for collection name to delete
collection_name = input("❓ Enter the exact name of the collection to delete:\n> ")

# Confirm deletion
confirm = input(f"⚠️ Are you sure you want to delete '{collection_name}'? (yes/no): ")
if confirm.lower() != "yes":
    print("❌ Deletion canceled.")
    exit()

# Delete the collection
try:
    client.delete_collection(name=collection_name)
    print(f"✅ Collection '{collection_name}' has been deleted.")
except Exception as e:
    print(f"❌ Failed to delete collection: {str(e)}")
