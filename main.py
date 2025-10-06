import os
from qdrant_helper import setup_qdrant, generate_json_from_docs

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEMO_COLLECTION = "demo_index"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

if __name__ == "__main__":
    # Step 1: Generate embeddings JSON from a real doc
    input_file = "my_doc.txt"
    if os.path.exists(input_file):
        generate_json_from_docs(input_file)
    else:
        print(f"Place your document at {input_file} to auto-generate embeddings.")

    # Step 2: Upload to Qdrant
    setup_qdrant(QDRANT_HOST, QDRANT_PORT, DEMO_COLLECTION)
