import json, os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

def generate_json_from_docs(input_file: str, output_path: str = "data/demo_data.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    data = []
    for i, text in enumerate(chunks):
        emb = ollama.embeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"), prompt=text)
        data.append({
            "id": i + 1,
            "text": text,
            "vector": emb["embedding"]
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} chunks with embeddings to {output_path}")
    return output_path


def setup_qdrant(host, port, collection_name, data_path="data/demo_data.json"):
    client = QdrantClient(host=host, port=port)
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    with open(data_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    points = [
        {
            "id": d["id"],
            "vector": d["vector"],
            "payload": {"text": d["text"]},
        }
        for d in docs
    ]

    client.upsert(collection_name=collection_name, points=points)
    print(f"Uploaded {len(points)} documents from {data_path} to collection '{collection_name}'.")
