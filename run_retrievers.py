# Enhanced retriever comparison script
from retrievers.multi_query import run_multi_query
from retrievers.self_querying import run_self_querying
from retrievers.parent_doc import run_parent_doc
import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

# Replace with your actual Qdrant connection details
HOST = "localhost"
PORT = 6333
COLLECTION = "demo_index"

def run_simple_similarity_search(host, port, collection, query):
    """Run a simple similarity search for comparison"""
    print("[Baseline Similarity Search]")
    print("Simple vector similarity search without any enhancement")
    print(f"Query: {query}\n")
    
    client = QdrantClient(host=host, port=port)
    embedding_fn = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name=collection, 
        embedding=embedding_fn,
        content_payload_key="text"
    )
    
    try:
        docs = vectorstore.similarity_search(query, k=3)
        for i, d in enumerate(docs, 1):
            print(f"Result {i}:")
            print(f"Content: {d.page_content[:200]}...")
            print(f"Metadata: {d.metadata}")
            print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== Enhanced Retriever Comparison Demo ===")
    print("Comparing different retrieval strategies using Qdrant + Ollama + LangChain\n")
    
    # Use the same query for all retrievers to compare results
    query = "How does vector storage work with embeddings?"
    
    print("🔍 QUERY FOR ALL RETRIEVERS:")
    print(f"'{query}'\n")
    print("="*80 + "\n")
    
    # Baseline comparison
    run_simple_similarity_search(HOST, PORT, COLLECTION, query)
    print("\n" + "="*80 + "\n")
    
    # Self-querying retriever
    run_self_querying(HOST, PORT, COLLECTION, query)
    print("\n" + "="*80 + "\n")

    # Multi-query retriever
    run_multi_query(HOST, PORT, COLLECTION, query)
    print("\n" + "="*80 + "\n")

    # Parent document retriever
    run_parent_doc(HOST, PORT, COLLECTION, query)
