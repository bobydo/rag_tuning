# Enhanced retriever comparison with detailed analysis
from retrievers.multi_query import run_multi_query
from retrievers.self_querying import run_self_querying
from retrievers.parent_doc import run_parent_doc
import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

# Connection details
HOST = "localhost"
PORT = 6333
COLLECTION = "demo_index"

def run_simple_similarity_search(host, port, collection, query):
    """Run a simple similarity search for comparison"""
    print("[🔍 Baseline Similarity Search]")
    print("Simple vector similarity search without any enhancement")
    print(f"Query: '{query}'\n")
    
    client = QdrantClient(host=host, port=port)
    embedding_fn = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name=collection, 
        embedding=embedding_fn,
        content_payload_key="text"
    )
    
    try:
        docs = vectorstore.similarity_search(query, k=2)  # Limit to 2 for cleaner output
        print(f"📋 Found {len(docs)} results:")
        for i, d in enumerate(docs, 1):
            content_preview = d.page_content.replace('\n', ' ')[:150]
            print(f"  {i}. {content_preview}...")
            print()
    except Exception as e:
        print(f"❌ Error: {e}")

def print_section_header(title, emoji="🔧"):
    print(f"\n{'='*80}")
    print(f"{emoji} {title}")
    print('='*80)

def main():
    print("🚀 RAG Retriever Comparison Demo")
    print("Testing different retrieval strategies with Qdrant + Ollama + LangChain")
    
    # Test queries that highlight different retriever strengths
    queries = [
        "How does vector storage work with embeddings?",
        "What are the steps to build a RAG system?",
        "Explain the retrieval process in detail"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🎯 TEST QUERY {i}: '{query}'")
        print("=" * 80)
        
        # Baseline
        run_simple_similarity_search(HOST, PORT, COLLECTION, query)
        
        print_section_header("Multi-Query Retriever", "🔍")
        run_multi_query(HOST, PORT, COLLECTION, query)
        
        print_section_header("Self-Querying Retriever", "🧠") 
        run_self_querying(HOST, PORT, COLLECTION, query)
        
        print_section_header("Parent Document Retriever", "📄")
        run_parent_doc(HOST, PORT, COLLECTION, query)
        
        if i < len(queries):
            print(f"\n{'🔄 NEXT QUERY':<80}")
            
    print(f"\n{'✅ COMPARISON COMPLETE':<80}")
    print("\n📊 SUMMARY:")
    print("• Baseline: Simple similarity search")
    print("• Multi-Query: Generates multiple related queries for broader search")
    print("• Self-Querying: Uses LLM to interpret and structure queries")
    print("• Parent Document: Searches chunks but returns larger context")

if __name__ == "__main__":
    main()