# Comprehensive RAG Retriever Comparison Tool
from retrievers.multi_query import run_multi_query
from retrievers.parent_doc import run_parent_doc
import os
import argparse
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers import MultiQueryRetriever
from qdrant_client import QdrantClient

# Connection details
HOST = "localhost"
PORT = 6333
COLLECTION = "demo_index"

def print_section_header(title, emoji="🔧"):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"{emoji} {title}")
    print('='*80)

def run_baseline_search(host, port, collection, query, k=3):
    """Run baseline similarity search"""
    print(f"[🔍 Baseline Similarity Search]")
    print("Direct vector similarity search - fast and straightforward")
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
        docs = vectorstore.similarity_search(query, k=k)
        print(f"📋 Found {len(docs)} results:")
        for i, d in enumerate(docs, 1):
            content_preview = d.page_content.replace('\n', ' ')[:120]
            print(f"  {i}. {content_preview}...")
            print(f"     Metadata: {d.metadata}")
            print()
        return docs
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def demonstrate_multi_query_behavior(query):
    """Show what Multi-Query retriever does conceptually"""
    print(f"🔄 Multi-Query Expansion Process:")
    print(f"Original: '{query}'")
    print("🤖 LLM generates variations like:")
    
    if "vector storage" in query.lower():
        variations = [
            "How do vector databases store embeddings?",
            "What is the embedding storage process?", 
            "Vector database storage architecture",
            "Embedding persistence mechanisms"
        ]
    elif "steps" in query.lower():
        variations = [
            "RAG system construction process",
            "Building retrieval augmented generation",
            "Steps to create RAG pipeline",
            "RAG implementation methodology"
        ]
    else:
        variations = [
            "Related query variation 1",
            "Alternative phrasing of question",
            "Different perspective on topic",
            "Broader context query"
        ]
    
    for i, var in enumerate(variations, 1):
        print(f"  {i}. {var}")
    print("➡️ Searches each variation, combines unique results")



def demonstrate_parent_doc_behavior(query):
    """Show what Parent Document retriever does conceptually"""
    print(f"📄 Parent Document Strategy:")
    print(f"Query: '{query}'")
    print("🎯 Two-level approach:")
    print("  1. 🔍 Search small chunks (400 chars) for precise matching")
    print("  2. 📋 Return large parent sections (2000+ chars) for context")
    print("  3. ⚖️ Balance: precision in search, completeness in results")
    print("➡️ Best of both worlds: accurate matching + comprehensive answers")

def compare_all_retrievers(queries, show_behavior=False):
    """Compare all retrievers with given queries"""
    print("🚀 RAG Retriever Comparison Analysis")
    print("Testing different retrieval strategies with Qdrant + Ollama + LangChain\n")
    
    for i, query in enumerate(queries, 1):
        print(f"🎯 TEST QUERY {i}: '{query}'")
        print("=" * 100)
        
        # Baseline
        baseline_docs = run_baseline_search(HOST, PORT, COLLECTION, query, k=2)
        
        if show_behavior:
            print_section_header("Multi-Query Retriever Behavior", "🔄")
            demonstrate_multi_query_behavior(query)
            
            print_section_header("Parent Document Retriever Behavior", "📄")
            demonstrate_parent_doc_behavior(query)
        else:
            print_section_header("Multi-Query Retriever Results", "🔄")
            run_multi_query(HOST, PORT, COLLECTION, query)
            
            print_section_header("Parent Document Retriever Results", "📄")
            run_parent_doc(HOST, PORT, COLLECTION, query)
        
        if i < len(queries):
            print(f"\n{'🔄 NEXT QUERY':<100}")

def print_summary():
    """Print comparison summary"""
    print(f"\n{'✅ COMPARISON COMPLETE':<100}")
    print("\n📊 PRACTICAL RETRIEVER COMPARISON:")
    print("┌─────────────────┬─────────────────────────────────────────────────────────┐")
    print("│ Retriever       │ Real-World Use Case                                     │")
    print("├─────────────────┼─────────────────────────────────────────────────────────┤")
    print("│ 🔍 Baseline     │ Fast, reliable queries - works with any content        │")
    print("│ 🔄 Multi-Query  │ Complex topics - improves recall through query expansion│")
    print("│ 📄 Parent Doc   │ Long documents - precise search, comprehensive context │")
    print("└─────────────────┴─────────────────────────────────────────────────────────┘")

def main():
    parser = argparse.ArgumentParser(description='RAG Retriever Comparison Tool')
    parser.add_argument('--mode', choices=['results', 'behavior'], default='results',
                        help='Show actual results or explain behavior (default: results)')
    parser.add_argument('--query', type=str, help='Custom query to test')
    parser.add_argument('--queries', nargs='+', help='Multiple queries to test')
    
    args = parser.parse_args()
    
    # Default test queries
    default_queries = [
        "How does vector storage work with embeddings?",
        "What are the steps to build a RAG system?",
        "Explain advanced retrieval strategies"
    ]
    
    # Determine queries to use
    if args.query:
        queries = [args.query]
    elif args.queries:
        queries = args.queries
    else:
        queries = default_queries
    
    # Run comparison
    show_behavior = (args.mode == 'behavior')
    compare_all_retrievers(queries, show_behavior)
    print_summary()

if __name__ == "__main__":
    main()
