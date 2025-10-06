import os
import time
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

def run_parent_doc(host, port, collection, query=None):
    if query is None:
        query = "How does vector storage work?"
    
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
        # Parent document retriever setup - NOTE: This needs documents loaded first!
        store = InMemoryStore()
        
        # Create text splitters for parent and child documents
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        # REALITY CHECK: Parent Document complexity in production
        print("⚠️  PRODUCTION REALITY: Parent-Child structure has major challenges!")
        print("   ❌ Complex setup: Need dual storage (vector DB + document store)")
        print("   ❌ Preprocessing: Hard to split documents naturally into parent-child")
        print("   ❌ Sync issues: What if vector DB and document store get out of sync?")
        print("   ❌ Storage cost: ~2x storage (both child embeddings + parent content)")
        print("   ✅ SIMPLER ALTERNATIVE: Use larger chunks (1000+ chars) with overlap")
        print("   ✅ OR: Retrieve neighboring chunks for context expansion")
        print("   Current demo shows 0 results because InMemoryStore is empty\n")

        print("Generating parent document retrieval...")
        
        # PERFORMANCE ANALYSIS: How much time does parent document retrieval take?
        start_time = time.time()
        docs = retriever.invoke(query)
        end_time = time.time()
        
        print(f"⏱️  Parent Document retrieval took: {end_time - start_time:.2f} seconds")
        print(f"📊 Retrieved {len(docs)} documents from parent document strategy")
        
        # REALITY CHECK: Is this practical for production?
        if end_time - start_time > 10:
            print(f"🚨 WARNING: Parent Document is TOO SLOW for real projects!")
            print(f"   - Document store lookup overhead is significant")
            print(f"   - Consider simpler similarity search for speed")
        else:
            print(f"✅ Performance acceptable for production use")
        
        # Compare with single query for reference
        print(f"\n🔍 For comparison - baseline similarity search:")
        single_start = time.time()
        single_docs = vectorstore.similarity_search(query, k=3)
        single_end = time.time()
        print(f"⏱️  Baseline query took: {single_end - single_start:.2f} seconds")
        print(f"📊 Retrieved {len(single_docs)} documents")
        
        # Performance comparison and recommendations
        speed_ratio = (end_time - start_time) / (single_end - single_start)
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   Parent Document is {speed_ratio:.1f}x slower than baseline query")
        if speed_ratio < 3:
            print(f"   🏭 PRODUCTION ASSESSMENT: Parent Document viable for production:")
            print(f"      - Minimal overhead for significantly better context")
            print(f"      - Good balance of speed vs answer quality")
            print(f"      - Suitable for real-time user queries")
        else:
            print(f"   ⚠️  High overhead - evaluate if context benefits justify cost")
        
        if len(docs) == 0:
            print("📝 EXPLANATION: Got 0 results because InMemoryStore is empty")
            print("   Parent Document retriever needs documents loaded with specific parent-child structure")
            print("   Current Qdrant collection has flat chunks, not hierarchical documents")
            print("\n🔄 Falling back to similarity search to show what results would look like...")
            
            fallback_docs = vectorstore.similarity_search(query, k=3)
            print(f"\nFallback Similarity Results ({len(fallback_docs)} documents):")
            for i, d in enumerate(fallback_docs, 1):
                print(f"Result {i}:")
                print(f"Content: {d.page_content[:200]}...")
                print(f"Metadata: {d.metadata}")
                print("-" * 50)
        else:
            print("\nParent Document Results:")
            for i, d in enumerate(docs, 1):
                print(f"Result {i}:")
                print(f"Content: {d.page_content[:200]}...")
                print(f"Metadata: {d.metadata}")
                print("-" * 50)
            
    except Exception as e:
        print(f"💥 TECHNICAL ERROR: {str(e)[:100]}...")
        print("📝 EXPLANATION: Parent document retriever needs specific setup:")
        print("   1. Documents must be loaded into InMemoryStore first")
        print("   2. Documents need parent-child chunk relationships")
        print("   3. Current demo uses flat Qdrant chunks, not hierarchical structure")
        print("\n🔄 Using fallback similarity search...")
        
        # Fallback to simple similarity search with timing
        fallback_start = time.time()
        docs = vectorstore.similarity_search(query, k=3)
        fallback_end = time.time()
        
        print(f"⏱️  Fallback similarity search took: {fallback_end - fallback_start:.2f} seconds")
        print(f"📊 Retrieved {len(docs)} documents")
        
        print("\nFallback Results:")
        for i, d in enumerate(docs, 1):
            print(f"Result {i}:")
            print(f"Content: {d.page_content[:200]}...")
            print(f"Metadata: {d.metadata}")
            print("-" * 50)
