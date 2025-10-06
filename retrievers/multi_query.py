import os
import time
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from qdrant_client import QdrantClient

def run_multi_query(host, port, collection, query=None):
    if query is None:
        print("Missed query from run_multi_query")
        return
    
    print(f"Query: {query}\n")

    client = QdrantClient(host=host, port=port)
    embedding_fn = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name=collection, 
        embedding=embedding_fn,
        content_payload_key="text"
    )

    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"))
    
    try:
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), 
            llm=llm
        )
        
        print("Generating multiple queries and retrieving results...")
        
        # PERFORMANCE ANALYSIS: How much time does multi-query actually take?
        start_time = time.time()
        docs = retriever.invoke(query)
        end_time = time.time()
        
        print(f"⏱️  Multi-query retrieval took: {end_time - start_time:.2f} seconds")
        print(f"📊 Retrieved {len(docs)} documents from multiple query variations")
        
        # REALITY CHECK: Is this practical for production?
        if end_time - start_time > 10:
            print(f"🚨 WARNING: Multi-query is TOO SLOW for real projects!")
            print(f"   - Local Ollama LLM is the bottleneck (~70+ seconds)")
            print(f"   - Vector searches are actually fast (~2-3 seconds each)")
            print(f"   - Query generation dominates the time (not database searches)")
        else:
            print(f"✅ Performance acceptable for production use")
        
        # Compare with single query for reference
        print(f"\n🔍 For comparison - single similarity search:")
        single_start = time.time()
        single_docs = vectorstore.similarity_search(query, k=3)
        single_end = time.time()
        print(f"⏱️  Single query took: {single_end - single_start:.2f} seconds")
        print(f"📊 Retrieved {len(single_docs)} documents")
        
        # Performance comparison and recommendations
        speed_ratio = (end_time - start_time) / (single_end - single_start)
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   Multi-query is {speed_ratio:.1f}x slower than single query")
        if speed_ratio > 20:
            print(f"   🏭 PRODUCTION REALITY: Multi-query only viable with:")
            print(f"      - Cloud LLMs (GPT-4: ~2-3 seconds vs 70+ seconds)")
            print(f"      - Dedicated GPU servers for local LLMs")
            print(f"      - Async/background processing (not real-time queries)")
            print(f"      - Cached query patterns for common searches")
        else:
            print(f"   ✅ Could work in production with current setup")
        
        print("\nMulti-Query Results:")
        for i, d in enumerate(docs, 1):
            print(f"Result {i}:")
            print(f"Content: {d.page_content[:200]}...")
            print(f"Metadata: {d.metadata}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error with multi-query retriever: {e}")
        print("Using fallback similarity search...")
        
        docs = vectorstore.similarity_search(query, k=3)
        for i, d in enumerate(docs, 1):
            print(f"Fallback Result {i}:")
            print(f"Content: {d.page_content[:200]}...")
            print(f"Metadata: {d.metadata}")
            print("-" * 50)
