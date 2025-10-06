import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers import MultiQueryRetriever
from qdrant_client import QdrantClient

def run_multi_query(host, port, collection, query=None):
    print("[Multi-Query Retriever]")
    print("Generates multiple sub-queries for broader recall and diverse perspectives.")
    print("Example: 'How does LangChain work?' → expands to multiple related queries")
    print("Benefits: Captures different aspects and reduces retrieval blind spots")

    if query is None:
        query = "Explain how Qdrant ensures scalability"
    
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
        docs = retriever.invoke(query)
        
        print("Multi-Query Results:")
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
