import os
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
        # Parent document retriever setup
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

        docs = retriever.invoke(query)
        
        print("Parent Document Results:")
        for i, d in enumerate(docs, 1):
            print(f"Result {i}:")
            print(f"Content: {d.page_content[:200]}...")
            print(f"Metadata: {d.metadata}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Note: Parent document retriever needs pre-loaded document store.")
        print(f"Technical error: {str(e)[:100]}...\n")
        print("Using fallback similarity search...")
        
        # Fallback to simple similarity search
        docs = vectorstore.similarity_search(query, k=3)
        print("Fallback Results:")
        for i, d in enumerate(docs, 1):
            print(f"Result {i}:")
            print(f"Content: {d.page_content[:200]}...")
            print(f"Metadata: {d.metadata}")
            print("-" * 50)
