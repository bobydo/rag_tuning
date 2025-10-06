import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

def run_parent_doc(host, port, collection):
    print("[Parent Document Retriever]")
    print("Handles large or hierarchical documents with chunk linking.")
    print("Example: Parent doc split into sections, retrieved as a whole.")

    client = QdrantClient(host=host, port=port)
    embedding_fn = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vectorstore = QdrantVectorStore(client=client, collection_name=collection, embedding=embedding_fn)

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

    query = "How does vector storage work?"
    print(f"\nQuery: {query}\n")

    try:
        docs = retriever.invoke(query)
        for d in docs:
            print(f"• Parent doc → {d.page_content[:100]}...")
    except Exception as e:
        print(f"Note: Parent document retriever requires document store setup. Error: {e}")
        # Fallback to simple similarity search
        docs = vectorstore.similarity_search(query)
        for d in docs:
            print(f"• Fallback search → {d.page_content[:100]}...")
