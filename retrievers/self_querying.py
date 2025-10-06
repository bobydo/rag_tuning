import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from qdrant_client import QdrantClient

def run_self_querying(host, port, collection, query=None):
    print("[Self-Querying Retriever]")
    print("Uses an LLM to interpret filters from natural language.")
    print("Example: 'Find documents about embeddings from LangChain' → interpreted as structured query")
    print("Benefits: Can apply semantic filters and constraints based on natural language")

    if query is None:
        query = "Find information about vector databases"
    
    print(f"Query: {query}\n")

    client = QdrantClient(host=host, port=port)
    embedding_fn = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name=collection, 
        embedding=embedding_fn,
        content_payload_key="text"
    )

    # Define metadata fields for self-querying
    metadata_field_info = [
        AttributeInfo(
            name="text",
            description="The content of the document about RAG systems and vector databases",
            type="string",
        ),
    ]

    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"))
    
    try:
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vectorstore,
            "Documents about building RAG systems with LangChain and Qdrant",
            metadata_field_info,
            verbose=True
        )
        docs = retriever.invoke(query)
        
        print("Self-Query Results:")
        for i, d in enumerate(docs, 1):
            print(f"Result {i}:")
            print(f"Content: {d.page_content[:200]}...")
            print(f"Metadata: {d.metadata}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Note: Self-querying requires complex metadata. Using fallback similarity search.")
        print(f"Technical error: {str(e)[:100]}...\n")
        
        # Fallback to simple similarity search
        docs = vectorstore.similarity_search(query, k=3)
        print("Fallback Results:")
        for i, d in enumerate(docs, 1):
            print(f"Result {i}:")
            print(f"Content: {d.page_content[:200]}...")
            print(f"Metadata: {d.metadata}")
            print("-" * 50)
