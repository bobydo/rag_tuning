import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from qdrant_client import QdrantClient

def run_self_querying(host, port, collection):
    print("[Self-Querying Retriever]")
    print("Uses an LLM to interpret filters from natural language.")
    example_query = "Find documents about embeddings from LangChain."
    print(f"Example Query: {example_query}")
    print("→ LLM interprets this as:")
    print("Filter: {'topic': 'embeddings', 'source': 'LangChain'}")

    client = QdrantClient(host=host, port=port)
    embedding_fn = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vectorstore = QdrantVectorStore(client=client, collection_name=collection, embedding=embedding_fn)

    # Define metadata fields for self-querying
    metadata_field_info = [
        AttributeInfo(
            name="text",
            description="The content of the document",
            type="string",
        ),
    ]

    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"))
    
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        "Documents about building RAG systems with LangChain and Qdrant",
        metadata_field_info,
        verbose=True
    )

    query = "Find information about vector databases"
    print(f"\nQuery: {query}\n")

    try:
        docs = retriever.invoke(query)
        for d in docs:
            print(f"• {d.metadata.get('title', 'No title')} → {d.page_content}")
    except Exception as e:
        print(f"Note: Self-querying retriever requires more complex metadata setup. Error: {e}")
        # Fallback to simple similarity search
        docs = vectorstore.similarity_search(query)
        for d in docs:
            print(f"• Fallback search → {d.page_content[:100]}...")
