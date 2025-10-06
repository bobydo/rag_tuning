import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers import MultiQueryRetriever
from qdrant_client import QdrantClient

def run_multi_query(host, port, collection):
    print("[Multi-Query Retriever]")
    print("Generates multiple sub-queries for broader recall.")
    example_query = "How does LangChain work with Qdrant?"
    print(f"Example Query: {example_query}")
    print("→ Expands into queries like:")
    print(" - 'LangChain integration Qdrant'")
    print(" - 'Use LangChain to store embeddings'")
    print(" - 'Qdrant vector store example'")


    client = QdrantClient(host=host, port=port)
    embedding_fn = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vectorstore = QdrantVectorStore(client=client, collection_name=collection, embedding=embedding_fn)

    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"))
    retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

    query = "Explain how Qdrant ensures scalability"
    print(f"\nQuery: {query}\n")

    docs = retriever.invoke(query)
    for d in docs:
        print(f"• {d.metadata.get('title', 'No title')} → {d.page_content}")
