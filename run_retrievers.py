# main.py
from retrievers.multi_query import run_multi_query
from retrievers.self_querying import run_self_querying
from retrievers.parent_doc import run_parent_doc

# Replace with your actual Qdrant connection details
HOST = "localhost"
PORT = 6333
COLLECTION = "demo_index"

if __name__ == "__main__":
    print("Running retriever demos using Qdrant + Ollama + LangChain\n")
    
    run_self_querying(HOST, PORT, COLLECTION)
    print("\n" + "="*60 + "\n")

    run_multi_query(HOST, PORT, COLLECTION)
    print("\n" + "="*60 + "\n")

    run_parent_doc(HOST, PORT, COLLECTION)
