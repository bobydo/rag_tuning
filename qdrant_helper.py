import json, os, re
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

def extract_metadata_from_text(text, chunk_id, source_file="unknown", document_title="untitled"):
    """
    Extract metadata from text content in a production-ready way.
    In real applications, this would be replaced with:
    - Document parsing libraries (e.g., for PDFs, DOCs)
    - Structured data extraction from HTML/XML
    - Database fields from existing systems
    - User-provided tags and categories
    - NLP-based topic modeling
    - External knowledge graphs
    """
    metadata = {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "document_title": document_title,
        "char_count": len(text),
        "word_count": len(text.split()),
    }
    
    # Extract structural information (headers, lists, code blocks)
    metadata["has_code"] = '```' in text or 'python' in text.lower()
    metadata["has_list"] = any(marker in text for marker in ['- ', '* ', '1. ', '2. '])
    metadata["is_header"] = text.strip().startswith('#')
    
    # Extract header level if it's a header
    if metadata["is_header"]:
        header_match = re.match(r'^(#{1,6})\s', text.strip())
        metadata["header_level"] = len(header_match.group(1)) if header_match else 0
    
    # In production, you would typically:
    # 1. Parse document metadata from file properties
    # 2. Use NLP libraries to extract entities, topics, sentiment
    # 3. Apply domain-specific classification models
    # 4. Use existing taxonomies or ontologies
    # 5. Leverage user-provided tags or categories
    
    # For this demo, we'll simulate some realistic metadata that would come from:
    # - Document management systems
    # - Content management systems  
    # - Database fields
    # - User input forms
    
    # Simulate document type classification (would be ML-based in production)
    if any(term in text.lower() for term in ['install', 'setup', 'configuration', 'config']):
        metadata["document_type"] = "tutorial"
        metadata["category"] = "setup"
    elif any(term in text.lower() for term in ['api', 'function', 'method', 'class']):
        metadata["document_type"] = "reference"
        metadata["category"] = "api_docs"
    elif any(term in text.lower() for term in ['example', 'demo', 'sample']):
        metadata["document_type"] = "example"
        metadata["category"] = "examples"
    elif any(term in text.lower() for term in ['concept', 'overview', 'introduction']):
        metadata["document_type"] = "conceptual"
        metadata["category"] = "concepts"
    else:
        metadata["document_type"] = "general"
        metadata["category"] = "documentation"
    
    # Simulate technical level (would use readability scores in production)
    technical_terms = ['implementation', 'architecture', 'optimization', 'configuration', 'deployment']
    basic_terms = ['overview', 'introduction', 'getting started', 'basics']
    
    if any(term in text.lower() for term in technical_terms):
        metadata["technical_level"] = "advanced"
    elif any(term in text.lower() for term in basic_terms):
        metadata["technical_level"] = "beginner"
    else:
        metadata["technical_level"] = "intermediate"
    
    return metadata

def generate_json_from_docs(input_file: str, output_path: str = "data/demo_data.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    data = []
    for i, text in enumerate(chunks):
        emb = ollama.embeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"), prompt=text)
        
        # Add realistic metadata based on content analysis
        metadata = extract_metadata_from_text(text, i + 1, input_file, "RAG System Documentation")
        
        data.append({
            "id": i + 1,
            "text": text,
            "vector": emb["embedding"],
            "metadata": metadata
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} chunks with embeddings to {output_path}")
    return output_path


def setup_qdrant(host, port, collection_name, data_path="data/demo_data.json"):
    client = QdrantClient(host=host, port=port)
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    with open(data_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    points = [
        PointStruct(
            id=d["id"],
            vector=d["vector"],
            payload={
                "text": d["text"],
                **d.get("metadata", {})  # Include all metadata fields
            }
        )
        for d in docs
    ]

    client.upsert(collection_name=collection_name, points=points)
    print(f"Uploaded {len(points)} documents from {data_path} to collection '{collection_name}'.")
