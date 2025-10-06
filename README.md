# RAG Retriever Tuning & Comparison Demo

This project demonstrates different retrieval strategies for RAG (Retrieval-Augmented Generation) systems using LangChain, Qdrant, and Ollama. It automatically converts text documents into embeddings and provides comprehensive comparisons between various retriever types.

## 🚀 Quick Start

### 1. Setup Data
1. Place your text document in the project root as `my_doc.txt`
2. Run the data preparation:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
3. The script will:
   - Split your document into chunks (500 chars each)
   - Generate embeddings using Ollama (`nomic-embed-text`)
   - Save them to `data/demo_data.json`
   - Upload to Qdrant under collection `demo_index`

### 2. Compare Retrievers

#### Basic Comparison
```bash
python run_retrievers.py
```
This runs all retriever types with a single query to compare their outputs.

#### Comprehensive Analysis
```bash
python enhanced_retriever_demo.py
```
This provides detailed comparison across multiple test queries with formatted output.

## 🔍 Retriever Types Demonstrated

| Retriever | Purpose | Best For |
|-----------|---------|----------|
| **Baseline Similarity** | Simple vector search | Quick, straightforward retrieval |
| **Multi-Query** | Generates multiple related queries | Broader recall, diverse perspectives |
| **Self-Querying** | LLM interprets natural language filters | Complex queries with semantic constraints |
| **Parent Document** | Returns larger context for small matches | Comprehensive context while maintaining precision |

## 🛠 Environment Variables
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## 📊 Usage Examples

### Test Different Queries
```bash
# Compare how each retriever handles different question types
python enhanced_retriever_demo.py
```

### Custom Query Testing
Modify the queries in `enhanced_retriever_demo.py`:
```python
queries = [
    "Your custom query here",
    "Another test question",
    "Specific domain question"
]
```

## 🔧 Prerequisites
- Python 3.8+
- Qdrant running locally (default: localhost:6333)
- Ollama with models: `llama3` and `nomic-embed-text`
- Virtual environment (recommended)

## 📁 Project Structure
```
├── main.py                     # Data preparation script
├── run_retrievers.py          # Basic retriever comparison
├── enhanced_retriever_demo.py # Comprehensive analysis tool
├── qdrant_helper.py          # Qdrant utilities
├── retrievers/               # Individual retriever implementations
│   ├── multi_query.py
│   ├── self_querying.py
│   └── parent_doc.py
└── data/
    └── demo_data.json        # Generated embeddings data
```
