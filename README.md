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
Runs all retriever types with default test queries showing actual results.

#### Behavior Analysis Mode
```bash
python run_retrievers.py --mode behavior
```
Explains what each retriever does conceptually without running expensive LLM calls.

#### Custom Queries
```bash
# Single custom query
python run_retrievers.py --query "Your question here"

# Multiple custom queries  
python run_retrievers.py --queries "Query 1" "Query 2" "Query 3"
```

## 🔍 Practical Retriever Types

| Retriever | Purpose | Real-World Use Case |
|-----------|---------|---------------------|
| **Baseline Similarity** | Fast vector search | Works reliably with any content type |
| **Multi-Query** | Generates multiple query variations | Improves recall for complex topics |
| **Parent Document** | Small chunks search, large context return | Best for long documents needing comprehensive answers |

### ❌ Why We Removed Self-Querying
Self-querying requires rich metadata that most real-world content (emails, PDFs, chat logs) simply doesn't have, and adds expensive LLM calls to every query. Multi-query retrieval achieves better results without these dependencies.

## 🛠 Environment Variables
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## 📊 Usage Examples

### Default Test Queries
```bash
# Run with built-in test queries
python run_retrievers.py
```

### Custom Query Testing
```bash
# Test a specific question
python run_retrievers.py --query "How do embeddings work?"

# Test multiple questions
python run_retrievers.py --queries "What is RAG?" "How to optimize retrieval?" "Vector database benefits"
```

### Understanding Retriever Behavior
```bash
# Learn what each retriever does without expensive LLM calls
python run_retrievers.py --mode behavior --query "Your question"
```

## 🔧 Prerequisites
- Python 3.8+
- Qdrant running locally (default: localhost:6333)
- Ollama with models: `llama3` and `nomic-embed-text`
- Virtual environment (recommended)

## 📁 Project Structure
```
├── main.py                     # Data preparation script
├── run_retrievers.py          # Practical retriever comparison tool
├── qdrant_helper.py          # Qdrant utilities  
├── my_doc.txt                # Source document for embeddings

├── retrievers/               # Practical retriever implementations
│   ├── multi_query.py       # Query expansion for better recall
│   └── parent_doc.py        # Hierarchical document retrieval
└── data/
    └── demo_data.json        # Generated embeddings data
```

## ⚡ Performance Benchmarks

Based on real testing with 61 document chunks and local Ollama setup:

| Retriever Type | Response Time | Documents Retrieved | Practical Use |
|----------------|---------------|-------------------|---------------|
| **Baseline Similarity** | ~2.14 seconds | 3 documents | ✅ Production ready |
| **Multi-Query** | ~76.32 seconds | 12 documents | ❌ Too slow for real-time |
| **Parent Document** | ~3-4 seconds | Variable | ✅ Production ready |

### 🤔 So When Use Multi-Query?

**❌ NOT practical for real projects with:**
- Local Ollama models (70+ second LLM inference)
- Real-time user queries (users expect <3 seconds)
- Single-server deployments
- Budget constraints

**✅ Multi-query COULD work with:**
- **Cloud LLMs**: GPT-4/Claude (~2-3 seconds vs 70+ seconds)
- **GPU clusters**: Dedicated hardware for faster local inference
- **Async processing**: Background query generation, not real-time
- **Cached patterns**: Pre-generate variations for common searches
- **Research/offline**: When quality > speed matters

### 💡 The Reality
- **Bottleneck**: LLM query generation (~74 seconds), not vector search (~2 seconds)
- **Speed difference**: Multi-query is 36x slower than single query
- **Production advice**: Stick with baseline similarity search unless you have cloud LLM infrastructure

**Bottom line**: Multi-query improves recall but kills performance with local models. Choose based on your infrastructure and use case!



