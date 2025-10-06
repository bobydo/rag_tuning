# Qdrant Embedding Generator Demo

This version automatically converts any local text file into embedded JSON data and uploads it to Qdrant.

## How it works
1. Place your text document in the project root as `my_doc.txt`
2. Run:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
3. The script will:
   - Split your document into chunks (500 chars each)
   - Generate embeddings using Ollama (`nomic-embed-text`)
   - Save them to `data/demo_data.json`
   - Upload to Qdrant under collection `demo_index`

## Env vars
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text
```
