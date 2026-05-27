# ConversionAdmin MCP RAG Server

## Overview

This project creates an MCP (Model Context Protocol) server for semantic search over internal Conversion Admin documentation using:

- Node.js MCP server
- ChromaDB vector database
- Sentence Transformers embeddings (`all-MiniLM-L6-v2`)
- Python embedding worker
- Docling document parser
- Incremental indexing
- Semantic search

The MCP tool can then be used inside VS Code / MCP-enabled clients to search internal documentation using natural language.

---

# Architecture Flow

```text
User Query
   ↓
MCP Client (VS Code / Claude Desktop / Cursor)
   ↓
Node.js MCP Server
   ↓
Semantic Search
   ↓
ChromaDB Vector Store
   ↓
Embeddings generated using MiniLM
   ↓
Relevant documentation chunks returned
```

---

# Full Data Indexing Flow

```text
Files (.txt/.md/.pdf/.docx)
   ↓
Docling Parser (Python)
   ↓
Structured text blocks
   ↓
Semantic chunk creation
   ↓
Node.js server sends chunk text to Python embedding worker
   ↓
SentenceTransformer generates embeddings
   ↓
Embeddings stored in ChromaDB
```

---

# Tech Stack

| Component | Purpose |
|---|---|
| Node.js | MCP server |
| Python | Embedding + parsing workers |
| ChromaDB | Vector database |
| sentence-transformers | Embedding model |
| Docling | Document parsing |
| MCP SDK | MCP protocol support |

---

# Folder Structure

```text
ConversionAdmin-RAG-MCP/
│
├── server.js
├── embed_server.py
├── docling_parser.py
├── package.json
├── mcp.yaml
│
├── chroma_db/
│
└── Conversion-Admin/
    ├── APIs/
    ├── WebServices/
    ├── Docs/
    └── PDFs/
```

---

# Step 1 — Install Dependencies

## Node.js Dependencies

```bash
npm install @modelcontextprotocol/sdk chromadb
```

---

## Python Dependencies

```bash
pip install sentence-transformers torch docling
```

---

# Step 2 — Python Embedding Worker

Create:

```text
embed_server.py
```

## Purpose

This script:

- Loads MiniLM once
- Keeps model in memory
- Receives text from Node.js via stdin
- Returns embeddings via stdout

This is MUCH faster than reloading the model for every embedding.

Benefits:

- Model loads only once
- Much faster embedding generation
- Lower memory usage

---

# Final Result

- MCP-compatible semantic search server
- Persistent embedding pipeline
- ChromaDB vector storage
- Incremental indexing
- Docling parsing
- Semantic retrieval
- VS Code MCP integration
- Fault-tolerant chunk embedding
- Background indexing

---

# Future Improvements

Possible next upgrades:

- Hybrid BM25 + vector search
- Metadata filtering
- Reranking models
- Streaming responses
- Better chunking strategy
- Persistent ChromaDB disk storage
- Multi-collection support
- Async embedding batches
- GPU acceleration
- OpenAI-compatible embeddings
- Query caching
- Source citations
- Tool auth
- Web UI

