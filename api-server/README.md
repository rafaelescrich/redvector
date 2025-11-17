# RedVector REST API Server

A Rust REST API server that connects to RedVector (rsedis) and provides AI-powered vector search examples, similar to Qdrant's blog examples.

## Features

- 🚀 **RESTful API** built with Axum
- 🔍 **Semantic Search** - Find documents by meaning, not keywords
- 📝 **Document Management** - Add and index documents with embeddings
- 🌐 **Web Interface** - Beautiful HTML frontend for testing in browser
- 🎯 **Qdrant-style Examples** - Similar API patterns to Qdrant blog posts

## Prerequisites

1. **RedVector (rsedis) running** on `localhost:6379`
   ```bash
   cd /path/to/rsedis
   cargo run --release
   ```

2. **Rust toolchain** (1.70+)

## Quick Start

1. **Start RedVector** (in one terminal):
   ```bash
   cd /path/to/rsedis
   cargo run --release
   ```

2. **Start the API server** (in another terminal):
   ```bash
   cd api-server
   cargo run --release
   ```

3. **Open in browser**:
   ```
   http://localhost:8081
   ```

## API Endpoints

### Health Check
```
GET /health
```

### Create Index
```
POST /api/index/:index_name
```

### Add Document
```
POST /api/index/:index_name/document
Body: {
  "id": "doc1",
  "text": "Your document text here",
  "metadata": { "optional": "metadata" }
}
```

### Search
```
GET /api/index/:index_name/search?query=your+search+query&limit=10
```

## Example Usage

### Using cURL

```bash
# Create index
curl -X POST http://localhost:8081/api/index/semantic_search

# Add document
curl -X POST http://localhost:8081/api/index/semantic_search/document \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "text": "Machine learning is a subset of artificial intelligence"
  }'

# Search
curl "http://localhost:8081/api/index/semantic_search/search?query=What%20is%20AI?&limit=5"
```

### Using the Web Interface

1. Open `http://localhost:8081` in your browser
2. The index is automatically created on page load
3. Add documents using the example buttons or custom text
4. Search for similar documents using natural language queries

## Example Use Cases

### 1. Semantic Document Search
Find documents by meaning, not exact keyword matches.

### 2. Question Answering
Match questions to relevant answer documents.

### 3. Content Recommendation
Find similar content based on semantic similarity.

### 4. Knowledge Base Search
Search through documentation or knowledge bases using natural language.

## Embedding Models

Currently uses simplified hash-based embeddings for demonstration. In production, replace with:

- **OpenAI**: `text-embedding-ada-002` (1536D)
- **Sentence Transformers**: `all-MiniLM-L6-v2` (384D)
- **Cohere**: `embed-english-v2.0` (4096D)
- **rust-bert**: Direct Rust integration

## Architecture

```
Browser → Axum Server → Redis Client → RedVector (rsedis)
```

- **Frontend**: HTML/JavaScript in `static/index.html`
- **Backend**: Rust Axum server in `src/main.rs`
- **Vector Store**: RedVector (rsedis) via Redis protocol

## Development

```bash
# Run in development mode
cargo run

# Build for production
cargo build --release

# Run tests (when added)
cargo test
```

## Similar to Qdrant Examples

This API follows similar patterns to Qdrant's blog examples:

- ✅ RESTful endpoints
- ✅ JSON request/response format
- ✅ Semantic search capabilities
- ✅ Document indexing
- ✅ Similarity scoring

## License

Same as rsedis project.

