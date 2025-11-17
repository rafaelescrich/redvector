# rsedis Examples

Examples demonstrating how to use rsedis with Redisearch for various use cases.

## Semantic Search Example

Demonstrates semantic search using text embeddings, similar to Qdrant's blog examples.

### Features

- Create vector indexes for semantic search
- Add documents with embeddings
- Search for similar documents using vector similarity
- Integration examples for real embedding models

### Usage

```bash
# Start rsedis
./target/release/rsedis

# In another terminal, run the example
cargo run --example semantic_search --release
```

### Real Embedding Models

The example includes placeholder embeddings. To use real embeddings:

1. **Sentence Transformers** (Python)
   - Model: `all-MiniLM-L6-v2` (384D)
   - Fast and efficient for most use cases

2. **OpenAI Embeddings**
   - Model: `text-embedding-ada-002` (1536D)
   - High quality, requires API key

3. **Cohere Embeddings**
   - Model: `embed-english-v2.0` (4096D)
   - Good for multilingual support

4. **rust-bert** (Rust)
   - Direct Rust integration
   - No Python dependency

### Example Use Cases

1. **Document Search**: Find documents by meaning, not keywords
2. **Recommendation Systems**: Find similar items
3. **Question Answering**: Match questions to answers
4. **Content Discovery**: Find related content

## Integration with AI Frameworks

### Python (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer
import redis

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to rsedis
r = redis.Redis(host='localhost', port=6379, db=0)

# Create index
r.execute_command('FT.CREATE', 'docs', 'SCHEMA', 
                  'title', 'TEXT', 
                  'embedding', 'VECTOR(384)')

# Add document
text = "Machine learning is fascinating"
embedding = model.encode(text)
embedding_str = ','.join(map(str, embedding))

r.execute_command('FT.ADD', 'docs', 'doc1', '1.0', 'FIELDS',
                  'title', text,
                  'embedding', embedding_str)

# Search
query = "What is AI?"
query_embedding = model.encode(query)
query_str = ','.join(map(str, query_embedding))

results = r.execute_command('FT.SEARCH', 'docs', query_str, 
                            'LIMIT', '0', '10', 'WITHSCORES')
```

### Rust (rust-bert)

```rust
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType
};
use redis::Commands;

// Load model
let model = SentenceEmbeddingsBuilder::remote(
    SentenceEmbeddingsModelType::AllMiniLmL6V2
).create_model()?;

// Connect to rsedis
let client = redis::Client::open("redis://127.0.0.1:6379/")?;
let mut conn = client.get_connection()?;

// Generate embedding
let embeddings = model.encode(&["Machine learning is fascinating"])?;
let embedding = &embeddings[0];

// Add to index
let embedding_str = embedding.iter()
    .map(|v| v.to_string())
    .collect::<Vec<_>>()
    .join(",");

redis::cmd("FT.ADD")
    .arg("docs")
    .arg("doc1")
    .arg("1.0")
    .arg("FIELDS")
    .arg("title")
    .arg("Machine learning is fascinating")
    .arg("embedding")
    .arg(&embedding_str)
    .query(&mut conn)?;
```

## Performance Tips

1. **Batch Operations**: Add multiple documents in batches
2. **Index Optimization**: Use appropriate vector dimensions
3. **Query Optimization**: Limit result sets appropriately
4. **Connection Pooling**: Reuse connections for better performance

## Next Steps

- Add more examples (image search, recommendation systems)
- Add performance benchmarks
- Add integration examples for more embedding models
- Add examples for hybrid search (text + vector)

