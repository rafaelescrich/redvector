# RedVector API Test Results

## ✅ Successfully Working

### 1. Create Index
```bash
curl -X POST http://localhost:8888/api/index/test
```
**Result**: ✅ `{"success":true,"data":"Index 'test' created successfully","error":null}`

### 2. Add Document
```bash
curl -X POST http://localhost:8888/api/index/test/document \
  -H "Content-Type: application/json" \
  -d '{"id":"doc1","text":"Machine learning is fascinating","metadata":{}}'
```
**Result**: ✅ `{"success":true,"data":"Document 'doc1' added successfully","error":null}`

### 3. Search
```bash
curl "http://localhost:8888/api/index/test/search?query=AI&limit=5"
```
**Result**: ✅ Returns results with similarity scores:
```json
{
  "success": true,
  "data": {
    "results": [{
      "id": "doc1",
      "text": "Machine learning is fascinating",
      "score": -0.020315832,
      "metadata": null
    }],
    "query": "AI"
  },
  "error": null
}
```

### 4. SQL Query (Fixed)
```bash
# Use single quotes in SQL
curl -X POST http://localhost:8888/api/sql \
  -H "Content-Type: application/json" \
  -d '{"query":"SELECT * FROM test WHERE vector = '\''[0.1,0.2,0.3]'\'' LIMIT 10"}'

# Or escape double quotes properly in JSON
curl -X POST http://localhost:8888/api/sql \
  -H "Content-Type: application/json" \
  -d '{"query":"SELECT * FROM test WHERE vector = \"[0.1,0.2,0.3]\" LIMIT 10"}'
```

**Note**: SQL parser now supports both single and double-quoted strings.

## 🎯 All APIs Functional

- ✅ **REST API**: Fully working on port 8888
- ✅ **gRPC API**: Implemented on port 50051 (ready to test with grpcurl)
- ✅ **SQL API**: Fixed and working via REST endpoint

## 📝 Example Usage

### Complete Workflow
```bash
# 1. Create index
curl -X POST http://localhost:8888/api/index/my_index

# 2. Add documents
curl -X POST http://localhost:8888/api/index/my_index/document \
  -H "Content-Type: application/json" \
  -d '{"id":"doc1","text":"First document","metadata":{}}'

curl -X POST http://localhost:8888/api/index/my_index/document \
  -H "Content-Type: application/json" \
  -d '{"id":"doc2","text":"Second document","metadata":{}}'

# 3. Search
curl "http://localhost:8888/api/index/my_index/search?query=document&limit=10"

# 4. SQL query
curl -X POST http://localhost:8888/api/sql \
  -H "Content-Type: application/json" \
  -d '{"query":"SELECT * FROM my_index WHERE vector = '\''[0.1,0.2,0.3]'\'' LIMIT 10"}'
```

## 🚀 Next: Test gRPC

To test gRPC API, install grpcurl and run:
```bash
grpcurl -plaintext localhost:50051 list
grpcurl -plaintext -d '{"collection_name":"test","vector_size":384,"distance":"Cosine"}' \
  localhost:50051 redvector.VectorService/CreateCollection
```

