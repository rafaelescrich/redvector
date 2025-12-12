# Testing gRPC API

## Installation Options

### Option 1: Download Pre-built Binary (Recommended)

```bash
# Download latest release
wget https://github.com/fullstorydev/grpcurl/releases/latest/download/grpcurl_$(uname -m)_linux.tar.gz

# Or specific version
wget https://github.com/fullstorydev/grpcurl/releases/download/v1.8.9/grpcurl_1.8.9_linux_x86_64.tar.gz
tar -xzf grpcurl_*.tar.gz
sudo mv grpcurl /usr/local/bin/
grpcurl --version
```

### Option 2: Package Manager

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install grpcurl

# Or snap
sudo snap install grpcurl
```

### Option 3: Build from Source (Requires Go)

```bash
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
```

### Option 4: Python Client (Alternative)

```bash
pip install grpcio grpcio-tools

# Generate Python client from proto
python -m grpc_tools.protoc -Iapi-server/proto \
  --python_out=. --grpc_python_out=. \
  api-server/proto/vector.proto
```

## Testing gRPC API

Once grpcurl is installed:

```bash
# 1. List available services
grpcurl -plaintext localhost:50051 list

# 2. List methods in VectorService
grpcurl -plaintext localhost:50051 list redvector.VectorService

# 3. Create a collection
grpcurl -plaintext -d '{
  "collection_name": "test_grpc",
  "vector_size": 384,
  "distance": "Cosine"
}' localhost:50051 redvector.VectorService/CreateCollection

# 4. Add vectors (Upsert)
grpcurl -plaintext -d '{
  "collection_name": "test_grpc",
  "points": [
    {
      "id": 1,
      "vector": [0.1, 0.2, 0.3],
      "payload": {}
    }
  ]
}' localhost:50051 redvector.VectorService/Upsert

# 5. Search
grpcurl -plaintext -d '{
  "collection_name": "test_grpc",
  "query_vector": [0.1, 0.2, 0.3],
  "top": 5
}' localhost:50051 redvector.VectorService/Search

# 6. Get collection info
grpcurl -plaintext -d '{
  "collection_name": "test_grpc"
}' localhost:50051 redvector.VectorService/GetCollectionInfo

# 7. Delete collection
grpcurl -plaintext -d '{
  "collection_name": "test_grpc"
}' localhost:50051 redvector.VectorService/DeleteCollection
```

## Quick Test Script

```bash
#!/bin/bash
# Test gRPC API

echo "Testing gRPC API on localhost:50051"
echo "===================================="

# Check if server is running
if ! grpcurl -plaintext localhost:50051 list > /dev/null 2>&1; then
    echo "❌ gRPC server not running on port 50051"
    exit 1
fi

echo "✅ Server is running"
echo ""

# List services
echo "Available services:"
grpcurl -plaintext localhost:50051 list
echo ""

# Test CreateCollection
echo "Creating collection..."
grpcurl -plaintext -d '{
  "collection_name": "test_grpc",
  "vector_size": 384,
  "distance": "Cosine"
}' localhost:50051 redvector.VectorService/CreateCollection

echo ""
echo "✅ gRPC API is working!"
```

## Python Example

```python
import grpc
from api_server.proto import vector_pb2
from api_server.proto import vector_pb2_grpc

# Connect to server
channel = grpc.insecure_channel('localhost:50051')
stub = vector_pb2_grpc.VectorServiceStub(channel)

# Create collection
request = vector_pb2.CreateCollectionRequest(
    collection_name="test_grpc",
    vector_size=384,
    distance="Cosine"
)
response = stub.CreateCollection(request)
print(f"Created: {response.success}")

# Search
search_request = vector_pb2.SearchRequest(
    collection_name="test_grpc",
    query_vector=[0.1] * 384,
    top=5
)
search_response = stub.Search(search_request)
print(f"Found {len(search_response.result)} results")
```

