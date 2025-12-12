#!/bin/bash
# Integration test script for RedVector with REST API

set -e

echo "🧪 Testing RedVector Integration"
echo "================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if rsedis is running
echo -e "${YELLOW}Checking if rsedis is running...${NC}"
if ! redis-cli -p 6379 ping > /dev/null 2>&1; then
    echo -e "${RED}❌ rsedis is not running on port 6379${NC}"
    echo "Please start it with: cargo run --release"
    exit 1
fi
echo -e "${GREEN}✅ rsedis is running${NC}"

# Check if API server is running
echo -e "${YELLOW}Checking if API server is running...${NC}"
if ! curl -s http://localhost:8081/health > /dev/null 2>&1; then
    echo -e "${RED}❌ API server is not running on port 8081${NC}"
    echo "Please start it with: cd api-server && cargo run --release"
    exit 1
fi
echo -e "${GREEN}✅ API server is running${NC}"

# Test REST API
echo ""
echo -e "${YELLOW}Testing REST API...${NC}"

# Test health endpoint
echo "1. Testing /health"
HEALTH=$(curl -s http://localhost:8081/health)
if [[ "$HEALTH" == *"ok"* ]] || [[ "$HEALTH" == *"OK"* ]]; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed: $HEALTH${NC}"
fi

# Test create index
echo "2. Testing POST /api/index/test_index"
CREATE_RESPONSE=$(curl -s -X POST http://localhost:8081/api/index/test_index)
if [[ "$CREATE_RESPONSE" == *"success"* ]] || [[ "$CREATE_RESPONSE" == *"OK"* ]]; then
    echo -e "${GREEN}✅ Index creation passed${NC}"
else
    echo -e "${YELLOW}⚠️  Index creation response: $CREATE_RESPONSE${NC}"
fi

# Test add document
echo "3. Testing POST /api/index/test_index/document"
DOC_RESPONSE=$(curl -s -X POST http://localhost:8081/api/index/test_index/document \
    -H "Content-Type: application/json" \
    -d '{"id":"doc1","text":"Machine learning is fascinating","metadata":{}}')
if [[ "$DOC_RESPONSE" == *"success"* ]] || [[ "$DOC_RESPONSE" == *"OK"* ]]; then
    echo -e "${GREEN}✅ Document addition passed${NC}"
else
    echo -e "${YELLOW}⚠️  Document addition response: $DOC_RESPONSE${NC}"
fi

# Test search
echo "4. Testing GET /api/index/test_index/search"
SEARCH_RESPONSE=$(curl -s "http://localhost:8081/api/index/test_index/search?query=AI&limit=5")
if [[ "$SEARCH_RESPONSE" == *"results"* ]] || [[ "$SEARCH_RESPONSE" == *"success"* ]]; then
    echo -e "${GREEN}✅ Search passed${NC}"
    echo "   Response preview: ${SEARCH_RESPONSE:0:100}..."
else
    echo -e "${YELLOW}⚠️  Search response: $SEARCH_RESPONSE${NC}"
fi

# Test Redis commands directly
echo ""
echo -e "${YELLOW}Testing Redis commands...${NC}"
echo "5. Testing FT.CREATE"
FT_CREATE=$(redis-cli -p 6379 FT.CREATE test_redis_index SCHEMA vector_field VECTOR 384 2>&1)
if [[ "$FT_CREATE" == "OK" ]]; then
    echo -e "${GREEN}✅ FT.CREATE passed${NC}"
else
    echo -e "${YELLOW}⚠️  FT.CREATE response: $FT_CREATE${NC}"
fi

echo ""
echo -e "${GREEN}✅ Integration tests completed!${NC}"

