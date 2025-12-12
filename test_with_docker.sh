#!/bin/bash

# Docker-based test script for RedVector
# This script builds and tests RedVector using Docker

set -e

echo "=========================================="
echo "RedVector Docker Test Suite"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not available. Please install Docker or use run_tests.sh for local testing.${NC}"
    exit 1
fi

# Build Docker image
echo -e "${YELLOW}[1/3] Building Docker image...${NC}"
docker build -t redvector:test .
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Start container
echo -e "${YELLOW}[2/3] Starting container...${NC}"
docker run -d --name redvector-test -p 6379:6379 redvector:test
sleep 2

# Test connectivity
echo -e "${YELLOW}[3/3] Testing server...${NC}"
if command -v redis-cli &> /dev/null; then
    if redis-cli -h 127.0.0.1 -p 6379 PING | grep -q PONG; then
        echo -e "${GREEN}✓ Server is responding${NC}"
    else
        echo -e "${RED}✗ Server not responding${NC}"
        docker logs redvector-test
        docker stop redvector-test
        docker rm redvector-test
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ redis-cli not available, skipping connectivity test${NC}"
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}✓ Docker build: SUCCESS${NC}"
echo -e "${GREEN}✓ Container: RUNNING${NC}"
echo ""
echo "Container is running. To stop it:"
echo "  docker stop redvector-test"
echo "  docker rm redvector-test"
echo ""
echo "To view logs:"
echo "  docker logs redvector-test"
echo ""
echo "To connect with redis-cli:"
echo "  redis-cli -h 127.0.0.1 -p 6379"

