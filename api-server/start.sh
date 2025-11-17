#!/bin/bash

echo "🚀 Starting RedVector API Server..."
echo ""
echo "Make sure RedVector (rsedis) is running on localhost:6379"
echo "If not, start it with: cd ../rsedis && cargo run --release"
echo ""
echo "Starting API server on http://localhost:8080"
echo ""

cargo run --release

