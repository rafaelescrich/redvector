# Multi-stage Dockerfile for RedVector - Redis-compatible Vector Database
# Stage 1: Build
# 1.85+ required: transitive crates (e.g. getrandom 0.4) use edition2024
FROM rust:1.85-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY proto/ ./proto/
COPY command/Cargo.toml ./command/
COPY config/Cargo.toml ./config/
COPY database/Cargo.toml ./database/
COPY database/rdbutil/Cargo.toml ./database/rdbutil/
COPY networking/Cargo.toml ./networking/
COPY parser/Cargo.toml ./parser/
COPY response/Cargo.toml ./response/
COPY persistence/Cargo.toml ./persistence/
COPY logger/Cargo.toml ./logger/
COPY util/Cargo.toml ./util/
COPY compat/Cargo.toml ./compat/

# Create dummy source files to build dependencies
RUN mkdir -p src command/src config/src database/src database/rdbutil/src \
    networking/src parser/src response/src persistence/src logger/src util/src compat/src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > command/src/lib.rs && \
    echo "pub fn dummy() {}" > config/src/lib.rs && \
    echo "pub fn dummy() {}" > database/src/lib.rs && \
    echo "pub fn dummy() {}" > database/rdbutil/src/lib.rs && \
    echo "pub fn dummy() {}" > networking/src/lib.rs && \
    echo "pub fn dummy() {}" > parser/src/lib.rs && \
    echo "pub fn dummy() {}" > response/src/lib.rs && \
    echo "pub fn dummy() {}" > persistence/src/lib.rs && \
    echo "pub fn dummy() {}" > logger/src/lib.rs && \
    echo "pub fn dummy() {}" > util/src/lib.rs && \
    echo "pub fn dummy() {}" > compat/src/lib.rs

# Build dependencies (cached layer)
RUN cargo build --release --features "vector-search,hnsw-backend" 2>&1 || true

# Copy actual source code
COPY . .

# Touch main.rs to force rebuild
RUN touch src/main.rs

# Build with all features
RUN cargo build --release --features "full" && \
    strip /build/target/release/redvector

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    groupadd -r -g 1000 redvector && \
    useradd -r -u 1000 -g redvector redvector

COPY --from=builder /build/target/release/redvector /usr/local/bin/redvector

RUN mkdir -p /data && chown -R redvector:redvector /data

WORKDIR /data

# Redis protocol + REST API + gRPC API
EXPOSE 6379 8888 50051

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8888/health || exit 1

USER redvector

ENTRYPOINT ["/usr/local/bin/redvector"]
CMD []
