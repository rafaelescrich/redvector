# Build environment for Anthropic OSS Scanner (https://github.com/anthropics/oss-scanner).
# Built with the repository root as context; network is removed after this build,
# so every dependency is fetched here.
FROM rust:1.91-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates pkg-config libssl-dev protobuf-compiler clang git tcl redis-tools \
    && rm -rf /var/lib/apt/lists/*

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src

# Resolve and download all crates while network is still available.
RUN cargo generate-lockfile && cargo fetch

ENV CARGO_NET_OFFLINE=true \
    RUSTFLAGS="-C force-frame-pointers=yes" \
    CARGO_PROFILE_RELEASE_DEBUG=true \
    RUST_BACKTRACE=1

# Debug build (fast to rebuild, debug assertions + overflow checks on) and a
# release build with debug info, both with every feature the server ships with.
RUN cargo build --all --features full \
    && cargo build --release --features full

# The root manifest has no [workspace]; unit tests live in the member crates.
ENV TEST_CRATES="-p redvector -p parser -p response -p database -p command -p networking -p persistence -p config -p util -p compat -p logger"

# Compile all test targets so agents can run them offline.
RUN cargo test --features full $TEST_CRATES --no-run

# Run the cheap tests (test_large_dataset takes ~15 min in debug), but do not fail the image on test failures.
RUN cargo test --features full $TEST_CRATES -- --skip test_large_dataset || echo "WARNING: cargo test reported failures"
