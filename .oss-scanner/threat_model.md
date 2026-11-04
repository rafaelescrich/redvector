# Threat model

## What this project does and where untrusted input enters
RedVector is an in-memory, Redis-compatible vector database written in Rust. It is a network
server; treat every byte arriving on its listeners as attacker-controlled:

- **RESP (Redis protocol), TCP 6379** — `parser/`, `networking/`, `command/`. Primary attack surface.
  Assume the attacker can connect and send arbitrary commands, including before `AUTH` when a
  password is configured.
- **REST API, TCP 8888** (`--features api-server`) — `src/api.rs`, `src/qdrant_api.rs` (Qdrant-compatible
  JSON API, including SQL parsing via `sqlparser`).
- **gRPC API, TCP 50051** (`--features api-server`) — `proto/`, `src/api.rs`.
- **Persistence files** — RDB/AOF loading in `persistence/` and `database/rdbutil/`. Treat files on
  disk as untrusted (e.g. an RDB copied from elsewhere or replicated from a malicious master).
- **Configuration** (`config/`, `redvector.conf`) is trusted operator input; out of scope unless a
  client can change it remotely (e.g. `CONFIG SET`).

## Components that matter most / least
- Most: RESP parser and networking, command dispatch and argument handling, authentication/ACL
  checks, vector index code (`database/`, HNSW, `redisearch-platform-core/`) reachable via `FT.*`,
  REST/gRPC handlers, RDB/AOF loaders.
- Less: `benchmarks/`, `deploy/`, `docs/`, Dockerfiles, CI workflows — out of scope.
- `redisearch-platform-core` GPU / RVF2 / S3 modules are feature-gated and not enabled by `full`;
  low priority.

## How to exercise it
- Build: `cargo build --features full` (already done in the image; binary at `target/debug/redvector`
  and `target/release/redvector`).
- Run: `target/debug/redvector` listens on 6379 (and 8888/50051 with `full`). Talk to it with
  `redis-cli` (installed) or raw TCP.
- Unit tests: `cargo test --features full $TEST_CRATES` (env var lists the member crates; the root
  manifest has no `[workspace]`, so plain `cargo test` only runs the binary's tests). `tests/` holds the Tcl Redis-compat suite
  (`tclsh` installed).

## How you rate severity
- **Critical**: remote code execution, or memory corruption in `unsafe` code reachable from the
  network; authentication bypass giving full data access.
- **High**: pre-auth or low-privilege remote crash (panic, abort, OOM, stack overflow) from a
  single request or small number of requests; reading/writing keys or vectors the client should
  not access; path traversal or arbitrary file write via commands or APIs.
- **Medium**: crash or resource exhaustion that needs an authenticated client and sustained or
  large traffic; unbounded allocation driven by a size field the client controls; malicious
  RDB/AOF files causing a crash on load.
- **Low**: information leaks of non-sensitive internals, incorrect error handling without
  security impact.
- A `panic!`/`unwrap()` that kills the whole server process from a client request is a DoS and
  should be reported (rate as above), not dismissed as "just a panic".

## Reports
- Include a minimal reproducer (a `redis-cli` command sequence, raw RESP bytes, `curl`
  request, or crafted file) plus the stack trace.
- A minimal patch that fixes the root cause is preferred over a large refactor.
- Deduplicate by root cause: many commands hitting the same faulty helper are one report.

## Anything to leave alone
- Differences from upstream Redis behaviour that have no security impact.
- Missing features or incomplete persistence documented in README "Current limitations".
- Vulnerabilities in third-party crates unless RedVector uses them in a way that is itself the bug.
- Lack of TLS or default no-password configuration (documented operator choice).
