//! # RVF v2: Multi-Vector Storage Format
//!
//! This module implements the RVF v2 (Rust Vector Filesystem v2) storage format
//! for efficient multi-vector retrieval (ColPali/ColBERT-style).
//!
//! ## Architecture
//!
//! ```text
//! Query → embed() → [pooled, tokens]
//!                        │
//!          ┌─────────────┴─────────────┐
//!          │                           │
//!          ▼                           ▼
//!   Stage A: ANN Search         Stage B: MaxSim Rerank
//!   (HNSW on pooled)            (fetch patches → score)
//!          │                           │
//!          └─────────────┬─────────────┘
//!                        ▼
//!                   Top-K results
//! ```
//!
//! ## Key Components
//!
//! - [`Manifest`]: Index metadata (segments, codecs, dimensions)
//! - [`DocIndex`]: O(log N) lookup of doc_id → (segment, offset, len)
//! - [`Segment`]: Immutable blob containing document records
//! - [`Codec`]: SQ8, FP16, PQ16 encoding/decoding
//! - [`SegmentStore`]: Tiered storage (mmap → cache → object store)
//! - [`MaxSim`]: SIMD-optimized late-interaction scoring
//!
//! ## Example
//!
//! ```rust,ignore
//! use rvf2::{Manifest, DocIndex, Segment, Sq8Codec};
//!
//! // Open existing index
//! let manifest = Manifest::load("index/manifest.rvf2")?;
//! let doc_index = DocIndex::mmap("index/doc_index.rvf2")?;
//!
//! // Lookup document
//! let entry = doc_index.get(doc_id)?;
//! let segment = Segment::mmap(&format!("index/seg/seg_{:06}.rvf2", entry.segment_id))?;
//! let record = segment.get_record(entry.offset, entry.len)?;
//!
//! // Decode and compute MaxSim
//! let patches = Sq8Codec::decode(&record.patch_codes, &record.scales)?;
//! let score = maxsim::compute(&query_tokens, &patches)?;
//! ```

pub mod codec;
pub mod doc_index;
pub mod manifest;
pub mod maxsim;
pub mod object_store;
pub mod prefetch;
pub mod segment;
pub mod tiered_store;

// Re-exports
pub use codec::{Codec, CodecType, Sq8Codec};
pub use doc_index::{DocEntry, DocIndex, DocIndexBuilder};
pub use manifest::{Manifest, SegmentMeta};
pub use maxsim::{MaxSimScorer, SimdMaxSimScorer};
pub use segment::{DocRecord, Segment, SegmentHeader, SegmentBuilder};
pub use tiered_store::{TieredStore, TieredStoreBuilder, TieredStoreStats};

/// RVF2 magic bytes
pub const MAGIC: [u8; 4] = *b"RVF2";

/// Doc index magic bytes
pub const DOC_INDEX_MAGIC: [u8; 4] = *b"RVDI";

/// Current format version
pub const VERSION: u32 = 2;

/// Default block size for SQ8 quantization
pub const DEFAULT_BLOCK_SIZE: usize = 32;

/// Default segment target size (512 MB)
pub const DEFAULT_SEGMENT_TARGET_BYTES: u64 = 512 * 1024 * 1024;

/// Result type for RVF2 operations
pub type Result<T> = std::result::Result<T, Rvf2Error>;

/// Errors that can occur during RVF2 operations
#[derive(Debug, thiserror::Error)]
pub enum Rvf2Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid magic bytes: expected {expected:?}, got {got:?}")]
    InvalidMagic { expected: [u8; 4], got: [u8; 4] },

    #[error("Version mismatch: expected {expected}, got {got}")]
    VersionMismatch { expected: u32, got: u32 },

    #[error("CRC32 checksum mismatch: expected {expected}, got {got}")]
    ChecksumMismatch { expected: u32, got: u32 },

    #[error("Document not found: {doc_id}")]
    DocNotFound { doc_id: u64 },

    #[error("Segment not found: {segment_id}")]
    SegmentNotFound { segment_id: u32 },

    #[error("Codec error: {0}")]
    Codec(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Object store error: {0}")]
    ObjectStore(String),

    #[error("Index full: max docs {max}")]
    IndexFull { max: u64 },
}

