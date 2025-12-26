//! Manifest file for RVF2 indexes
//!
//! The manifest is a small file that is atomically replaced on updates.
//! It contains index metadata and a list of all segments.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::rvf2::{Result, Rvf2Error, MAGIC, VERSION};

/// Manifest file - atomically replaced on updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Magic bytes (verified on load)
    #[serde(skip)]
    pub magic: [u8; 4],

    /// Format version
    pub version: u32,

    /// Vector dimension (e.g., 128 for ColPali)
    pub dims: u16,

    /// Default codec for patches
    pub codec: CodecType,

    /// Expected patches per doc (can vary per doc)
    pub patches_per_doc: u16,

    /// Target segment size in bytes
    pub segment_target_bytes: u64,

    /// Block size for SQ8 quantization
    pub block_size: u16,

    /// List of segments
    pub segments: Vec<SegmentMeta>,

    /// Total document count across all segments
    pub total_docs: u64,

    /// Creation timestamp (Unix epoch seconds)
    pub created_at_unix: u64,

    /// Last update timestamp
    pub updated_at_unix: u64,
}

/// Segment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    /// Unique segment identifier
    pub segment_id: u32,

    /// Filename (e.g., "seg_000123.rvf2")
    pub file_name: String,

    /// Total bytes
    pub bytes: u64,

    /// Number of documents in segment
    pub doc_count: u32,

    /// CRC32C checksum
    pub crc32c: u32,

    /// Min doc_id in segment (for range queries)
    pub min_doc_id: u64,

    /// Max doc_id in segment
    pub max_doc_id: u64,

    /// Codec used (can differ from manifest default)
    pub codec: CodecType,

    /// Whether segment is sealed (immutable)
    pub sealed: bool,
}

/// Codec type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CodecType {
    /// Full precision float32 (no compression)
    Fp32 = 0,

    /// Int8 scalar quantization (4x compression)
    Sq8 = 1,

    /// Float16 (2x compression, high accuracy)
    Fp16 = 2,

    /// Product quantization m=16 (32x compression)
    Pq16 = 3,
}

impl Default for CodecType {
    fn default() -> Self {
        Self::Sq8
    }
}

impl Manifest {
    /// Create a new manifest with default settings
    pub fn new(dims: u16, patches_per_doc: u16) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            magic: MAGIC,
            version: VERSION,
            dims,
            codec: CodecType::Sq8,
            patches_per_doc,
            segment_target_bytes: super::DEFAULT_SEGMENT_TARGET_BYTES,
            block_size: super::DEFAULT_BLOCK_SIZE as u16,
            segments: Vec::new(),
            total_docs: 0,
            created_at_unix: now,
            updated_at_unix: now,
        }
    }

    /// Load manifest from file (MessagePack format)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Parse manifest from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // Check magic (first 4 bytes before MessagePack data)
        if bytes.len() < 4 {
            return Err(Rvf2Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Manifest too short",
            )));
        }

        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != MAGIC {
            return Err(Rvf2Error::InvalidMagic {
                expected: MAGIC,
                got: magic,
            });
        }

        // Deserialize rest as MessagePack
        let manifest: Manifest = rmp_serde::from_slice(&bytes[4..])
            .map_err(|e| Rvf2Error::Serialization(e.to_string()))?;

        // Version check
        if manifest.version != VERSION {
            return Err(Rvf2Error::VersionMismatch {
                expected: VERSION,
                got: manifest.version,
            });
        }

        Ok(manifest)
    }

    /// Save manifest to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_bytes()?;

        // Write atomically: write to temp file, then rename
        let path = path.as_ref();
        let temp_path = path.with_extension("rvf2.tmp");

        std::fs::write(&temp_path, &bytes)?;
        std::fs::rename(&temp_path, path)?;

        Ok(())
    }

    /// Serialize manifest to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(4096);

        // Write magic
        bytes.extend_from_slice(&MAGIC);

        // Write MessagePack data
        rmp_serde::encode::write(&mut bytes, self)
            .map_err(|e| Rvf2Error::Serialization(e.to_string()))?;

        Ok(bytes)
    }

    /// Add a new segment
    pub fn add_segment(&mut self, meta: SegmentMeta) {
        self.total_docs += meta.doc_count as u64;
        self.segments.push(meta);
        self.updated_at_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Get next segment ID
    pub fn next_segment_id(&self) -> u32 {
        self.segments
            .iter()
            .map(|s| s.segment_id)
            .max()
            .map(|id| id + 1)
            .unwrap_or(0)
    }

    /// Find segment containing a doc_id (binary search)
    pub fn find_segment(&self, doc_id: u64) -> Option<&SegmentMeta> {
        self.segments
            .iter()
            .find(|s| doc_id >= s.min_doc_id && doc_id <= s.max_doc_id)
    }
}

impl SegmentMeta {
    /// Create metadata for a new segment
    pub fn new(segment_id: u32, codec: CodecType) -> Self {
        Self {
            segment_id,
            file_name: format!("seg_{:06}.rvf2", segment_id),
            bytes: 0,
            doc_count: 0,
            crc32c: 0,
            min_doc_id: u64::MAX,
            max_doc_id: 0,
            codec,
            sealed: false,
        }
    }

    /// Update doc_id range
    pub fn update_doc_range(&mut self, doc_id: u64) {
        self.min_doc_id = self.min_doc_id.min(doc_id);
        self.max_doc_id = self.max_doc_id.max(doc_id);
        self.doc_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_roundtrip() {
        let mut manifest = Manifest::new(128, 1024);
        manifest.add_segment(SegmentMeta::new(0, CodecType::Sq8));

        let bytes = manifest.to_bytes().unwrap();
        let loaded = Manifest::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.dims, 128);
        assert_eq!(loaded.patches_per_doc, 1024);
        assert_eq!(loaded.segments.len(), 1);
    }

    #[test]
    fn test_codec_type_default() {
        assert_eq!(CodecType::default(), CodecType::Sq8);
    }
}

