//! Segment: Immutable blob containing document records
//!
//! Each segment is a contiguous file containing:
//! 1. SegmentHeader (64 bytes)
//! 2. DocRecord[] (variable size per document)
//! 3. Optional footer index (for intra-segment random access)

use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::rvf2::manifest::CodecType;
use crate::rvf2::{Result, Rvf2Error, MAGIC, VERSION};

/// Segment file header - 64 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct SegmentHeader {
    /// Magic: "RVF2"
    pub magic: [u8; 4],

    /// Format version
    pub version: u32,

    /// Segment ID
    pub segment_id: u32,

    /// Default codec for this segment
    pub codec: u8,

    /// Flags
    pub flags: u8,

    /// Header length in bytes
    pub header_len: u16,

    /// Number of documents in segment
    pub doc_count: u32,

    /// Offset to optional footer index
    pub index_offset: u32,

    /// Length of footer index
    pub index_len: u32,

    /// CRC32C of header + payload (excluding this field)
    pub crc32c: u32,

    /// Dimension
    pub dims: u16,

    /// Expected patches per doc
    pub patches_per_doc: u16,

    /// Reserved for alignment to 64 bytes
    pub reserved: [u8; 24],
}

impl SegmentHeader {
    /// Size of header in bytes
    pub const SIZE: usize = 64;

    /// Create new header
    pub fn new(segment_id: u32, codec: CodecType, dims: u16, patches_per_doc: u16) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            segment_id,
            codec: codec as u8,
            flags: 0,
            header_len: Self::SIZE as u16,
            doc_count: 0,
            index_offset: 0,
            index_len: 0,
            crc32c: 0,
            dims,
            patches_per_doc,
            reserved: [0; 24],
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        let mut offset = 0;

        // magic: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.magic);
        offset += 4;

        // version: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.version.to_le_bytes());
        offset += 4;

        // segment_id: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.segment_id.to_le_bytes());
        offset += 4;

        // codec: 1 byte
        bytes[offset] = self.codec;
        offset += 1;

        // flags: 1 byte
        bytes[offset] = self.flags;
        offset += 1;

        // header_len: 2 bytes
        bytes[offset..offset + 2].copy_from_slice(&self.header_len.to_le_bytes());
        offset += 2;

        // doc_count: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.doc_count.to_le_bytes());
        offset += 4;

        // index_offset: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.index_offset.to_le_bytes());
        offset += 4;

        // index_len: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.index_len.to_le_bytes());
        offset += 4;

        // crc32c: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.crc32c.to_le_bytes());
        offset += 4;

        // dims: 2 bytes
        bytes[offset..offset + 2].copy_from_slice(&self.dims.to_le_bytes());
        offset += 2;

        // patches_per_doc: 2 bytes
        bytes[offset..offset + 2].copy_from_slice(&self.patches_per_doc.to_le_bytes());
        offset += 2;

        // reserved: 24 bytes
        bytes[offset..offset + 24].copy_from_slice(&self.reserved);

        bytes
    }
}

/// Segment header flags
pub mod segment_flags {
    /// Has footer index for random access
    pub const HAS_FOOTER_INDEX: u8 = 1 << 0;
    /// Segment is compressed (LZ4)
    pub const COMPRESSED: u8 = 1 << 1;
}

/// Per-document record header - 32 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DocRecordHeader {
    /// Document ID
    pub doc_id: u64,

    /// Number of patches
    pub n_patches: u16,

    /// Dimension
    pub dims: u16,

    /// Codec for pooled vector
    pub pooled_codec: u8,

    /// Codec for patch vectors
    pub patches_codec: u8,

    /// Reserved
    pub reserved: u16,

    /// Length of pooled vector blob
    pub pooled_len: u32,

    /// Length of patch codes blob
    pub patches_len: u32,

    /// Length of scales/aux data
    pub scales_len: u32,

    /// CRC32C of this record
    pub record_crc32c: u32,
}

impl DocRecordHeader {
    /// Size of record header in bytes
    pub const SIZE: usize = 32;

    /// Total payload size (after header)
    pub fn payload_size(&self) -> usize {
        self.pooled_len as usize + self.patches_len as usize + self.scales_len as usize
    }

    /// Total record size (header + payload)
    pub fn total_size(&self) -> usize {
        Self::SIZE + self.payload_size()
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        let mut offset = 0;

        // doc_id: 8 bytes
        bytes[offset..offset + 8].copy_from_slice(&self.doc_id.to_le_bytes());
        offset += 8;

        // n_patches: 2 bytes
        bytes[offset..offset + 2].copy_from_slice(&self.n_patches.to_le_bytes());
        offset += 2;

        // dims: 2 bytes
        bytes[offset..offset + 2].copy_from_slice(&self.dims.to_le_bytes());
        offset += 2;

        // pooled_codec: 1 byte
        bytes[offset] = self.pooled_codec;
        offset += 1;

        // patches_codec: 1 byte
        bytes[offset] = self.patches_codec;
        offset += 1;

        // reserved: 2 bytes
        bytes[offset..offset + 2].copy_from_slice(&self.reserved.to_le_bytes());
        offset += 2;

        // pooled_len: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.pooled_len.to_le_bytes());
        offset += 4;

        // patches_len: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.patches_len.to_le_bytes());
        offset += 4;

        // scales_len: 4 bytes
        bytes[offset..offset + 4].copy_from_slice(&self.scales_len.to_le_bytes());
        offset += 4;

        // record_crc32c: 4 bytes (at offset 28)
        bytes[offset..offset + 4].copy_from_slice(&self.record_crc32c.to_le_bytes());

        bytes
    }
}

/// Parsed document record
#[derive(Debug)]
pub struct DocRecord<'a> {
    /// Record header
    pub header: DocRecordHeader,

    /// Pooled vector bytes (encoded)
    pub pooled: &'a [u8],

    /// Patch codes bytes (encoded matrix)
    pub patches: &'a [u8],

    /// Scales/aux data (for SQ8, PQ, etc.)
    pub scales: &'a [u8],
}

/// Read-only memory-mapped segment
pub struct Segment {
    mmap: Mmap,
    header: SegmentHeader,
}

impl Segment {
    /// Open existing segment as read-only mmap
    pub fn mmap<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Parse header
        if mmap.len() < SegmentHeader::SIZE {
            return Err(Rvf2Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Segment too short",
            )));
        }

        let header: SegmentHeader =
            unsafe { std::ptr::read_unaligned(mmap.as_ptr() as *const SegmentHeader) };

        // Validate magic
        if header.magic != MAGIC {
            return Err(Rvf2Error::InvalidMagic {
                expected: MAGIC,
                got: header.magic,
            });
        }

        // Validate version
        if header.version != VERSION {
            return Err(Rvf2Error::VersionMismatch {
                expected: VERSION,
                got: header.version,
            });
        }

        Ok(Self { mmap, header })
    }

    /// Get segment header
    pub fn header(&self) -> &SegmentHeader {
        &self.header
    }

    /// Get segment ID
    pub fn segment_id(&self) -> u32 {
        self.header.segment_id
    }

    /// Get document count
    pub fn doc_count(&self) -> u32 {
        self.header.doc_count
    }

    /// Get raw bytes at offset (for range-based access)
    pub fn get_bytes(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if offset + len > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[offset..offset + len])
    }

    /// Get document record at offset
    pub fn get_record(&self, offset: u32, len: u32) -> Result<DocRecord<'_>> {
        let offset = offset as usize;
        let len = len as usize;

        if offset + len > self.mmap.len() || len < DocRecordHeader::SIZE {
            return Err(Rvf2Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid record bounds",
            )));
        }

        // Parse header
        let header: DocRecordHeader = unsafe {
            std::ptr::read_unaligned(self.mmap.as_ptr().add(offset) as *const DocRecordHeader)
        };

        // Validate sizes
        if header.total_size() != len {
            return Err(Rvf2Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Record size mismatch: header says {}, entry says {}",
                    header.total_size(),
                    len
                ),
            )));
        }

        // Extract payload slices
        let payload_start = offset + DocRecordHeader::SIZE;
        let pooled_end = payload_start + header.pooled_len as usize;
        let patches_end = pooled_end + header.patches_len as usize;
        let scales_end = patches_end + header.scales_len as usize;

        Ok(DocRecord {
            header,
            pooled: &self.mmap[payload_start..pooled_end],
            patches: &self.mmap[pooled_end..patches_end],
            scales: &self.mmap[patches_end..scales_end],
        })
    }

    /// Get raw mmap slice for zero-copy GPU upload
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
}

/// Segment builder for writing new segments
pub struct SegmentBuilder {
    header: SegmentHeader,
    data: Vec<u8>,
    doc_offsets: Vec<(u64, u32, u32)>, // (doc_id, offset, len)
}

impl SegmentBuilder {
    /// Create new segment builder
    pub fn new(segment_id: u32, codec: CodecType, dims: u16, patches_per_doc: u16) -> Self {
        let header = SegmentHeader::new(segment_id, codec, dims, patches_per_doc);

        // Pre-allocate space for header
        let mut data = vec![0u8; SegmentHeader::SIZE];

        Self {
            header,
            data,
            doc_offsets: Vec::new(),
        }
    }

    /// Add a document record
    pub fn add_record(
        &mut self,
        doc_id: u64,
        n_patches: u16,
        pooled_codec: CodecType,
        patches_codec: CodecType,
        pooled: &[u8],
        patches: &[u8],
        scales: &[u8],
    ) -> Result<()> {
        let offset = self.data.len() as u32;

        let record_header = DocRecordHeader {
            doc_id,
            n_patches,
            dims: self.header.dims,
            pooled_codec: pooled_codec as u8,
            patches_codec: patches_codec as u8,
            reserved: 0,
            pooled_len: pooled.len() as u32,
            patches_len: patches.len() as u32,
            scales_len: scales.len() as u32,
            record_crc32c: 0, // TODO: compute CRC
        };

        let len = record_header.total_size() as u32;

        // Write header
        let header_bytes = record_header.to_bytes();
        self.data.extend_from_slice(&header_bytes);

        // Write payloads
        self.data.extend_from_slice(pooled);
        self.data.extend_from_slice(patches);
        self.data.extend_from_slice(scales);

        // Track offset
        self.doc_offsets.push((doc_id, offset, len));
        self.header.doc_count += 1;

        Ok(())
    }

    /// Get current size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get doc offsets for building doc index
    pub fn doc_offsets(&self) -> &[(u64, u32, u32)] {
        &self.doc_offsets
    }

    /// Build and write segment to file
    pub fn build<P: AsRef<Path>>(mut self, path: P) -> Result<()> {
        // Update header
        self.header.crc32c = crc32fast::hash(&self.data[SegmentHeader::SIZE..]);

        // Serialize header to bytes
        let header_bytes = self.header.to_bytes();
        self.data[..SegmentHeader::SIZE].copy_from_slice(&header_bytes);

        // Write file
        std::fs::write(path, &self.data)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_header_size() {
        assert_eq!(SegmentHeader::SIZE, 64);
        assert_eq!(DocRecordHeader::SIZE, 32);
    }

    #[test]
    fn test_segment_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg_000000.rvf2");

        // Build segment
        let mut builder = SegmentBuilder::new(0, CodecType::Sq8, 128, 1024);

        let pooled = vec![0u8; 128]; // FP32 pooled
        let patches = vec![0i8; 1024 * 128]; // SQ8 patches
        let patches_u8: Vec<u8> = patches.iter().map(|&x| x as u8).collect();
        let scales = vec![0u8; 32 * 2]; // f16 scales for 32 blocks

        builder
            .add_record(
                100,
                1024,
                CodecType::Fp32,
                CodecType::Sq8,
                &pooled,
                &patches_u8,
                &scales,
            )
            .unwrap();

        let offsets = builder.doc_offsets().to_vec();
        builder.build(&path).unwrap();

        // Read segment
        let segment = Segment::mmap(&path).unwrap();
        assert_eq!(segment.doc_count(), 1);

        // Read record (copy fields to avoid packed struct alignment issues)
        let (doc_id, offset, len) = offsets[0];
        let record = segment.get_record(offset, len).unwrap();
        assert_eq!({ record.header.doc_id }, doc_id);
        assert_eq!({ record.header.n_patches }, 1024);
        assert_eq!(record.patches.len(), 1024 * 128);
    }
}

