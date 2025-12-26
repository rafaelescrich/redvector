//! Doc Index: O(log N) or O(1) lookup of doc_id → (segment, offset, len)
//!
//! The doc index is a memory-mapped array of fixed-size entries, sorted by doc_id.
//! This allows binary search for O(log N) lookups with minimal memory overhead.

use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::rvf2::{Result, Rvf2Error, DOC_INDEX_MAGIC, VERSION};

/// Doc index header - 32 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DocIndexHeader {
    /// Magic: "RVDI"
    pub magic: [u8; 4],

    /// Version
    pub version: u32,

    /// Number of entries
    pub entry_count: u64,

    /// Flags (sorted, has_hash_index, etc.)
    pub flags: u32,

    /// Reserved for alignment
    pub reserved: [u8; 12],
}

/// Flags for doc index
pub mod flags {
    /// Entries are sorted by doc_id
    pub const SORTED: u32 = 1 << 0;
    /// Has auxiliary hash index
    pub const HAS_HASH: u32 = 1 << 1;
    /// Has tombstone entries
    pub const HAS_TOMBSTONES: u32 = 1 << 2;
}

/// Single doc entry - 24 bytes, fixed size for mmap
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocEntry {
    /// External document ID (Redis key hash or sequential ID)
    pub doc_id: u64,

    /// Segment containing this document
    pub segment_id: u32,

    /// Byte offset within segment
    pub offset: u32,

    /// Byte length of record
    pub len: u32,

    /// Flags: codec override, tombstone, etc.
    pub flags: u16,

    /// Reserved for future use
    pub reserved: u16,
}

impl DocEntry {
    /// Size of a doc entry in bytes
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Flag: entry is a tombstone (deleted)
    pub const FLAG_TOMBSTONE: u16 = 1 << 0;

    /// Flag: uses codec override
    pub const FLAG_CODEC_OVERRIDE: u16 = 1 << 1;

    /// Flag: variable patch count
    pub const FLAG_VARIABLE_PATCHES: u16 = 1 << 2;

    /// Create a new entry
    pub fn new(doc_id: u64, segment_id: u32, offset: u32, len: u32) -> Self {
        Self {
            doc_id,
            segment_id,
            offset,
            len,
            flags: 0,
            reserved: 0,
        }
    }

    /// Check if entry is a tombstone
    pub fn is_tombstone(&self) -> bool {
        self.flags & Self::FLAG_TOMBSTONE != 0
    }

    /// Mark as tombstone
    pub fn mark_tombstone(&mut self) {
        self.flags |= Self::FLAG_TOMBSTONE;
    }
}

/// Memory-mapped doc index for read-only access
pub struct DocIndex {
    mmap: Mmap,
    header: DocIndexHeader,
}

impl DocIndex {
    /// Open existing doc index as read-only mmap
    pub fn mmap<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Parse header
        if mmap.len() < std::mem::size_of::<DocIndexHeader>() {
            return Err(Rvf2Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Doc index too short",
            )));
        }

        let header: DocIndexHeader =
            unsafe { std::ptr::read_unaligned(mmap.as_ptr() as *const DocIndexHeader) };

        // Validate magic
        if header.magic != DOC_INDEX_MAGIC {
            return Err(Rvf2Error::InvalidMagic {
                expected: DOC_INDEX_MAGIC,
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

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.header.entry_count as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get entry by index (O(1))
    pub fn get_by_index(&self, index: usize) -> Option<DocEntry> {
        if index >= self.len() {
            return None;
        }

        let offset = std::mem::size_of::<DocIndexHeader>() + index * DocEntry::SIZE;
        if offset + DocEntry::SIZE > self.mmap.len() {
            return None;
        }

        let entry: DocEntry =
            unsafe { std::ptr::read_unaligned(self.mmap.as_ptr().add(offset) as *const DocEntry) };

        Some(entry)
    }

    /// Lookup by doc_id using binary search (O(log N))
    /// Requires entries to be sorted by doc_id
    pub fn get(&self, doc_id: u64) -> Result<DocEntry> {
        if self.header.flags & flags::SORTED == 0 {
            // Linear scan if not sorted (shouldn't happen in practice)
            return self.linear_scan(doc_id);
        }

        let mut left = 0;
        let mut right = self.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let entry = self.get_by_index(mid).ok_or(Rvf2Error::DocNotFound { doc_id })?;

            if entry.doc_id == doc_id {
                if entry.is_tombstone() {
                    return Err(Rvf2Error::DocNotFound { doc_id });
                }
                return Ok(entry);
            } else if entry.doc_id < doc_id {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        Err(Rvf2Error::DocNotFound { doc_id })
    }

    /// Linear scan (fallback for unsorted index)
    fn linear_scan(&self, doc_id: u64) -> Result<DocEntry> {
        for i in 0..self.len() {
            if let Some(entry) = self.get_by_index(i) {
                if entry.doc_id == doc_id && !entry.is_tombstone() {
                    return Ok(entry);
                }
            }
        }
        Err(Rvf2Error::DocNotFound { doc_id })
    }

    /// Iterate over all entries (for batch operations)
    pub fn iter(&self) -> DocIndexIter<'_> {
        DocIndexIter {
            index: self,
            pos: 0,
        }
    }
}

/// Iterator over doc index entries
pub struct DocIndexIter<'a> {
    index: &'a DocIndex,
    pos: usize,
}

impl<'a> Iterator for DocIndexIter<'a> {
    type Item = DocEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.index.len() {
            return None;
        }
        let entry = self.index.get_by_index(self.pos)?;
        self.pos += 1;
        Some(entry)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.index.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DocIndexIter<'_> {}

/// Mutable doc index builder for writes
pub struct DocIndexBuilder {
    entries: Vec<DocEntry>,
}

impl DocIndexBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Add an entry
    pub fn add(&mut self, entry: DocEntry) {
        self.entries.push(entry);
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build and write to file
    pub fn build<P: AsRef<Path>>(mut self, path: P) -> Result<()> {
        // Sort by doc_id for binary search
        self.entries.sort_by_key(|e| e.doc_id);

        let header = DocIndexHeader {
            magic: DOC_INDEX_MAGIC,
            version: VERSION,
            entry_count: self.entries.len() as u64,
            flags: flags::SORTED,
            reserved: [0; 12],
        };

        let total_size =
            std::mem::size_of::<DocIndexHeader>() + self.entries.len() * DocEntry::SIZE;

        // Create file with correct size
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(total_size as u64)?;

        // mmap for writing
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write header
        unsafe {
            std::ptr::write_unaligned(mmap.as_mut_ptr() as *mut DocIndexHeader, header);
        }

        // Write entries
        let entries_ptr = unsafe {
            mmap.as_mut_ptr()
                .add(std::mem::size_of::<DocIndexHeader>()) as *mut DocEntry
        };

        for (i, entry) in self.entries.iter().enumerate() {
            unsafe {
                std::ptr::write_unaligned(entries_ptr.add(i), *entry);
            }
        }

        // Flush to disk
        mmap.flush()?;

        Ok(())
    }
}

impl Default for DocIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_doc_entry_size() {
        // Ensure entry size is exactly 24 bytes
        assert_eq!(DocEntry::SIZE, 24);
    }

    #[test]
    fn test_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("doc_index.rvf2");

        // Build index
        let mut builder = DocIndexBuilder::new();
        builder.add(DocEntry::new(100, 0, 0, 1000));
        builder.add(DocEntry::new(200, 0, 1000, 1000));
        builder.add(DocEntry::new(50, 1, 0, 2000));
        builder.build(&path).unwrap();

        // Read index
        let index = DocIndex::mmap(&path).unwrap();
        assert_eq!(index.len(), 3);

        // Entries should be sorted (copy fields to avoid packed struct alignment issues)
        let e0 = index.get_by_index(0).unwrap();
        let e1 = index.get_by_index(1).unwrap();
        let e2 = index.get_by_index(2).unwrap();
        assert_eq!({ e0.doc_id }, 50);
        assert_eq!({ e1.doc_id }, 100);
        assert_eq!({ e2.doc_id }, 200);

        // Binary search
        let found = index.get(100).unwrap();
        assert_eq!({ found.segment_id }, 0);
        assert_eq!({ found.offset }, 0);
    }

    #[test]
    fn test_tombstone() {
        let mut entry = DocEntry::new(100, 0, 0, 1000);
        assert!(!entry.is_tombstone());

        entry.mark_tombstone();
        assert!(entry.is_tombstone());
    }
}

