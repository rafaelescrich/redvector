//! GPU memory management and buffer pools

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Memory pool for GPU buffer reuse
pub struct BufferPool<T> {
    /// Available buffers
    available: Mutex<VecDeque<T>>,

    /// Maximum buffers to keep
    max_buffers: usize,

    /// Total allocated buffers
    total_allocated: Mutex<usize>,
}

impl<T> BufferPool<T> {
    /// Create new buffer pool
    pub fn new(max_buffers: usize) -> Self {
        Self {
            available: Mutex::new(VecDeque::with_capacity(max_buffers)),
            max_buffers,
            total_allocated: Mutex::new(0),
        }
    }

    /// Get a buffer from the pool (or None if none available)
    pub fn get(&self) -> Option<T> {
        self.available.lock().unwrap().pop_front()
    }

    /// Return a buffer to the pool
    pub fn put(&self, buffer: T) {
        let mut available = self.available.lock().unwrap();
        if available.len() < self.max_buffers {
            available.push_back(buffer);
        }
        // If pool is full, buffer is dropped
    }

    /// Number of available buffers
    pub fn available_count(&self) -> usize {
        self.available.lock().unwrap().len()
    }

    /// Track new allocation
    pub fn track_allocation(&self) {
        *self.total_allocated.lock().unwrap() += 1;
    }

    /// Get total allocations
    pub fn total_allocations(&self) -> usize {
        *self.total_allocated.lock().unwrap()
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.available.lock().unwrap().clear();
    }
}

/// GPU memory statistics
#[derive(Debug, Clone, Default)]
pub struct GpuMemoryStats {
    /// Total memory used by vectors
    pub vector_memory: usize,

    /// Total memory used by index structures
    pub index_memory: usize,

    /// Total memory used by temporary buffers
    pub temp_memory: usize,

    /// Peak memory usage
    pub peak_memory: usize,

    /// Number of allocations
    pub allocation_count: usize,
}

impl GpuMemoryStats {
    /// Total memory in use
    pub fn total(&self) -> usize {
        self.vector_memory + self.index_memory + self.temp_memory
    }

    /// Update peak if current usage is higher
    pub fn update_peak(&mut self) {
        let current = self.total();
        if current > self.peak_memory {
            self.peak_memory = current;
        }
    }
}

/// Managed GPU buffer with automatic pool return
pub struct ManagedBuffer<T> {
    buffer: Option<T>,
    pool: Arc<BufferPool<T>>,
}

impl<T> ManagedBuffer<T> {
    /// Create from existing buffer and pool
    pub fn new(buffer: T, pool: Arc<BufferPool<T>>) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get reference to underlying buffer
    pub fn get(&self) -> &T {
        self.buffer.as_ref().unwrap()
    }

    /// Get mutable reference to underlying buffer
    pub fn get_mut(&mut self) -> &mut T {
        self.buffer.as_mut().unwrap()
    }
}

impl<T> Drop for ManagedBuffer<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.put(buffer);
        }
    }
}

impl<T> std::ops::Deref for ManagedBuffer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> std::ops::DerefMut for ManagedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// Vector data aligned for GPU memory
#[derive(Debug, Clone)]
pub struct AlignedVectors {
    /// Raw vector data
    data: Vec<f32>,

    /// Vector dimension
    dim: usize,

    /// Number of vectors
    count: usize,

    /// Stride between vectors (may include padding)
    stride: usize,
}

impl AlignedVectors {
    /// Create new aligned vector storage
    ///
    /// - `dim`: Vector dimension
    /// - `alignment`: Dimension alignment (e.g., 32 for GPU coalescing)
    pub fn new(dim: usize, alignment: usize) -> Self {
        let stride = if alignment > 0 {
            ((dim + alignment - 1) / alignment) * alignment
        } else {
            dim
        };

        Self {
            data: Vec::new(),
            dim,
            count: 0,
            stride,
        }
    }

    /// Create with capacity
    pub fn with_capacity(dim: usize, alignment: usize, capacity: usize) -> Self {
        let stride = if alignment > 0 {
            ((dim + alignment - 1) / alignment) * alignment
        } else {
            dim
        };

        Self {
            data: Vec::with_capacity(capacity * stride),
            dim,
            count: 0,
            stride,
        }
    }

    /// Add vectors (copies and aligns them)
    pub fn add(&mut self, vectors: &[f32]) {
        let n_vectors = vectors.len() / self.dim;

        for i in 0..n_vectors {
            let src = &vectors[i * self.dim..(i + 1) * self.dim];

            // Add vector data
            self.data.extend_from_slice(src);

            // Add padding if needed
            let padding = self.stride - self.dim;
            self.data.extend(std::iter::repeat(0.0).take(padding));

            self.count += 1;
        }
    }

    /// Get vector by index
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        if index >= self.count {
            return None;
        }
        let start = index * self.stride;
        Some(&self.data[start..start + self.dim])
    }

    /// Get raw data (with padding)
    pub fn raw_data(&self) -> &[f32] {
        &self.data
    }

    /// Get vector count
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get stride
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Clear all vectors
    pub fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool() {
        let pool: BufferPool<Vec<f32>> = BufferPool::new(4);

        // Put some buffers
        pool.put(vec![1.0, 2.0, 3.0]);
        pool.put(vec![4.0, 5.0, 6.0]);

        assert_eq!(pool.available_count(), 2);

        // Get a buffer
        let buf = pool.get().unwrap();
        assert_eq!(buf, vec![1.0, 2.0, 3.0]);
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_aligned_vectors() {
        let mut av = AlignedVectors::new(3, 4);

        // Add some vectors
        av.add(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(av.len(), 2);
        assert_eq!(av.stride(), 4); // Aligned to 4

        // Check first vector
        let v0 = av.get(0).unwrap();
        assert_eq!(v0, &[1.0, 2.0, 3.0]);

        // Raw data should have padding
        assert_eq!(av.raw_data().len(), 8); // 2 vectors × 4 stride
    }
}

