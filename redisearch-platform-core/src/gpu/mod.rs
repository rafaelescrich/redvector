//! # GPU Acceleration for Vector Search
//!
//! This module provides GPU-accelerated vector operations for RedVector,
//! supporting multiple backends:
//!
//! - **wgpu**: Cross-platform (Vulkan, Metal, DX12) - default
//! - **CUDA**: NVIDIA-specific maximum performance - optional
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     GpuVectorIndex Trait                        │
//! │           add() | search() | batch_search()                     │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!        ┌─────────────────────┼─────────────────────┐
//!        ▼                     ▼                     ▼
//!   ┌─────────┐          ┌─────────┐          ┌─────────┐
//!   │  wgpu   │          │ cudarc  │          │   CPU   │
//!   │ Backend │          │ Backend │          │ Fallback│
//!   └─────────┘          └─────────┘          └─────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use redisearch_platform_core::gpu::{GpuConfig, GpuFlatIndex};
//!
//! // Auto-detect best GPU backend
//! let config = GpuConfig::auto_detect()?;
//! let mut index = GpuFlatIndex::new(768, DistanceMetric::Cosine, config)?;
//!
//! // Add vectors
//! index.add(&[1, 2, 3], &vectors)?;
//!
//! // Search
//! let results = index.search(&query, 10)?;
//! ```

pub mod config;
pub mod traits;
pub mod distance;
pub mod flat_index;
pub mod memory;

#[cfg(feature = "gpu-wgpu")]
pub mod wgpu_backend;

#[cfg(feature = "gpu-cuda")]
pub mod cuda_backend;

pub mod ivf;
pub mod quantization;

// Re-exports
pub use config::{GpuBackend, GpuConfig, GpuDeviceInfo};
pub use traits::{GpuVectorIndex, SearchResult};
pub use distance::DistanceMetric;
pub use flat_index::GpuFlatIndex;
pub use ivf::{IvfConfig, GpuIvfIndex};
pub use quantization::{QuantizerType, Sq8Quantizer};

/// GPU module error type
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("No GPU device available")]
    NoDevice,

    #[error("GPU backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),

    #[error("GPU memory allocation failed: {0}")]
    MemoryAllocationFailed(String),

    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),

    #[error("GPU compute error: {0}")]
    ComputeError(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Index not trained")]
    NotTrained,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for GPU operations
pub type Result<T> = std::result::Result<T, GpuError>;

/// Check if any GPU backend is available
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "gpu-wgpu")]
    {
        if wgpu_backend::is_available() {
            return true;
        }
    }

    #[cfg(feature = "gpu-cuda")]
    {
        if cuda_backend::is_available() {
            return true;
        }
    }

    false
}

/// Get information about available GPU devices
pub fn enumerate_devices() -> Vec<GpuDeviceInfo> {
    let mut devices = Vec::new();

    #[cfg(feature = "gpu-wgpu")]
    {
        devices.extend(wgpu_backend::enumerate_devices());
    }

    #[cfg(feature = "gpu-cuda")]
    {
        devices.extend(cuda_backend::enumerate_devices());
    }

    devices
}

