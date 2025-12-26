//! GPU configuration and device management

use serde::{Deserialize, Serialize};

/// GPU backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// Automatic backend selection (prefer CUDA on NVIDIA, else wgpu)
    Auto,

    /// wgpu backend (cross-platform: Vulkan, Metal, DX12)
    Wgpu,

    /// CUDA backend (NVIDIA only, maximum performance)
    Cuda,

    /// CPU fallback (no GPU acceleration)
    Cpu,
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::Auto
    }
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device name
    pub name: String,

    /// Backend type
    pub backend: GpuBackend,

    /// Device index (for multi-GPU)
    pub device_index: usize,

    /// Total memory in bytes
    pub total_memory: u64,

    /// Available memory in bytes (approximate)
    pub available_memory: u64,

    /// Compute capability (CUDA) or feature level
    pub compute_capability: String,

    /// Whether device supports FP16
    pub supports_fp16: bool,

    /// Whether device supports INT8
    pub supports_int8: bool,
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Backend to use
    pub backend: GpuBackend,

    /// Device index (for multi-GPU systems)
    pub device_index: usize,

    /// Maximum memory to use (bytes, 0 = unlimited)
    pub max_memory: u64,

    /// Batch size for GPU operations
    pub batch_size: usize,

    /// Enable async operations
    pub async_ops: bool,

    /// Memory pool size for buffer reuse
    pub memory_pool_size: usize,

    /// Preferred vector dimension alignment (for memory coalescing)
    pub dimension_alignment: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Auto,
            device_index: 0,
            max_memory: 0, // Unlimited
            batch_size: 1024,
            async_ops: true,
            memory_pool_size: 16,
            dimension_alignment: 32,
        }
    }
}

impl GpuConfig {
    /// Create config with wgpu backend
    pub fn wgpu() -> Self {
        Self {
            backend: GpuBackend::Wgpu,
            ..Default::default()
        }
    }

    /// Create config with CUDA backend
    pub fn cuda() -> Self {
        Self {
            backend: GpuBackend::Cuda,
            ..Default::default()
        }
    }

    /// Create config with CPU fallback
    pub fn cpu() -> Self {
        Self {
            backend: GpuBackend::Cpu,
            ..Default::default()
        }
    }

    /// Auto-detect best available backend
    pub fn auto_detect() -> super::Result<Self> {
        let mut config = Self::default();

        // Try CUDA first (best performance on NVIDIA)
        #[cfg(feature = "gpu-cuda")]
        {
            if super::cuda_backend::is_available() {
                config.backend = GpuBackend::Cuda;
                return Ok(config);
            }
        }

        // Fall back to wgpu (cross-platform)
        #[cfg(feature = "gpu-wgpu")]
        {
            if super::wgpu_backend::is_available() {
                config.backend = GpuBackend::Wgpu;
                return Ok(config);
            }
        }

        // CPU fallback
        config.backend = GpuBackend::Cpu;
        Ok(config)
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set device index
    pub fn with_device(mut self, index: usize) -> Self {
        self.device_index = index;
        self
    }

    /// Set max memory
    pub fn with_max_memory(mut self, bytes: u64) -> Self {
        self.max_memory = bytes;
        self
    }
}

