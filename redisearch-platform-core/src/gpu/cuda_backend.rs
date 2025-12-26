//! CUDA backend for NVIDIA GPU acceleration
//!
//! Provides maximum performance on NVIDIA GPUs using cudarc.

use super::{GpuBackend, GpuDeviceInfo, GpuError, Result};

/// Check if CUDA backend is available
pub fn is_available() -> bool {
    #[cfg(feature = "gpu-cuda")]
    {
        match cudarc::driver::CudaDevice::new(0) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    #[cfg(not(feature = "gpu-cuda"))]
    {
        false
    }
}

/// Enumerate available CUDA devices
pub fn enumerate_devices() -> Vec<GpuDeviceInfo> {
    #[cfg(feature = "gpu-cuda")]
    {
        let mut devices = Vec::new();
        let mut device_index = 0;

        while let Ok(device) = cudarc::driver::CudaDevice::new(device_index) {
            // Get device properties
            let name = format!("CUDA Device {}", device_index);

            devices.push(GpuDeviceInfo {
                name,
                backend: GpuBackend::Cuda,
                device_index,
                total_memory: 0, // Would need cudarc API to get this
                available_memory: 0,
                compute_capability: String::new(),
                supports_fp16: true,
                supports_int8: true,
            });

            device_index += 1;

            // Don't enumerate too many
            if device_index > 8 {
                break;
            }
        }

        devices
    }

    #[cfg(not(feature = "gpu-cuda"))]
    {
        Vec::new()
    }
}

/// CUDA context wrapper
#[cfg(feature = "gpu-cuda")]
pub struct CudaContext {
    pub device: std::sync::Arc<cudarc::driver::CudaDevice>,
}

#[cfg(feature = "gpu-cuda")]
impl CudaContext {
    /// Create new CUDA context
    pub fn new(device_index: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(device_index)
            .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;

        Ok(Self { device })
    }

    /// Allocate device memory
    pub fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.device
            .alloc_zeros(len)
            .map_err(|e| GpuError::MemoryAllocationFailed(e.to_string()))
    }

    /// Copy data to device
    pub fn htod<T: cudarc::driver::DeviceRepr>(
        &self,
        data: &[T],
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.device
            .htod_sync_copy(data)
            .map_err(|e| GpuError::ComputeError(e.to_string()))
    }

    /// Copy data from device
    pub fn dtoh<T: cudarc::driver::DeviceRepr>(
        &self,
        slice: &cudarc::driver::CudaSlice<T>,
    ) -> Result<Vec<T>> {
        self.device
            .dtoh_sync_copy(slice)
            .map_err(|e| GpuError::ComputeError(e.to_string()))
    }

    /// Load PTX module
    pub fn load_ptx(
        &self,
        ptx: &str,
        module_name: &str,
        func_names: &[&str],
    ) -> Result<()> {
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx);
        self.device
            .load_ptx(ptx, module_name, func_names)
            .map_err(|e| GpuError::ShaderCompilationFailed(e.to_string()))
    }
}

/// CUDA kernels for distance computation
#[cfg(feature = "gpu-cuda")]
pub mod kernels {
    /// L2 squared distance kernel (PTX would be pre-compiled)
    pub const L2_SQUARED_CUDA: &str = r#"
extern "C" __global__ void l2_squared(
    const float* __restrict__ query,
    const float* __restrict__ database,
    float* __restrict__ distances,
    int dim,
    int n_vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    float sum = 0.0f;
    const float* db_vec = database + idx * dim;

    for (int d = 0; d < dim; d++) {
        float diff = query[d] - db_vec[d];
        sum += diff * diff;
    }

    distances[idx] = sum;
}
"#;

    /// Cosine similarity kernel
    pub const COSINE_CUDA: &str = r#"
extern "C" __global__ void cosine_similarity(
    const float* __restrict__ query,
    const float* __restrict__ database,
    float* __restrict__ similarities,
    int dim,
    int n_vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    const float* db_vec = database + idx * dim;

    for (int d = 0; d < dim; d++) {
        float a = query[d];
        float b = db_vec[d];
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);

    if (norm_a > 0.0f && norm_b > 0.0f) {
        similarities[idx] = dot / (norm_a * norm_b);
    } else {
        similarities[idx] = 0.0f;
    }
}
"#;

    /// Inner product kernel
    pub const INNER_PRODUCT_CUDA: &str = r#"
extern "C" __global__ void inner_product(
    const float* __restrict__ query,
    const float* __restrict__ database,
    float* __restrict__ products,
    int dim,
    int n_vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    float dot = 0.0f;
    const float* db_vec = database + idx * dim;

    for (int d = 0; d < dim; d++) {
        dot += query[d] * db_vec[d];
    }

    products[idx] = dot;
}
"#;
}

