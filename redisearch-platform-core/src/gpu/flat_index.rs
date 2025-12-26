//! GPU Flat Index (Brute Force)
//!
//! Exact nearest neighbor search using GPU-accelerated distance computation.
//! Best for datasets up to ~1M vectors.

use std::collections::HashMap;
use std::sync::RwLock;

use super::{
    DistanceMetric, GpuConfig, GpuError, GpuVectorIndex, Result, SearchResult,
    memory::AlignedVectors,
};

/// GPU Flat Index for exact nearest neighbor search
pub struct GpuFlatIndex {
    /// Vector dimension
    dim: usize,

    /// Distance metric
    metric: DistanceMetric,

    /// GPU configuration
    config: GpuConfig,

    /// Stored vectors (aligned for GPU)
    vectors: RwLock<AlignedVectors>,

    /// ID mapping: internal index -> external ID
    ids: RwLock<Vec<u64>>,

    /// Reverse mapping: external ID -> internal index
    id_to_index: RwLock<HashMap<u64, usize>>,

    /// GPU backend state
    #[cfg(feature = "gpu-wgpu")]
    wgpu_state: Option<WgpuFlatState>,

    #[cfg(feature = "gpu-cuda")]
    cuda_state: Option<CudaFlatState>,
}

#[cfg(feature = "gpu-wgpu")]
struct WgpuFlatState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    distance_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu-cuda")]
struct CudaFlatState {
    device: cudarc::driver::CudaDevice,
    // CUDA-specific state
}

impl GpuFlatIndex {
    /// Create new GPU flat index
    pub fn new(dim: usize, metric: DistanceMetric, config: GpuConfig) -> Result<Self> {
        let alignment = config.dimension_alignment;

        #[cfg(feature = "gpu-wgpu")]
        let wgpu_state = if config.backend == super::GpuBackend::Wgpu
            || config.backend == super::GpuBackend::Auto
        {
            Self::init_wgpu(&config, &metric)?
        } else {
            None
        };

        #[cfg(feature = "gpu-cuda")]
        let cuda_state = if config.backend == super::GpuBackend::Cuda {
            Self::init_cuda(&config)?
        } else {
            None
        };

        Ok(Self {
            dim,
            metric,
            config,
            vectors: RwLock::new(AlignedVectors::new(dim, alignment)),
            ids: RwLock::new(Vec::new()),
            id_to_index: RwLock::new(HashMap::new()),
            #[cfg(feature = "gpu-wgpu")]
            wgpu_state,
            #[cfg(feature = "gpu-cuda")]
            cuda_state,
        })
    }

    #[cfg(feature = "gpu-wgpu")]
    fn init_wgpu(config: &GpuConfig, metric: &DistanceMetric) -> Result<Option<WgpuFlatState>> {
        use pollster::FutureExt;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()
            .ok_or_else(|| GpuError::NoDevice)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("RedVector GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .block_on()
            .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;

        // Select shader based on metric
        let shader_source = match metric {
            DistanceMetric::L2 => super::distance::shaders::L2_SQUARED_WGSL,
            DistanceMetric::Cosine => super::distance::shaders::COSINE_WGSL,
            DistanceMetric::InnerProduct => super::distance::shaders::INNER_PRODUCT_WGSL,
        };

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Distance Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Distance Bind Group Layout"),
            entries: &[
                // Query buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Database buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Distance Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let distance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Distance Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Some(WgpuFlatState {
            device,
            queue,
            distance_pipeline,
            bind_group_layout,
        }))
    }

    #[cfg(feature = "gpu-cuda")]
    fn init_cuda(config: &GpuConfig) -> Result<Option<CudaFlatState>> {
        // CUDA initialization would go here
        // For now, return None
        Ok(None)
    }

    /// CPU fallback search
    fn search_cpu(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let vectors = self.vectors.read().unwrap();
        let ids = self.ids.read().unwrap();

        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Compute all distances
        let mut results: Vec<SearchResult> = (0..vectors.len())
            .map(|i| {
                let vec = vectors.get(i).unwrap();
                let score = self.metric.compute(query, vec);
                SearchResult::new(ids[i], score)
            })
            .collect();

        // Sort by score
        if self.metric.is_similarity() {
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        } else {
            results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        }

        // Take top k
        results.truncate(k);
        Ok(results)
    }

    /// GPU search using wgpu
    #[cfg(feature = "gpu-wgpu")]
    fn search_wgpu(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let state = self.wgpu_state.as_ref().ok_or(GpuError::NoDevice)?;
        let vectors = self.vectors.read().unwrap();
        let ids = self.ids.read().unwrap();

        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        use wgpu::util::DeviceExt;

        // Create buffers
        let query_buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Query Buffer"),
                contents: bytemuck::cast_slice(query),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let database_buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Database Buffer"),
                contents: bytemuck::cast_slice(vectors.raw_data()),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let n_vectors = vectors.len();
        let output_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (n_vectors * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (n_vectors * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params: dim, n_vectors, query_offset, padding
        let params = [self.dim as u32, n_vectors as u32, 0u32, 0u32];
        let params_buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Distance Bind Group"),
            layout: &state.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: database_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode and submit
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Distance Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Distance Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&state.distance_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (n_vectors as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());

        state.queue.submit(Some(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        state.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .unwrap()
            .map_err(|e| GpuError::ComputeError(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let scores: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Build results with scores
        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .enumerate()
            .map(|(i, score)| SearchResult::new(ids[i], score))
            .collect();

        // Sort and truncate
        if self.metric.is_similarity() {
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        } else {
            results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        }

        results.truncate(k);
        Ok(results)
    }
}

impl GpuVectorIndex for GpuFlatIndex {
    fn add(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        if vectors.len() % self.dim != 0 {
            return Err(GpuError::DimensionMismatch {
                expected: self.dim,
                got: vectors.len() % self.dim,
            });
        }

        let n_vectors = vectors.len() / self.dim;
        if ids.len() != n_vectors {
            return Err(GpuError::DimensionMismatch {
                expected: n_vectors,
                got: ids.len(),
            });
        }

        let mut vec_storage = self.vectors.write().unwrap();
        let mut id_vec = self.ids.write().unwrap();
        let mut id_map = self.id_to_index.write().unwrap();

        for (i, &id) in ids.iter().enumerate() {
            let index = vec_storage.len();
            let vec_data = &vectors[i * self.dim..(i + 1) * self.dim];
            vec_storage.add(vec_data);
            id_vec.push(id);
            id_map.insert(id, index);
        }

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(GpuError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        match self.config.backend {
            #[cfg(feature = "gpu-wgpu")]
            super::GpuBackend::Wgpu | super::GpuBackend::Auto if self.wgpu_state.is_some() => {
                self.search_wgpu(query, k)
            }
            #[cfg(feature = "gpu-cuda")]
            super::GpuBackend::Cuda if self.cuda_state.is_some() => {
                // CUDA search would go here
                self.search_cpu(query, k)
            }
            _ => self.search_cpu(query, k),
        }
    }

    fn batch_search(&self, queries: &[f32], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        let n_queries = queries.len() / self.dim;
        let mut results = Vec::with_capacity(n_queries);

        for i in 0..n_queries {
            let query = &queries[i * self.dim..(i + 1) * self.dim];
            results.push(self.search(query, k)?);
        }

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.vectors.read().unwrap().len()
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn config(&self) -> &GpuConfig {
        &self.config
    }

    fn remove(&mut self, ids: &[u64]) -> Result<usize> {
        // For flat index, removal requires rebuilding
        // This is a simple implementation that marks as removed
        let id_map = self.id_to_index.read().unwrap();
        let count = ids.iter().filter(|id| id_map.contains_key(id)).count();
        // TODO: Implement proper removal with compaction
        Ok(count)
    }

    fn clear(&mut self) -> Result<()> {
        self.vectors.write().unwrap().clear();
        self.ids.write().unwrap().clear();
        self.id_to_index.write().unwrap().clear();
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // No pending operations for flat index
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.vectors.read().unwrap().memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_index_cpu() {
        let config = GpuConfig::cpu();
        let mut index = GpuFlatIndex::new(3, DistanceMetric::L2, config).unwrap();

        // Add some vectors
        let ids = vec![1, 2, 3];
        let vectors = vec![
            1.0, 0.0, 0.0, // vec 1
            0.0, 1.0, 0.0, // vec 2
            0.0, 0.0, 1.0, // vec 3
        ];

        index.add(&ids, &vectors).unwrap();
        assert_eq!(index.len(), 3);

        // Search
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // Closest to query
        assert!(results[0].score < 0.01); // L2 distance should be ~0
    }

    #[test]
    fn test_flat_index_cosine() {
        let config = GpuConfig::cpu();
        let mut index = GpuFlatIndex::new(3, DistanceMetric::Cosine, config).unwrap();

        let ids = vec![1, 2];
        let vectors = vec![
            1.0, 0.0, 0.0, // vec 1
            0.0, 1.0, 0.0, // vec 2
        ];

        index.add(&ids, &vectors).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results[0].id, 1);
        assert!((results[0].score - 1.0).abs() < 0.01); // Cosine similarity = 1
    }
}

