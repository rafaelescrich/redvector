//! wgpu backend for cross-platform GPU acceleration
//!
//! Supports Vulkan (Linux/Windows), Metal (macOS), and DX12 (Windows).

use super::{GpuDeviceInfo, GpuBackend, GpuError, Result};

/// Check if wgpu backend is available
pub fn is_available() -> bool {
    use pollster::FutureExt;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .block_on()
        .is_some()
}

/// Enumerate available wgpu devices
pub fn enumerate_devices() -> Vec<GpuDeviceInfo> {
    use pollster::FutureExt;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    adapters
        .into_iter()
        .enumerate()
        .map(|(i, adapter)| {
            let info = adapter.get_info();
            GpuDeviceInfo {
                name: info.name.clone(),
                backend: GpuBackend::Wgpu,
                device_index: i,
                total_memory: 0, // wgpu doesn't expose this directly
                available_memory: 0,
                compute_capability: format!("{:?}", info.backend),
                supports_fp16: true,
                supports_int8: true,
            }
        })
        .collect()
}

/// wgpu device context
pub struct WgpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuContext {
    /// Create new wgpu context
    pub fn new(device_index: usize) -> Result<Self> {
        use pollster::FutureExt;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        if adapters.is_empty() {
            return Err(GpuError::NoDevice);
        }

        let adapter = adapters
            .into_iter()
            .nth(device_index)
            .ok_or(GpuError::NoDevice)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("RedVector"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .block_on()
            .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Get device info
    pub fn info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Create a compute pipeline from WGSL source
    pub fn create_compute_pipeline(&self, shader_source: &str, entry_point: &str) -> wgpu::ComputePipeline {
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point,
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Create a storage buffer
    pub fn create_storage_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Storage Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | usage,
            mapped_at_creation: false,
        })
    }

    /// Create a buffer initialized with data
    pub fn create_buffer_init<T: bytemuck::Pod>(&self, data: &[T], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Init Buffer"),
            contents: bytemuck::cast_slice(data),
            usage,
        })
    }

    /// Submit a compute pass and wait for completion
    pub fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Read buffer contents to CPU
    pub fn read_buffer<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<T>> {
        let size = (count * std::mem::size_of::<T>()) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.submit_and_wait(encoder);

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .unwrap()
            .map_err(|e| GpuError::ComputeError(format!("Buffer read failed: {:?}", e)))?;

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

/// Compute dispatch helper
pub struct ComputeDispatch<'a> {
    context: &'a WgpuContext,
    encoder: wgpu::CommandEncoder,
}

impl<'a> ComputeDispatch<'a> {
    pub fn new(context: &'a WgpuContext) -> Self {
        let encoder = context.device.create_command_encoder(&Default::default());
        Self { context, encoder }
    }

    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let mut pass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    pub fn finish(self) {
        self.context.submit_and_wait(self.encoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_available() {
        // This might fail on CI without GPU
        let available = is_available();
        println!("wgpu available: {}", available);
    }

    #[test]
    fn test_enumerate_devices() {
        let devices = enumerate_devices();
        for (i, dev) in devices.iter().enumerate() {
            println!("Device {}: {} ({:?})", i, dev.name, dev.compute_capability);
        }
    }
}

