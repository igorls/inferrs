use crate::metal::{Buffer, CommandSemaphore, CommandStatus, ComputePipeline, MetalResource};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandEncoder, MTLComputeCommandEncoder, MTLResourceUsage, MTLSize,
};
use std::{ffi::c_void, ptr, sync::Arc};

pub struct ComputeCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
    /// When `false`, `Drop` does NOT call `endEncoding`.  Used for the
    /// persistent-encoder fast path where the encoder is shared across many
    /// dispatches and ended explicitly at commit time.
    owned: bool,
}

impl AsRef<ComputeCommandEncoder> for ComputeCommandEncoder {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self
    }
}
impl ComputeCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw,
            semaphore,
            owned: true,
        }
    }

    /// Create a non-owning alias that encodes dispatches into the same underlying
    /// `MTLComputeCommandEncoder` without calling `endEncoding` on drop.
    ///
    /// Used by the persistent-encoder path in `Commands::command_encoder` so
    /// that callers can dispatch kernels through the encoder without ending it.
    /// Create a non-owning encoder from raw parts: does NOT call `endEncoding` on drop.
    pub fn new_non_owning(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw,
            semaphore,
            owned: false,
        }
    }

    /// Create a non-owning alias that shares the underlying `MTLComputeCommandEncoder`.
    pub fn non_owning_alias(&self) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw: self.raw.clone(),
            semaphore: Arc::clone(&self.semaphore),
            owned: false,
        }
    }

    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup)
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
    }

    pub fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
    }

    pub fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const c_void) {
        let pointer = ptr::NonNull::new(bytes as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(pointer, length, index) }
    }

    pub fn set_bytes<T>(&self, index: usize, data: &T) {
        let size = core::mem::size_of::<T>();
        let ptr = ptr::NonNull::new(data as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(ptr, size, index) }
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    pub fn use_resource<'a>(
        &self,
        resource: impl Into<&'a MetalResource>,
        resource_usage: MTLResourceUsage,
    ) {
        self.raw.useResource_usage(resource.into(), resource_usage)
    }

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding();
        self.signal_encoding_ended();
    }

    pub fn encode_pipeline(&mut self, pipeline: &ComputePipeline) {
        use MTLComputeCommandEncoder as _;
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }
}

impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        if self.owned {
            self.end_encoding();
        }
        // Non-owning aliases (owned=false) do nothing on drop — the persistent
        // encoder in EntryState is ended explicitly by commit_swap_locked.
    }
}

pub struct BlitCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
}

impl AsRef<BlitCommandEncoder> for BlitCommandEncoder {
    fn as_ref(&self) -> &BlitCommandEncoder {
        self
    }
}

impl BlitCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> BlitCommandEncoder {
        BlitCommandEncoder { raw, semaphore }
    }

    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding();
        self.signal_encoding_ended();
    }

    pub fn set_label(&self, label: &str) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }

    pub fn copy_from_buffer(
        &self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer.as_ref(),
                    src_offset,
                    dst_buffer.as_ref(),
                    dst_offset,
                    size,
                )
        }
    }

    pub fn fill_buffer(&self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.raw.fillBuffer_range_value(
            buffer.as_ref(),
            NSRange {
                location: range.0,
                length: range.1,
            },
            value,
        )
    }
}
