use std::sync::Arc;
use std::vec::Vec;

use crate::driver::{
    result::{self, DriverError},
    sys,
};

use super::{CudaEvent, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceRepr};

/// Configuration for [result::launch_kernel]
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)
/// for description of each parameter.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    /// (width, height, depth) of grid in blocks
    pub grid_dim: (u32, u32, u32),

    /// (x, y, z) dimension of each thread block
    pub block_dim: (u32, u32, u32),

    /// Dynamic shared-memory size per thread block in bytes
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Creates a [LaunchConfig] with:
    /// - block_dim == `1024`
    /// - grid_dim == `(n + 1023) / 1024`
    /// - shared_mem_bytes == `0`
    pub fn for_num_elems(n: u32) -> Self {
        const NUM_THREADS: u32 = 1024;
        let num_blocks = n.div_ceil(NUM_THREADS);
        Self {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// The kernel launch builder. Instantiate with [CudaStream::launch_builder()], and then
/// launch the kernel with [LaunchArgs::launch()]
///
/// Anything added as a kernel argument with [LaunchArgs::arg()] must either:
/// 1. Implement [DeviceRepr]
/// 2. Add a custom implementation of `impl<'a> PushKernelArg<T> for LaunchArgs<'a>`, where `T` is your type.
#[derive(Debug)]
pub struct LaunchArgs<'a> {
    pub(super) stream: &'a Arc<CudaStream>,
    pub(super) func: &'a CudaFunction,
    pub(super) waits: Vec<&'a CudaEvent>,
    pub(super) records: Vec<&'a CudaEvent>,
    pub(super) args: Vec<*mut std::ffi::c_void>,
    pub(super) flags: Option<sys::CUevent_flags>,
}

impl CudaStream {
    /// Creates a new kernel launch builder that will launch `func` on stream `self`.
    ///
    /// Add arguments to the builder using [LaunchArgs::arg()], and submit it to the stream
    /// using [LaunchArgs::launch()].
    pub fn launch_builder<'a>(self: &'a Arc<Self>, func: &'a CudaFunction) -> LaunchArgs<'a> {
        LaunchArgs {
            stream: self,
            func,
            waits: Vec::new(),
            records: Vec::new(),
            args: Vec::new(),
            flags: None,
        }
    }
}

/// Something that can be copied to device memory and
/// turned into a parameter for [result::launch_kernel].
///
/// See the [cuda docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-function-argument-processing)
/// on argument processing.
///
/// # Safety
///
/// This is unsafe you need to ensure that T can be represented
/// in CUDA and also references to it can be properly passed to CUDA.
///
/// Most implementations will be required to use `#[inline(always)]`
/// to ensure that references are handled properly.
pub unsafe trait PushKernelArg<T> {
    fn arg(&mut self, arg: T) -> &mut Self;
}

unsafe impl<'a, 'b: 'a, T: DeviceRepr> PushKernelArg<&'b T> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b T) -> &mut Self {
        self.args.push(arg as *const T as *mut _);
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b CudaSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaSlice<T>) -> &mut Self {
        if self.stream.context().is_in_multi_stream_mode() {
            if let Some(write) = arg.write.as_ref() {
                self.waits.push(write);
            }
            if let Some(read) = arg.read.as_ref() {
                self.records.push(read);
            }
        }
        self.args
            .push((&arg.cu_device_ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b mut CudaSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaSlice<T>) -> &mut Self {
        if self.stream.context().is_in_multi_stream_mode() {
            if let Some(read) = arg.read.as_ref() {
                self.waits.push(read);
            }
            if let Some(write) = arg.write.as_ref() {
                self.waits.push(write);
                self.records.push(write);
            }
        }
        self.args
            .push((&arg.cu_device_ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b CudaView<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaView<'c, T>) -> &mut Self {
        if self.stream.context().is_in_multi_stream_mode() {
            if let Some(write) = arg.write.as_ref() {
                self.waits.push(write);
            }
            if let Some(read) = arg.read.as_ref() {
                self.records.push(read);
            }
        }
        self.args.push((&arg.ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b mut CudaViewMut<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaViewMut<'c, T>) -> &mut Self {
        if self.stream.context().is_in_multi_stream_mode() {
            if let Some(read) = arg.read.as_ref() {
                self.waits.push(read);
            }
            if let Some(write) = arg.write.as_ref() {
                self.waits.push(write);
                self.records.push(write);
            }
        }
        self.args.push((&arg.ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

impl LaunchArgs<'_> {
    /// Calling this will make [LaunchArgs::launch()] and [LaunchArgs::launch_cooperative()]
    /// return 2 [CudaEvent]s that recorded before and after the kernel is submitted.
    pub fn record_kernel_launch(&mut self, flags: sys::CUevent_flags) -> &mut Self {
        self.flags = Some(flags);
        self
    }

    /// Submits the configuration [CudaFunction] to execute asychronously on
    /// the configured device stream.
    ///
    /// # Safety
    ///
    /// This is generally unsafe for two main reasons:
    ///
    /// 1. We can't guarantee that the arguments are valid for the configured [CudaFunction].
    ///    We don't know if the types are correct, if the arguments are in the correct order,
    ///    if the types are representable in CUDA, etc.
    /// 2. We can't guarantee that the cuda kernel follows the mutability of the arguments
    ///    configured with [LaunchArgs::arg()]. For instance, you can pass a reference to a [CudaSlice],
    ///    which on rust side can't be mutated, but on cuda side the kernel can mutate it.
    /// 3. [CudaFunction] can access memory outside of limits.
    ///
    /// ## Handling asynchronous mutation
    ///
    /// All [CudaSlice]/[CudaView]/[CudaViewMut] contain 2 events that record
    /// when the data associated with them are read from/written to.
    ///
    /// The [PushKernelArg] implementation of these adds these events to [LaunchArgs],
    /// so when [LaunchArgs::launch()] is called, we properly do multi stream synchronization.
    ///
    /// So in practice it is not possible to have multiple kernels concurrently modify device
    /// data while using the safe api.
    ///
    /// ## Handling use after free
    ///
    /// Since [LaunchArgs::launch()] properly records reads/writes for [CudaSlice]/[CudaView]/[CudaViewMut],
    /// and the drop implementation of [CudaSlice] waits on those events to finish,
    /// we will never encounter a use after free situation.
    #[inline(always)]
    unsafe fn inner_launch(
        &mut self,
        cfg: LaunchConfig,
    ) -> Result<Option<(CudaEvent, CudaEvent)>, DriverError> {
        self.stream.ctx.bind_to_thread()?;
        for &event in self.waits.iter() {
            self.stream.wait(event)?;
        }
        let start_event = self
            .flags
            .map(|flags| self.stream.record_event(Some(flags)))
            .transpose()?;
        result::launch_kernel(
            self.func.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream,
            &mut self.args,
        )?;
        let end_event = self
            .flags
            .map(|flags| self.stream.record_event(Some(flags)))
            .transpose()?;
        for &event in self.records.iter() {
            event.record(self.stream)?;
        }
        Ok(start_event.zip(end_event))
    }

    #[inline(always)]
    pub unsafe fn launch(
        &mut self,
        cfg: LaunchConfig,
    ) -> Result<Option<(CudaEvent, CudaEvent)>, DriverError> {
        let result = self.inner_launch(cfg);
        if self.stream.fuel_check {
            match self.perform_fuel_check() {
                Ok(()) => {}
                Err(e) => {
                    return Err(e);
                }
            }
        }
        result
    }

    /// Launch a cooperative kernel.
    ///
    /// # Safety
    /// See [LaunchArgs::launch()]
    #[inline(always)]
    unsafe fn inner_launch_cooperative(
        &mut self,
        cfg: LaunchConfig,
    ) -> Result<Option<(CudaEvent, CudaEvent)>, DriverError> {
        self.stream.ctx.bind_to_thread()?;
        for &event in self.waits.iter() {
            self.stream.wait(event)?;
        }
        let start_event = self
            .flags
            .map(|flags| self.stream.record_event(Some(flags)))
            .transpose()?;
        result::launch_cooperative_kernel(
            self.func.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream,
            &mut self.args,
        )?;
        let end_event = self
            .flags
            .map(|flags| self.stream.record_event(Some(flags)))
            .transpose()?;
        for &event in self.records.iter() {
            event.record(self.stream)?;
        }
        let result = Ok(start_event.zip(end_event));
        if self.stream.fuel_check {
            match self.perform_fuel_check() {
                Ok(()) => {}
                Err(e) => {
                    return Err(e);
                }
            }
        }
        result
    }

    #[inline(always)]
    pub unsafe fn launch_cooperative(
        &mut self,
        cfg: LaunchConfig,
    ) -> Result<Option<(CudaEvent, CudaEvent)>, DriverError> {
        let result = self.inner_launch_cooperative(cfg);
        if self.stream.fuel_check {
            match self.perform_fuel_check() {
                Ok(()) => {}
                Err(e) => {
                    return Err(e);
                }
            }
        }
        result
    }

    #[inline(always)]
    fn perform_fuel_check(&self) -> Result<(), DriverError> {
        // Try to load and launch a finalization kernel if it exists
        let finalize_kernel_result = self.func.module.load_function("finalize_kernel");
        if let Ok(finalize_kernel) = finalize_kernel_result {
            // Create device memory for the three output parameters
            let mut d_fuelusage = self.stream.alloc_zeros::<u64>(1)?;
            let mut d_signature = self.stream.alloc_zeros::<u64>(1)?;
            let mut d_errorstat = self.stream.alloc_zeros::<u64>(1)?;

            // Create a minimal config for a 1-thread kernel
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            // Launch the finalize kernel with the three parameters
            unsafe {
                self.stream
                    .launch_builder(&finalize_kernel)
                    .arg(&mut d_fuelusage)
                    .arg(&mut d_signature)
                    .arg(&mut d_errorstat)
                    .inner_launch(cfg)?;
            }
            // Optionally: Check the error status to see if anything went wrong
            // For example, if FUELUSAGE_EXCEEDED, we could return an error
            let errorstat = self.stream.memcpy_dtov(&d_errorstat)?[0];
            if errorstat != 0 {
                return Err(DriverError(sys::cudaError_enum::TIG_ERROR_OUT_OF_FUEL));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        driver::{CudaContext, DriverError},
        nvrtc::compile_ptx_with_opts,
    };

    use super::*;

    #[test]
    fn test_launch_arrays() -> Result<(), DriverError> {
        #[repr(C)]
        struct TensorMeta {
            num_dims: usize,
            strides: [usize; 128],
            shape: [usize; 128],
        }
        unsafe impl DeviceRepr for TensorMeta {}

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let ptx = compile_ptx_with_opts(
            "
struct TensorMeta {
    size_t num_dims;
    size_t shape[128];
    size_t strides[128];
};

extern \"C\" __global__ void kernel(const TensorMeta meta) {
    for (int i = 0;i < meta.num_dims;i++) {
        assert(meta.shape[i] == i);
        assert(meta.strides[i] == i);
    }
}
        ",
            Default::default(),
        )
        .unwrap();

        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function("kernel").unwrap();

        let meta = TensorMeta {
            num_dims: 128,
            shape: std::array::from_fn(|i| i),
            strides: std::array::from_fn(|i| i),
        };

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&meta)
                .launch(LaunchConfig::for_num_elems(1))
        }?;

        stream.synchronize()?;

        Ok(())
    }

    const SIN_CU: &str = "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}";

    #[test]
    fn test_launch_with_mut_and_ref_cudarc() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let sin_kernel = module.load_function("sin_kernel").unwrap();

        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];

        let a_dev = stream.memcpy_stod(&a_host).unwrap();
        let mut b_dev = a_dev.clone();

        unsafe {
            stream
                .launch_builder(&sin_kernel)
                .arg(&mut b_dev)
                .arg(&a_dev)
                .arg(&10usize)
                .launch(LaunchConfig::for_num_elems(10))
        }
        .unwrap();

        let b_host = stream.memcpy_dtov(&b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    #[test]
    fn test_large_launches() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let sin_kernel = module.load_function("sin_kernel").unwrap();

        for numel in [256, 512, 1024, 1280, 1536, 2048] {
            let mut a = Vec::with_capacity(numel);
            a.resize(numel, 1.0f32);

            let a = stream.memcpy_stod(&a).unwrap();
            let mut b = stream.alloc_zeros::<f32>(numel).unwrap();
            unsafe {
                stream
                    .launch_builder(&sin_kernel)
                    .arg(&mut b)
                    .arg(&a)
                    .arg(&numel)
                    .launch(LaunchConfig::for_num_elems(numel as u32))
            }
            .unwrap();

            let b = stream.memcpy_dtov(&b).unwrap();
            for v in b {
                assert_eq!(v, 0.841471);
            }
        }
    }

    #[test]
    fn test_launch_with_views() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function("sin_kernel").unwrap();

        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];
        let a_dev = stream.memcpy_stod(&a_host).unwrap();
        let mut b_dev = a_dev.clone();

        for i in 0..5 {
            let a_sub = a_dev.try_slice(i * 2..).unwrap();
            assert_eq!(a_sub.len, 10 - 2 * i);
            let mut b_sub = b_dev.try_slice_mut(i * 2..).unwrap();
            assert_eq!(b_sub.len, 10 - 2 * i);
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&mut b_sub)
                    .arg(&a_sub)
                    .arg(&2usize)
                    .launch(LaunchConfig::for_num_elems(2))
            }
            .unwrap();
        }

        let b_host = stream.memcpy_dtov(&b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    const TEST_KERNELS: &str = "
extern \"C\" __global__ void int_8bit(signed char s_min, char s_max, unsigned char u_min, unsigned char u_max) {
    assert(s_min == -128);
    assert(s_max == 127);
    assert(u_min == 0);
    assert(u_max == 255);
}

extern \"C\" __global__ void int_16bit(signed short s_min, short s_max, unsigned short u_min, unsigned short u_max) {
    assert(s_min == -32768);
    assert(s_max == 32767);
    assert(u_min == 0);
    assert(u_max == 65535);
}

extern \"C\" __global__ void int_32bit(signed int s_min, int s_max, unsigned int u_min, unsigned int u_max) {
    assert(s_min == -2147483648);
    assert(s_max == 2147483647);
    assert(u_min == 0);
    assert(u_max == 4294967295);
}

extern \"C\" __global__ void int_64bit(signed long s_min, long s_max, unsigned long u_min, unsigned long u_max) {
    assert(s_min == -9223372036854775808);
    assert(s_max == 9223372036854775807);
    assert(u_min == 0);
    assert(u_max == 18446744073709551615);
}

extern \"C\" __global__ void floating(float f, double d) {
    assert(fabs(f - 1.2345678) <= 1e-7);
    assert(fabs(d - -10.123456789876543) <= 1e-16);
}
";

    #[test]
    fn test_launch_with_8bit() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function("int_8bit").unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&i8::MIN)
                .arg(&i8::MAX)
                .arg(&u8::MIN)
                .arg(&u8::MAX)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_16bit() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function("int_16bit").unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&i16::MIN)
                .arg(&i16::MAX)
                .arg(&u16::MIN)
                .arg(&u16::MAX)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_32bit() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function("int_32bit").unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&i32::MIN)
                .arg(&i32::MAX)
                .arg(&u32::MIN)
                .arg(&u32::MAX)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_64bit() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function("int_64bit").unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&i64::MIN)
                .arg(&i64::MAX)
                .arg(&u64::MIN)
                .arg(&u64::MAX)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_floats() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function("floating").unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&1.2345678f32)
                .arg(&-10.123456789876543f64)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        stream.synchronize().unwrap();
    }

    #[cfg(feature = "f16")]
    const HALF_KERNELS: &str = "
#include \"cuda_fp16.h\"

extern \"C\" __global__ void halfs(__half h) {
    assert(__habs(h - __float2half(1.234)) <= __float2half(1e-4));
}
";

    #[cfg(feature = "f16")]
    #[test]
    fn test_launch_with_half() {
        use crate::nvrtc::CompileOptions;

        let ptx = compile_ptx_with_opts(
            HALF_KERNELS,
            CompileOptions {
                include_paths: std::vec!["/usr/include".into()],
                arch: Some("compute_53"),
                ..Default::default()
            },
        )
        .unwrap();
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function("halfs").unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&half::f16::from_f32(1.234))
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        stream.synchronize().unwrap();
    }

    const SLOW_KERNELS: &str = "
extern \"C\" __global__ void slow_worker(const float *data, const size_t len, float *out) {
    float tmp = 0.0;
    for(size_t i = 0; i < 1000000; i++) {
        tmp += data[i % len];
    }
    *out = tmp;
}
";

    #[test]
    fn test_par_launch() -> Result<(), DriverError> {
        let ptx = compile_ptx_with_opts(SLOW_KERNELS, Default::default()).unwrap();
        let ctx = CudaContext::new(0)?;
        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function("slow_worker").unwrap();

        let stream = ctx.new_stream()?;
        let slice = stream.alloc_zeros::<f32>(1000)?;
        let mut a = stream.alloc_zeros::<f32>(1)?;
        let mut b = stream.alloc_zeros::<f32>(1)?;
        stream.synchronize()?;

        let cfg = LaunchConfig::for_num_elems(1);

        let double_launch_ms = {
            // launch two kernels on the default stream
            let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&slice)
                    .arg(&slice.len())
                    .arg(&mut a)
                    .launch(cfg)?
            };
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&slice)
                    .arg(&slice.len())
                    .arg(&mut b)
                    .launch(cfg)?
            };
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            stream.synchronize()?;
            start.elapsed_ms(&end)?
        };

        let stream2 = stream.fork()?;
        let par_launch_ms = {
            // create a new stream & launch them concurrently
            let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&slice)
                    .arg(&slice.len())
                    .arg(&mut a)
                    .launch(cfg)?
            };
            unsafe {
                stream2
                    .launch_builder(&f)
                    .arg(&slice)
                    .arg(&slice.len())
                    .arg(&mut b)
                    .launch(cfg)?
            };
            stream.join(&stream2)?;
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            stream.synchronize()?;
            start.elapsed_ms(&end)?
        };

        assert!(
            (double_launch_ms - 2.0 * par_launch_ms).abs() < 0.2 * double_launch_ms,
            "par={:?} dbl={:?}",
            par_launch_ms,
            double_launch_ms
        );
        Ok(())
    }

    #[test]
    fn test_multi_stream_concurrent_reads() -> Result<(), DriverError> {
        let ptx = compile_ptx_with_opts(SLOW_KERNELS, Default::default()).unwrap();
        let ctx = CudaContext::new(0)?;
        let module = ctx.load_module(ptx)?;
        let f = module.load_function("slow_worker")?;

        let stream1 = ctx.new_stream()?;

        let src = stream1.alloc_zeros::<f32>(1000)?;
        let numel = src.len();
        let mut dst1 = stream1.alloc_zeros::<f32>(1)?;
        let mut dst2 = stream1.alloc_zeros::<f32>(1)?;

        let stream2 = stream1.fork()?;

        let cfg = LaunchConfig::for_num_elems(1);

        let mut builder = stream1.launch_builder(&f);
        builder
            .arg(&src)
            .arg(&numel)
            .arg(&mut dst1)
            .record_kernel_launch(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let (_, stream1_finish) = unsafe { builder.launch(cfg) }?.unwrap();

        let mut builder = stream2.launch_builder(&f);
        builder
            .arg(&src)
            .arg(&numel)
            .arg(&mut dst2)
            .record_kernel_launch(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let (stream2_start, _) = unsafe { builder.launch(cfg) }?.unwrap();

        stream1.synchronize()?;
        stream2.synchronize()?;

        assert!(stream2_start.elapsed_ms(&stream1_finish)? >= 0.0);
        Ok(())
    }

    #[test]
    fn test_multi_stream_writes_block() -> Result<(), DriverError> {
        let ptx = compile_ptx_with_opts(SLOW_KERNELS, Default::default()).unwrap();
        let ctx = CudaContext::new(0)?;
        let module = ctx.load_module(ptx)?;
        let f = module.load_function("slow_worker")?;

        let stream1 = ctx.new_stream()?;

        let src = stream1.alloc_zeros::<f32>(1000)?;
        let numel = src.len();
        let mut dst = stream1.alloc_zeros::<f32>(1)?;
        let cfg = LaunchConfig::for_num_elems(1);

        let stream2 = stream1.fork()?;

        let mut builder = stream1.launch_builder(&f);
        builder
            .arg(&src)
            .arg(&numel)
            .arg(&mut dst)
            .record_kernel_launch(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let (_, stream1_finish) = unsafe { builder.launch(cfg) }?.unwrap();

        let mut builder = stream2.launch_builder(&f);
        builder
            .arg(&src)
            .arg(&numel)
            .arg(&mut dst)
            .record_kernel_launch(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let (stream2_start, _) = unsafe { builder.launch(cfg) }?.unwrap();

        stream1.synchronize()?;
        stream2.synchronize()?;

        assert!(stream1_finish.elapsed_ms(&stream2_start)? >= 0.0);
        Ok(())
    }

    #[test]
    #[ignore = "must be executed by itself"]
    fn test_device_side_assert() -> Result<(), DriverError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let inp = stream.memcpy_stod(&[1.0f32; 100])?;
        let mut out = stream.alloc_zeros::<f32>(100)?;
        let ptx = crate::nvrtc::compile_ptx(
            "
    extern \"C\" __global__ void foo(float *out, const float *inp, const size_t numel) {
        assert(0);
    }",
        )
        .unwrap();
        let module = ctx.load_module(ptx)?;
        let foo = module.load_function("foo")?;
        let mut builder = stream.launch_builder(&foo);
        builder.arg(&mut out);
        builder.arg(&inp);
        builder.arg(&100usize);
        unsafe { builder.launch(LaunchConfig::for_num_elems(100)) }?;
        std::thread::sleep(std::time::Duration::from_secs(1));
        stream
            .synchronize()
            .expect_err("Should've had device side assert");
        Ok(())
    }
}
