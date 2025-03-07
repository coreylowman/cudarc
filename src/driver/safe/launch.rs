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

#[derive(Debug)]
pub struct LaunchArgs<'a> {
    stream: &'a CudaStream,
    func: &'a CudaFunction,
    waits: Vec<&'a CudaEvent>,
    records: Vec<&'a CudaEvent>,
    args: Vec<*mut std::ffi::c_void>,
    start_event: Option<&'a CudaEvent>,
    end_event: Option<&'a CudaEvent>,
}

impl CudaStream {
    pub fn launch_builder<'a>(&'a self, func: &'a CudaFunction) -> LaunchArgs<'a> {
        LaunchArgs {
            stream: self,
            func,
            waits: Vec::new(),
            records: Vec::new(),
            args: Vec::new(),
            start_event: None,
            end_event: None,
        }
    }
}

/// Something that can be copied to device memory and
/// turned into a parameter for [result::launch_kernel].
///
/// # Safety
///
/// This is unsafe because a struct should likely
/// be `#[repr(C)]` to be represented in cuda memory,
/// and not all types are valid.
pub unsafe trait PushKernelArg<T> {
    fn arg(&mut self, arg: T) -> &mut Self;
}

unsafe impl<T: DeviceRepr> PushKernelArg<T> for LaunchArgs<'_> {
    #[inline(always)]
    fn arg(&mut self, arg: T) -> &mut Self {
        self.args.push((&arg) as *const _ as *mut _);
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b CudaSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaSlice<T>) -> &mut Self {
        self.waits.push(&arg.write);
        self.records.push(&arg.read);
        self.args
            .push((&arg.cu_device_ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b mut CudaSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaSlice<T>) -> &mut Self {
        self.waits.push(&arg.read);
        self.waits.push(&arg.write);
        self.records.push(&arg.write);
        self.args
            .push((&arg.cu_device_ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b CudaView<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaView<'c, T>) -> &mut Self {
        self.waits.push(arg.write);
        self.records.push(arg.read);
        self.args.push((&arg.ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b mut CudaViewMut<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaViewMut<'c, T>) -> &mut Self {
        self.waits.push(arg.read);
        self.waits.push(arg.write);
        self.records.push(arg.write);
        self.args.push((&arg.ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

impl<'a> LaunchArgs<'a> {
    pub fn record_start(&mut self, event: &'a CudaEvent) -> &mut Self {
        self.start_event = Some(event);
        self
    }

    pub fn record_end(&mut self, event: &'a CudaEvent) -> &mut Self {
        self.end_event = Some(event);
        self
    }

    /// Consumes a [CudaFunction] to execute asychronously on the device with
    /// params determined by generic parameter `Params`.
    ///
    /// This is impl'd multiple times for different number and types of params. In
    /// general, `Params` should impl [DeviceRepr].
    ///
    /// ```ignore
    /// # use cudarc::driver::*;
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let my_kernel: CudaFunction = dev.get_func("my_module", "my_kernel").unwrap();
    /// let cfg: LaunchConfig = LaunchConfig {
    ///     grid_dim: (1, 1, 1),
    ///     block_dim: (1, 1, 1),
    ///     shared_mem_bytes: 0,
    /// };
    /// let params = (1i32, 2u64, 3usize);
    /// unsafe { my_kernel.launch(cfg, params) }.unwrap();
    /// ```
    ///
    /// # Safety
    ///
    /// This is not safe really ever, because there's no garuntee that `Params`
    /// will work for any [CudaFunction] passed in. Great care should be taken
    /// to ensure that [CudaFunction] works with `Params` and that the correct
    /// parameters have `&mut` in front of them.
    ///
    /// Additionally, kernels can mutate data that is marked as immutable,
    /// such as `&CudaSlice<T>`.
    ///
    /// See [LaunchAsync::launch] for more details
    ///
    /// # TODO Pt 2
    ///
    /// Launches the [CudaFunction] with the corresponding `Params`.
    ///
    /// # Safety
    ///
    /// This method is **very** unsafe.
    ///
    /// See cuda documentation notes on this as well:
    /// <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#functions>
    ///
    /// 1. `params` can be changed regardless of `&` or `&mut` usage.
    /// 2. `params` will be changed at some later point after the
    ///    function returns because the kernel is executed async.
    /// 3. There are no guaruntees that the `params`
    ///    are the correct number/types/order for `func`.
    /// 4. Specifying the wrong values for [LaunchConfig] can result
    ///    in accessing/modifying values past memory limits.
    ///
    /// ## Asynchronous mutation
    ///
    /// Since this library queues kernels to be launched on a single
    /// stream, and really the only way to modify [crate::driver::CudaSlice] is through
    /// kernels, mutating the same [crate::driver::CudaSlice] with multiple kernels
    /// is safe. This is because each kernel is executed sequentially
    /// on the stream.
    ///
    /// **Modifying a value on the host that is in used by a
    /// kernel is undefined behavior.** But is hard to do
    /// accidentally.
    ///
    /// Also for this reason, do not pass in any values to kernels
    /// that can be modified on the host. This is the reason
    /// [DeviceRepr] is not implemented for rust primitive
    /// references.
    ///
    /// ## Use after free
    ///
    /// Since the drop implementation for [crate::driver::CudaSlice] also occurs
    /// on the device's single stream, any kernels launched before
    /// the drop will complete before the value is actually freed.
    ///
    /// **If you launch a kernel or drop a value on a different stream
    /// this may not hold**
    pub unsafe fn launch(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        for &event in self.waits.iter() {
            self.stream.wait(event)?;
        }
        if let Some(event) = self.start_event {
            event.record(self.stream)?;
        }
        result::launch_kernel(
            self.func.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream,
            &mut self.args,
        )?;
        if let Some(event) = self.end_event {
            event.record(self.stream)?;
        }
        for &event in self.records.iter() {
            event.record(self.stream)?;
        }
        Ok(())
    }

    /// Launch a Cooperative kernel. See [LaunchArgs::launch()]
    ///
    /// # Safety
    ///
    /// See [LaunchArgs::launch()]
    pub unsafe fn launch_cooperative(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        for &event in self.waits.iter() {
            self.stream.wait(event)?;
        }
        if let Some(event) = self.start_event {
            event.record(self.stream)?;
        }
        result::launch_cooperative_kernel(
            self.func.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream,
            &mut self.args,
        )?;
        if let Some(event) = self.end_event {
            event.record(self.stream)?;
        }
        for &event in self.records.iter() {
            event.record(self.stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::{
        driver::{CudaContext, CudaDevice, DeviceSlice, DriverError},
        nvrtc::compile_ptx_with_opts,
    };

    use super::*;

    const SIN_CU: &str = "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}";

    #[test]
    fn test_launch_with_mut_and_ref_cudarc() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sin", &["sin_kernel"]).unwrap();

        let sin_kernel = dev.get_func("sin", "sin_kernel").unwrap();

        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];

        let a_dev = dev.htod_copy(a_host.clone().to_vec()).unwrap();

        let mut b_dev = a_dev.clone();

        unsafe {
            dev.stream()
                .launch_builder(&sin_kernel)
                .arg(&mut b_dev)
                .arg(&a_dev)
                .arg(10usize)
                .launch(LaunchConfig::for_num_elems(10))
        }
        .unwrap();

        let b_host = dev.sync_reclaim(b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    #[test]
    fn test_large_launches() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sin", &["sin_kernel"]).unwrap();
        for numel in [256, 512, 1024, 1280, 1536, 2048] {
            let mut a = Vec::with_capacity(numel);
            a.resize(numel, 1.0f32);

            let a = dev.htod_copy(a).unwrap();
            let mut b = dev.alloc_zeros::<f32>(numel).unwrap();

            let sin_kernel = dev.get_func("sin", "sin_kernel").unwrap();
            unsafe {
                dev.stream()
                    .launch_builder(&sin_kernel)
                    .arg(&mut b)
                    .arg(&a)
                    .arg(numel)
                    .launch(LaunchConfig::for_num_elems(numel as u32))
            }
            .unwrap();

            let b = dev.sync_reclaim(b).unwrap();
            for v in b {
                assert_eq!(v, 0.841471);
            }
        }
    }

    #[test]
    fn test_launch_with_views() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sin", &["sin_kernel"]).unwrap();

        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];
        let a_dev = dev.htod_copy(a_host.clone().to_vec()).unwrap();
        let mut b_dev = a_dev.clone();

        for i in 0..5 {
            let a_sub = a_dev.try_slice(i * 2..).unwrap();
            assert_eq!(a_sub.len, 10 - 2 * i);
            let mut b_sub = b_dev.try_slice_mut(i * 2..).unwrap();
            assert_eq!(b_sub.len, 10 - 2 * i);
            let f = dev.get_func("sin", "sin_kernel").unwrap();
            unsafe {
                dev.stream()
                    .launch_builder(&f)
                    .arg(&mut b_sub)
                    .arg(&a_sub)
                    .arg(2usize)
                    .launch(LaunchConfig::for_num_elems(2))
            }
            .unwrap();
        }

        let b_host = dev.sync_reclaim(b_dev).unwrap();

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
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["int_8bit"]).unwrap();
        let f = dev.get_func("tests", "int_8bit").unwrap();
        unsafe {
            dev.stream()
                .launch_builder(&f)
                .arg(i8::MIN)
                .arg(i8::MAX)
                .arg(u8::MIN)
                .arg(u8::MAX)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();

        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_16bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["int_16bit"]).unwrap();
        let f = dev.get_func("tests", "int_16bit").unwrap();
        unsafe {
            dev.stream()
                .launch_builder(&f)
                .arg(i16::MIN)
                .arg(i16::MAX)
                .arg(u16::MIN)
                .arg(u16::MAX)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_32bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["int_32bit"]).unwrap();
        let f = dev.get_func("tests", "int_32bit").unwrap();
        unsafe {
            dev.stream()
                .launch_builder(&f)
                .arg(i32::MIN)
                .arg(i32::MAX)
                .arg(u32::MIN)
                .arg(u32::MAX)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_64bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["int_64bit"]).unwrap();
        let f = dev.get_func("tests", "int_64bit").unwrap();
        unsafe {
            dev.stream()
                .launch_builder(&f)
                .arg(i64::MIN)
                .arg(i64::MAX)
                .arg(u64::MIN)
                .arg(u64::MAX)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_floats() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["floating"]).unwrap();
        let f = dev.get_func("tests", "floating").unwrap();
        unsafe {
            dev.stream()
                .launch_builder(&f)
                .arg(1.2345678f32)
                .arg(-10.123456789876543f64)
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        dev.synchronize().unwrap();
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
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["halfs"]).unwrap();
        let f = dev.get_func("tests", "halfs").unwrap();
        unsafe {
            dev.stream()
                .launch_builder(&f)
                .arg(half::f16::from_f32(1.234))
                .launch(LaunchConfig::for_num_elems(1))
        }
        .unwrap();
        dev.synchronize().unwrap();
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
        ctx.set_blocking_synchronize()?;

        let module = ctx.load_ptx(ptx, &["slow_worker"]).unwrap();
        let f = module.get_func("slow_worker").unwrap();

        let stream = ctx.new_stream()?;
        let slice = stream.alloc_zeros::<f32>(1000)?;
        let mut a = stream.alloc_zeros::<f32>(1)?;
        let mut b = stream.alloc_zeros::<f32>(1)?;
        stream.synchronize()?;

        let cfg = LaunchConfig::for_num_elems(1);

        let start = Instant::now();
        {
            // launch two kernels on the default stream
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&slice)
                    .arg(slice.len())
                    .arg(&mut a)
                    .launch(cfg)?
            };
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&slice)
                    .arg(slice.len())
                    .arg(&mut b)
                    .launch(cfg)?
            };
            stream.synchronize()?;
        }
        let double_launch_s = start.elapsed().as_secs_f64();

        let stream2 = stream.fork()?;
        let start = Instant::now();
        {
            // create a new stream & launch them concurrently
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&slice)
                    .arg(slice.len())
                    .arg(&mut a)
                    .launch(cfg)?
            };
            unsafe {
                stream2
                    .launch_builder(&f)
                    .arg(&slice)
                    .arg(slice.len())
                    .arg(&mut b)
                    .launch(cfg)?
            };
            stream.synchronize()?;
            stream2.synchronize()?;
        }
        let par_launch_s = start.elapsed().as_secs_f64();

        assert!(
            (double_launch_s - 2.0 * par_launch_s).abs() < 20.0 / 100.0,
            "par={:?} dbl={:?}",
            par_launch_s,
            double_launch_s
        );
        Ok(())
    }

    #[test]
    fn test_multi_stream_concurrent_reads() -> Result<(), DriverError> {
        let ptx = compile_ptx_with_opts(SLOW_KERNELS, Default::default()).unwrap();
        let ctx = CudaContext::new(0)?;
        let module = ctx.load_ptx(ptx, &["slow_worker"])?;
        let f = module.get_func("slow_worker").unwrap();

        let stream1_start = ctx.empty_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stream1_finish = ctx.empty_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stream2_start = ctx.empty_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stream2_finish = ctx.empty_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;

        let stream1 = ctx.new_stream()?;

        let src = stream1.alloc_zeros::<f32>(1000)?;
        let mut dst1 = stream1.alloc_zeros::<f32>(1)?;
        let mut dst2 = stream1.alloc_zeros::<f32>(1)?;

        let stream2 = stream1.fork()?;

        let cfg = LaunchConfig::for_num_elems(1);

        let mut builder = stream1.launch_builder(&f);
        builder
            .arg(&src)
            .arg(src.len())
            .arg(&mut dst1)
            .record_start(&stream1_start)
            .record_end(&stream1_finish);
        unsafe { builder.launch(cfg) }?;

        let mut builder = stream2.launch_builder(&f);
        builder
            .arg(&src)
            .arg(src.len())
            .arg(&mut dst2)
            .record_start(&stream2_start)
            .record_end(&stream2_finish);
        unsafe { builder.launch(cfg) }?;

        stream1.synchronize()?;
        stream2.synchronize()?;

        assert!(stream2_start.elapsed_ms(&stream1_finish)? >= 0.0);
        Ok(())
    }

    #[test]
    fn test_multi_stream_writes_block() -> Result<(), DriverError> {
        let ptx = compile_ptx_with_opts(SLOW_KERNELS, Default::default()).unwrap();
        let ctx = CudaContext::new(0)?;
        let module = ctx.load_ptx(ptx, &["slow_worker"])?;
        let f = module.get_func("slow_worker").unwrap();

        let stream1_start = ctx.empty_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stream1_finish = ctx.empty_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stream2_start = ctx.empty_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stream2_finish = ctx.empty_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;

        let stream1 = ctx.new_stream()?;

        let src = stream1.alloc_zeros::<f32>(1000)?;
        let mut dst = stream1.alloc_zeros::<f32>(1)?;
        let cfg = LaunchConfig::for_num_elems(1);

        let stream2 = stream1.fork()?;

        let mut builder = stream1.launch_builder(&f);
        builder
            .arg(&src)
            .arg(src.len())
            .arg(&mut dst)
            .record_start(&stream1_start)
            .record_end(&stream1_finish);
        unsafe { builder.launch(cfg) }?;

        let mut builder = stream2.launch_builder(&f);
        builder
            .arg(&src)
            .arg(src.len())
            .arg(&mut dst)
            .record_start(&stream2_start)
            .record_end(&stream2_finish);
        unsafe { builder.launch(cfg) }?;

        stream1.synchronize()?;
        stream2.synchronize()?;

        let elapsed_ms = stream1_finish.elapsed_ms(&stream2_start)?;
        assert!(elapsed_ms >= 0.0, "{elapsed_ms}");
        Ok(())
    }
}
