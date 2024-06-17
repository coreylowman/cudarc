use crate::runtime::result;

use super::alloc::DeviceRepr;
use super::core::{CudaFunction, CudaStream};
use super::RuntimeError;

use std::vec::Vec;

impl CudaFunction {
    #[inline(always)]
    unsafe fn launch_async_impl(
        self,
        cfg: LaunchConfig,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), result::RuntimeError> {
        result::launch_kernel(
            self.cuda_function as *const std::ffi::c_void,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.device.stream,
            params,
        )
    }

    #[inline(always)]
    unsafe fn par_launch_async_impl(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), result::RuntimeError> {
        if self._driver_device.is_some() {
            if let Err(e) = crate::driver::result::launch_kernel(
                self.cuda_function as crate::driver::sys::CUfunction,
                cfg.grid_dim,
                cfg.block_dim,
                cfg.shared_mem_bytes as u32,
                stream.stream as crate::driver::sys::CUstream,
                params,
            ) {
                Err(RuntimeError::from(e.0))
            } else {
                Ok(())
            }
        } else {
            result::launch_kernel(
                self.cuda_function as *const std::ffi::c_void,
                cfg.grid_dim,
                cfg.block_dim,
                cfg.shared_mem_bytes,
                stream.stream,
                params,
            )
        }
    }
}

/// Configuration for [result::launch_kernel]
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)
/// for description of each parameter.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    /// (width, height, depth) of grid in blocks
    pub grid_dim: (u32, u32, u32),

    /// (x, y, z) dimension of each thread block
    pub block_dim: (u32, u32, u32),

    /// Dynamic shared-memory size per thread block in bytes
    pub shared_mem_bytes: usize,
}

impl LaunchConfig {
    /// Creates a [LaunchConfig] with:
    /// - block_dim == `1024`
    /// - grid_dim == `(n + 1023) / 1024`
    /// - shared_mem_bytes == `0`
    pub fn for_num_elems(n: u32) -> Self {
        const NUM_THREADS: u32 = 1024;
        let num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        Self {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
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
pub unsafe trait LaunchAsync<Params> {
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
    /// function returns because the kernel is executed async.
    /// 3. There are no guaruntees that the `params`
    /// are the correct number/types/order for `func`.
    /// 4. Specifying the wrong values for [LaunchConfig] can result
    /// in accessing/modifying values past memory limits.
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
    unsafe fn launch(self, cfg: LaunchConfig, params: Params) -> Result<(), result::RuntimeError>;

    /// Launch the function on a stream concurrent to the device's default
    /// work stream.
    ///
    /// # Safety
    /// This method is even more unsafe than [LaunchAsync::launch], all the same rules apply,
    /// except now things are executing in parallel to each other.
    ///
    /// That means that if any of the kernels modify the same memory location, you'll get race
    /// conditions or potentially undefined behavior.
    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: Params,
    ) -> Result<(), result::RuntimeError>;
}

unsafe impl LaunchAsync<&mut [*mut std::ffi::c_void]> for CudaFunction {
    #[inline(always)]
    unsafe fn launch(
        self,
        cfg: LaunchConfig,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), result::RuntimeError> {
        self.launch_async_impl(cfg, args)
    }

    #[inline(always)]
    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), result::RuntimeError> {
        self.par_launch_async_impl(stream, cfg, args)
    }
}

unsafe impl LaunchAsync<&mut Vec<*mut std::ffi::c_void>> for CudaFunction {
    #[inline(always)]
    unsafe fn launch(
        self,
        cfg: LaunchConfig,
        args: &mut Vec<*mut std::ffi::c_void>,
    ) -> Result<(), result::RuntimeError> {
        self.launch_async_impl(cfg, args)
    }

    #[inline(always)]
    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        args: &mut Vec<*mut std::ffi::c_void>,
    ) -> Result<(), result::RuntimeError> {
        self.par_launch_async_impl(stream, cfg, args)
    }
}

macro_rules! impl_launch {
    ([$($Vars:tt),*], [$($Idx:tt),*]) => {
        unsafe impl<$($Vars: DeviceRepr),*> LaunchAsync<($($Vars, )*)> for CudaFunction {
            #[inline(always)]
            unsafe fn launch(
                self,
                cfg: LaunchConfig,
                args: ($($Vars, )*)
            ) -> Result<(), result::RuntimeError> {
                let params = &mut [$(args.$Idx.as_kernel_param(), )*];
                self.launch_async_impl(cfg, params)
            }

            #[inline(always)]
            unsafe fn launch_on_stream(
                self,
                stream: &CudaStream,
                cfg: LaunchConfig,
                args: ($($Vars, )*)
            ) -> Result<(), result::RuntimeError> {
                let params = &mut [$(args.$Idx.as_kernel_param(), )*];
                self.par_launch_async_impl(stream, cfg, params)
            }
        }
    };
}

impl_launch!([A], [0]);
impl_launch!([A, B], [0, 1]);
impl_launch!([A, B, C], [0, 1, 2]);
impl_launch!([A, B, C, D], [0, 1, 2, 3]);
impl_launch!([A, B, C, D, E], [0, 1, 2, 3, 4]);
impl_launch!([A, B, C, D, E, F], [0, 1, 2, 3, 4, 5]);
impl_launch!([A, B, C, D, E, F, G], [0, 1, 2, 3, 4, 5, 6]);
impl_launch!([A, B, C, D, E, F, G, H], [0, 1, 2, 3, 4, 5, 6, 7]);
impl_launch!([A, B, C, D, E, F, G, H, I], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K, L],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
);

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use libloading::Symbol;

    use super::*;
    use crate::runtime::{sys, CudaDevice};

    #[test]
    fn test_mut_into_kernel_param_no_inc_rc() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([0.0f32; 1].to_vec()).unwrap();
        let _r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        let _ = (&t).as_kernel_param();
        assert_eq!(Arc::strong_count(&device), 3);
    }

    #[test]
    fn test_ref_into_kernel_param_inc_rc() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([0.0f32; 1].to_vec()).unwrap();
        let _r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        let _ = (&t).as_kernel_param();
        assert_eq!(Arc::strong_count(&device), 3);
    }

    // extern C __global__ void sin_kernel(float *out, const float *inp, size_t numel) {
    //     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (i < numel) {
    //         out[i] = sin(inp[i]);
    //     }
    // }
    //
    // extern C void sin_kernel_wrapper(float *out, const float *inp, size_t numel) {
    //     const size_t block_size = 256;
    //     const size_t grid_size = (numel + block_size - 1) / block_size;
    //     sin_kernel<<<grid_size, block_size>>>(out, inp, numel);
    // }

    #[test]
    fn test_launch_with_mut_and_ref_cudarc() {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();

        let sin_kernel: Symbol<unsafe extern "C" fn(*mut f32, *const f32, usize)> =
            unsafe { test_lib.get(b"sin_kernel_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];
        let a_dev = dev.htod_copy(a_host.clone().to_vec()).unwrap();
        let b_dev = a_dev.clone();

        let numel = a_host.len();
        unsafe {
            sin_kernel(b_dev.device_ptr, a_dev.device_ptr as *const f32, numel);
        }
        let b_host = dev.sync_reclaim(b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    #[test]
    fn test_large_launches() {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();
        let sin_kernel: Symbol<unsafe extern "C" fn(*mut f32, *const f32, usize)> =
            unsafe { test_lib.get(b"sin_kernel_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        for numel in [256, 512, 1024, 1280, 1536, 2048] {
            let mut a = Vec::with_capacity(numel);
            a.resize(numel, 1.0f32);

            let a = dev.htod_copy(a).unwrap();
            let b = dev.alloc_zeros::<f32>(numel).unwrap();

            unsafe {
                sin_kernel(b.device_ptr, a.device_ptr as *const f32, numel);
            }

            let b = dev.sync_reclaim(b).unwrap();
            for v in b {
                assert_eq!(v, 0.841471);
            }
        }
    }

    #[test]
    fn test_launch_with_views() {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();
        let sin_kernel: Symbol<unsafe extern "C" fn(*mut f32, *const f32, usize)> =
            unsafe { test_lib.get(b"sin_kernel_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();

        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];
        let a_dev = dev.htod_copy(a_host.clone().to_vec()).unwrap();
        let mut b_dev = a_dev.clone();

        for i in 0..5 {
            let a_sub = a_dev.try_slice(i * 2..).unwrap();
            assert_eq!(a_sub.len, 10 - 2 * i);
            let b_sub = b_dev.try_slice_mut(i * 2..).unwrap();
            assert_eq!(b_sub.len, 10 - 2 * i);
            unsafe {
                sin_kernel(b_sub.ptr, a_sub.ptr as *const f32, 2);
            }
        }

        let b_host = dev.sync_reclaim(b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    // extern C __global__ void int_8bit(signed char s_min, char s_max, unsigned char u_min, unsigned char u_max) {
    //     assert(s_min == -128);
    //     assert(s_max == 127);
    //     assert(u_min == 0);
    //     assert(u_max == 255);
    // }
    //
    // extern C __global__ void int_16bit(signed short s_min, short s_max, unsigned short u_min, unsigned short u_max) {
    //     assert(s_min == -32768);
    //     assert(s_max == 32767);
    //     assert(u_min == 0);
    //     assert(u_max == 65535);
    // }
    //
    // extern C __global__ void int_32bit(signed int s_min, int s_max, unsigned int u_min, unsigned int u_max) {
    //     assert(s_min == -2147483648);
    //     assert(s_max == 2147483647);
    //     assert(u_min == 0);
    //     assert(u_max == 4294967295);
    // }
    //
    // extern C __global__ void int_64bit(signed long s_min, long s_max, unsigned long u_min, unsigned long u_max) {
    //     assert(s_min == -9223372036854775808);
    //     assert(s_max == 9223372036854775807);
    //     assert(u_min == 0);
    //     assert(u_max == 18446744073709551615);
    // }
    //
    // extern C __global__ void floating(float f, double d) {
    //     assert(fabs(f - 1.2345678) <= 1e-7);
    //     assert(fabs(d - -10.123456789876543) <= 1e-16);
    // }

    #[test]
    fn test_launch_with_8bit() {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();
        let int_8bit: Symbol<unsafe extern "C" fn(i8, i8, u8, u8)> =
            unsafe { test_lib.get(b"int_8bit_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        unsafe { int_8bit(i8::MIN, i8::MAX, u8::MIN, u8::MAX) };

        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_16bit() {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();
        let int_16bit: Symbol<unsafe extern "C" fn(i16, i16, u16, u16)> =
            unsafe { test_lib.get(b"int_16bit_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        unsafe { int_16bit(i16::MIN, i16::MAX, u16::MIN, u16::MAX) };

        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_32bit() {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();
        let int_32bit: Symbol<unsafe extern "C" fn(i32, i32, u32, u32)> =
            unsafe { test_lib.get(b"int_32bit_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        unsafe { int_32bit(i32::MIN, i32::MAX, u32::MIN, u32::MAX) };

        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_64bit() {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();
        let int_64bit: Symbol<unsafe extern "C" fn(i64, i64, u64, u64)> =
            unsafe { test_lib.get(b"int_64bit_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        unsafe { int_64bit(i64::MIN, i64::MAX, u64::MIN, u64::MAX) };

        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_floats() {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();
        let floating: Symbol<unsafe extern "C" fn(f32, f64)> =
            unsafe { test_lib.get(b"floating_wrapper_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        unsafe { floating(1.2345678, -10.123456789876543) };

        dev.synchronize().unwrap();
    }

    // #include cuda_fp16.h
    //
    // extern C __global__ void halfs(__half h) {
    //     assert(__habs(h - __float2half(1.234)) <= __float2half(1e-4));
    // }

    #[cfg(feature = "f16")]
    #[test]
    fn test_launch_with_half() {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();
        let halfs: Symbol<unsafe extern "C" fn(half::f16)> =
            unsafe { test_lib.get(b"halfs_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        unsafe {
            halfs(half::f16::from_f32(1.234));
        };
        dev.synchronize().unwrap();
    }

    // extern C __global__ void slow_worker(const float *data, const size_t len, float *out) {
    //     float tmp = 0.0;
    //     for(size_t i = 0; i < 1000000; i++) {
    //         tmp += data[i % len];
    //     }
    //     *out = tmp;
    // }
    //
    // extern "C" void slow_worker_wrapper(const float *data, const size_t len, float *out) {
    //     slow_worker<<<1, 1>>>(data, len, out);
    // }
    //
    // extern "C" void slow_worker_stream_wrapper(const float *data, const size_t len, float *out, cudaStream_t stream) {
    //     slow_worker<<<1, 1, 0, stream>>>(data, len, out);
    // }

    #[test]
    fn test_par_launch() -> Result<(), RuntimeError> {
        let test_lib =
            unsafe { libloading::Library::new(libloading::library_filename("testkernel")) }
                .unwrap();
        let slow_worker: Symbol<unsafe extern "C" fn(*const f32, usize, *mut f32)> =
            unsafe { test_lib.get(b"slow_worker_wrapper\0").unwrap() };

        let slow_worker_stream: Symbol<
            unsafe extern "C" fn(*const f32, usize, *mut f32, sys::cudaStream_t),
        > = unsafe { test_lib.get(b"slow_worker_stream_wrapper\0").unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        let slice = dev.alloc_zeros::<f32>(1000)?;
        let a = dev.alloc_zeros::<f32>(1)?;
        let b = dev.alloc_zeros::<f32>(1)?;

        let start = Instant::now();
        {
            // launch two kernels on the default stream
            unsafe { slow_worker(slice.device_ptr as *const f32, slice.len, a.device_ptr) };
            unsafe { slow_worker(slice.device_ptr as *const f32, slice.len, b.device_ptr) };
            dev.synchronize()?;
        }
        let double_launch_s = start.elapsed().as_secs_f64();

        let start = Instant::now();
        {
            // create a new stream & launch them concurrently
            let stream = dev.fork_default_stream()?;
            unsafe {
                slow_worker_stream(
                    slice.device_ptr as *const f32,
                    slice.len,
                    a.device_ptr,
                    stream.stream,
                )
            };
            unsafe {
                slow_worker_stream(
                    slice.device_ptr as *const f32,
                    slice.len,
                    b.device_ptr,
                    stream.stream,
                )
            };
            dev.wait_for(&stream)?;
            dev.synchronize()?;
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
}
