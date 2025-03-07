//! Safe abstractions around [crate::curand::result] with [CudaRng].

use super::{result, sys};
use crate::driver::{CudaContext, CudaStream, DevicePtrMut};
use std::sync::Arc;

/// Host side RNG that can fill [CudaSlice] with random values.
///
/// 1. Create:
/// ```rust
/// # use cudarc::{driver::*, curand::*};
/// let device = CudaDevice::new(0).unwrap();
/// let rng = CudaRng::new(0, device).unwrap();
/// ```
/// 2. Fill device memory:
/// ```rust
/// # use cudarc::{driver::*, curand::*};
/// # let device = CudaDevice::new(0).unwrap();
/// # let rng = CudaRng::new(0, device.clone()).unwrap();
/// let mut a_dev = device.alloc_zeros::<f32>(10).unwrap();
/// rng.fill_with_uniform(&mut a_dev).unwrap();
/// ```
///
/// The three distributions are:
/// 1. Uniform - [CudaRng::fill_with_uniform()]
/// 2. Normal - [CudaRng::fill_with_normal()]
/// 3. LogNormal - [CudaRng::fill_with_log_normal()]
pub struct CudaRng {
    pub(crate) gen: sys::curandGenerator_t,
    #[allow(unused)]
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaRng {
    /// Constructs the RNG with the given `seed`. Requires the stream from [CudaDevice] to submit kernels.
    pub fn new(seed: u64, ctx: Arc<CudaContext>) -> Result<Self, result::CurandError> {
        ctx.bind_to_thread().unwrap();
        let gen = result::create_generator()?;
        let mut rng = Self { gen, ctx };
        rng.set_seed(seed)?;
        Ok(rng)
    }

    /// # Safety
    /// Users must ensure this stream is properly synchronized
    pub unsafe fn set_stream(
        &self,
        opt_stream: Option<&CudaStream>,
    ) -> Result<(), result::CurandError> {
        result::set_stream(
            self.gen,
            opt_stream
                .map(|stream| stream.cu_stream)
                .unwrap_or(std::ptr::null_mut()) as *mut _,
        )
    }

    /// Re-seed the RNG.
    pub fn set_seed(&mut self, seed: u64) -> Result<(), result::CurandError> {
        unsafe { result::set_seed(self.gen, seed) }
    }

    pub fn set_offset(&mut self, offset: u64) -> Result<(), result::CurandError> {
        unsafe { result::set_offset(self.gen, offset) }
    }

    /// Fill the [CudaSlice] with data from a `Uniform` distribution
    pub fn fill_with_uniform<T, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
    ) -> Result<(), result::CurandError>
    where
        sys::curandGenerator_t: result::UniformFill<T>,
    {
        unsafe { result::UniformFill::fill(self.gen, *dst.device_ptr_mut() as *mut T, dst.len()) }
    }

    /// Fill the [CudaSlice] with data from a `Normal(mean, std)` distribution.
    pub fn fill_with_normal<T, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
        mean: T,
        std: T,
    ) -> Result<(), result::CurandError>
    where
        sys::curandGenerator_t: result::NormalFill<T>,
    {
        unsafe {
            result::NormalFill::fill(
                self.gen,
                *dst.device_ptr_mut() as *mut T,
                dst.len(),
                mean,
                std,
            )
        }
    }

    /// Fill the `CudaRc` with data from a `LogNormal(mean, std)` distribution.
    pub fn fill_with_log_normal<T, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
        mean: T,
        std: T,
    ) -> Result<(), result::CurandError>
    where
        sys::curandGenerator_t: result::LogNormalFill<T>,
    {
        unsafe {
            result::LogNormalFill::fill(
                self.gen,
                *dst.device_ptr_mut() as *mut T,
                dst.len(),
                mean,
                std,
            )
        }
    }
}

impl Drop for CudaRng {
    fn drop(&mut self) {
        let gen = std::mem::replace(&mut self.gen, std::ptr::null_mut());
        if !gen.is_null() {
            unsafe { result::destroy_generator(gen) }.unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::{
        curand::result::{LogNormalFill, NormalFill, UniformFill},
        driver::*,
    };
    use std::vec::Vec;

    fn gen_uniform<T: ValidAsZeroBits + Clone + Default + Unpin + DeviceRepr>(
        seed: u64,
        n: usize,
    ) -> Vec<T>
    where
        super::sys::curandGenerator_t: UniformFill<T>,
    {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut a_dev = stream.alloc_zeros::<T>(n).unwrap();
        let rng = CudaRng::new(seed, ctx.clone()).unwrap();
        unsafe { rng.set_stream(Some(&stream)) }.unwrap();
        rng.fill_with_uniform(&mut a_dev).unwrap();
        stream.memcpy_dtov(&a_dev).unwrap()
    }

    fn gen_normal<T: ValidAsZeroBits + Clone + Default + Unpin + DeviceRepr>(
        seed: u64,
        n: usize,
        mean: T,
        std: T,
    ) -> Vec<T>
    where
        super::sys::curandGenerator_t: NormalFill<T>,
    {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut a_dev = stream.alloc_zeros::<T>(n).unwrap();
        let rng = CudaRng::new(seed, ctx.clone()).unwrap();
        unsafe { rng.set_stream(Some(&stream)) }.unwrap();
        rng.fill_with_normal(&mut a_dev, mean, std).unwrap();
        stream.memcpy_dtov(&a_dev).unwrap()
    }

    fn gen_log_normal<T: ValidAsZeroBits + Clone + Default + Unpin + DeviceRepr>(
        seed: u64,
        n: usize,
        mean: T,
        std: T,
    ) -> Vec<T>
    where
        super::sys::curandGenerator_t: LogNormalFill<T>,
    {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut a_dev = stream.alloc_zeros::<T>(n).unwrap();
        let rng = CudaRng::new(seed, ctx.clone()).unwrap();
        unsafe { rng.set_stream(Some(&stream)) }.unwrap();
        rng.fill_with_log_normal(&mut a_dev, mean, std).unwrap();
        stream.memcpy_dtov(&a_dev).unwrap()
    }

    #[test]
    fn test_seed_reproducible() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let mut a_dev = stream.alloc_zeros::<f32>(10).unwrap();
        let mut b_dev = a_dev.clone();

        let a_rng = CudaRng::new(0, ctx.clone()).unwrap();
        let b_rng = CudaRng::new(0, ctx.clone()).unwrap();

        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        b_rng.fill_with_uniform(&mut b_dev).unwrap();

        let a_host = stream.memcpy_dtov(&a_dev).unwrap();
        let b_host = stream.memcpy_dtov(&b_dev).unwrap();
        assert_eq!(a_host, b_host);
    }

    #[test]
    fn test_different_seeds_neq() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let mut a_dev = stream.alloc_zeros::<f32>(10).unwrap();
        let mut b_dev = a_dev.clone();

        let a_rng = CudaRng::new(0, ctx.clone()).unwrap();
        let b_rng = CudaRng::new(1, ctx.clone()).unwrap();

        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        b_rng.fill_with_uniform(&mut b_dev).unwrap();

        let a_host = stream.memcpy_dtov(&a_dev).unwrap();
        let b_host = stream.memcpy_dtov(&b_dev).unwrap();
        assert_ne!(a_host, b_host);
    }

    #[test]
    fn test_set_offset() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let mut a_dev = stream.alloc_zeros::<f32>(10).unwrap();
        let mut a_rng = CudaRng::new(0, ctx.clone()).unwrap();

        a_rng.set_seed(42).unwrap();
        a_rng.set_offset(0).unwrap();
        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        let a_host = stream.memcpy_dtov(&a_dev).unwrap();

        a_rng.set_seed(42).unwrap();
        a_rng.set_offset(0).unwrap();
        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        let b_host = stream.memcpy_dtov(&a_dev).unwrap();

        assert_eq!(a_host, b_host);
    }

    const N: usize = 1000;

    #[test]
    fn test_uniform_f32() {
        let a = gen_uniform::<f32>(0, N);
        for i in 0..N {
            assert!(0.0 < a[i] && a[i] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_f64() {
        let a = gen_uniform::<f64>(0, N);
        for i in 0..N {
            assert!(0.0 < a[i] && a[i] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_u32() {
        let a = gen_uniform::<u32>(0, N);
        for i in 0..N {
            assert!(a[i] > 0);
        }
    }

    #[test]
    fn test_normal_f32() {
        let a = gen_normal::<f32>(0, N, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_normal::<f32>(0, N, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }

    #[test]
    fn test_normal_f64() {
        let a = gen_normal::<f64>(0, N, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_normal::<f64>(0, N, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }

    #[test]
    fn test_log_normal_f32() {
        let a = gen_log_normal::<f32>(0, N, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_log_normal::<f32>(0, N, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }

    #[test]
    fn test_log_normal_f64() {
        let a = gen_log_normal::<f64>(0, N, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_log_normal::<f64>(0, N, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }
}
