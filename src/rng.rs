//! Safe abstractions around [crate::curand::result] with [CudaRng].

use crate::arrays::NumElements;
use crate::cudarc::{CudaDevice, CudaRc};
use crate::curand::{result, sys};
use std::rc::Rc;

/// Host side RNG that can fill [CudaRc] with random values.
///
/// 1. Create:
/// ```rust
/// # use cudarc::{prelude::*, rng::*};
/// let device = CudaDeviceBuilder::new(0).build().unwrap();
/// let rng = CudaRng::new(0, device).unwrap();
/// ```
/// 2. Fill device memory:
/// ```rust
/// # use cudarc::{prelude::*, rng::*};
/// # let device = CudaDeviceBuilder::new(0).build().unwrap();
/// # let rng = CudaRng::new(0, device.clone()).unwrap();
/// let mut a_dev = device.alloc_zeros::<[f32; 10]>().unwrap();
/// rng.fill_with_uniform(&mut a_dev).unwrap();
/// ```
///
/// The three distributions are:
/// 1. Uniform - [CudaRng::fill_with_uniform()]
/// 2. Normal - [CudaRng::fill_with_normal()]
/// 3. LogNormal - [CudaRng::fill_with_log_normal()]
pub struct CudaRng {
    gen: sys::curandGenerator_t,
    device: Rc<CudaDevice>,
}

impl CudaRng {
    /// Constructs the RNG with the given `seed`. Requires the stream from [CudaDevice] to submit kernels.
    pub fn new(seed: u64, device: Rc<CudaDevice>) -> Result<Self, result::CurandError> {
        let gen = result::create_generator()?;
        let mut rng = Self { gen, device };
        rng.set_seed(seed)?;
        unsafe { result::set_stream(rng.gen, rng.device.cu_stream as *mut _) }?;
        Ok(rng)
    }

    /// Re-seed the RNG.
    pub fn set_seed(&mut self, seed: u64) -> Result<(), result::CurandError> {
        unsafe { result::set_seed(self.gen, seed) }
    }

    /// Fill the [CudaRc] with data from a `Uniform` distribution
    pub fn fill_with_uniform<T>(&self, t: &mut CudaRc<T>) -> Result<(), result::CurandError>
    where
        T: NumElements,
        sys::curandGenerator_t: result::UniformFill<T::Dtype>,
    {
        let out = Rc::make_mut(&mut t.t_cuda);
        unsafe {
            result::UniformFill::<T::Dtype>::fill(
                self.gen,
                out.cu_device_ptr as *mut T::Dtype,
                T::NUMEL,
            )
        }
    }

    /// Fill the [CudaRc] with data from a `Normal(mean, std)` distribution.
    pub fn fill_with_normal<T>(
        &self,
        t: &mut CudaRc<T>,
        mean: T::Dtype,
        std: T::Dtype,
    ) -> Result<(), result::CurandError>
    where
        T: NumElements,
        sys::curandGenerator_t: result::NormalFill<T::Dtype>,
    {
        let out = Rc::make_mut(&mut t.t_cuda);
        unsafe {
            result::NormalFill::<T::Dtype>::fill(
                self.gen,
                out.cu_device_ptr as *mut T::Dtype,
                T::NUMEL,
                mean,
                std,
            )
        }
    }

    /// Fill the `CudaRc` with data from a `LogNormal(mean, std)` distribution.
    pub fn fill_with_log_normal<T>(
        &self,
        t: &mut CudaRc<T>,
        mean: T::Dtype,
        std: T::Dtype,
    ) -> Result<(), result::CurandError>
    where
        T: NumElements,
        sys::curandGenerator_t: result::LogNormalFill<T::Dtype>,
    {
        let out = Rc::make_mut(&mut t.t_cuda);
        unsafe {
            result::LogNormalFill::<T::Dtype>::fill(
                self.gen,
                out.cu_device_ptr as *mut T::Dtype,
                T::NUMEL,
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
    use super::*;
    use crate::{
        cudarc::ValidAsZeroBits,
        curand::result::{LogNormalFill, NormalFill, UniformFill},
        prelude::*, arrays::Array,
    };

    fn gen_uniform<T: Clone + NumElements + ValidAsZeroBits>(seed: u64) -> Rc<T>
    where
        sys::curandGenerator_t: UniformFill<T::Dtype>,
    {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();
        let mut a_dev = dev.alloc_zeros::<T>().unwrap();
        let rng = CudaRng::new(seed, dev).unwrap();
        rng.fill_with_uniform(&mut a_dev).unwrap();
        a_dev.into_host().unwrap()
    }

    fn gen_normal<T: Clone + NumElements + ValidAsZeroBits>(
        seed: u64,
        mean: T::Dtype,
        std: T::Dtype,
    ) -> Rc<T>
    where
        sys::curandGenerator_t: NormalFill<T::Dtype>,
    {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();
        let mut a_dev = dev.alloc_zeros::<T>().unwrap();
        let rng = CudaRng::new(seed, dev).unwrap();
        rng.fill_with_normal(&mut a_dev, mean, std).unwrap();
        a_dev.into_host().unwrap()
    }

    fn gen_log_normal<T: Clone + NumElements + ValidAsZeroBits>(
        seed: u64,
        mean: T::Dtype,
        std: T::Dtype,
    ) -> Rc<T>
    where
        sys::curandGenerator_t: LogNormalFill<T::Dtype>,
    {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();
        let mut a_dev = dev.alloc_zeros::<T>().unwrap();
        let rng = CudaRng::new(seed, dev).unwrap();
        rng.fill_with_log_normal(&mut a_dev, mean, std).unwrap();
        a_dev.into_host().unwrap()
    }

    #[test]
    fn test_rc_counts() {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();
        assert_eq!(Rc::strong_count(&dev), 1);
        let a_rng = CudaRng::new(0, dev.clone()).unwrap();
        assert_eq!(Rc::strong_count(&dev), 2);
        let a_dev = dev.alloc_zeros::<[f32; 10]>().unwrap();
        assert_eq!(Rc::strong_count(&dev), 3);
        drop(a_rng);
        assert_eq!(Rc::strong_count(&dev), 2);
        drop(a_dev);
        assert_eq!(Rc::strong_count(&dev), 1);
    }

    #[test]
    fn test_seed_reproducible() {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();

        let mut a_dev = dev.alloc_zeros::<[f32; 10]>().unwrap();
        let mut b_dev = a_dev.clone();

        let a_rng = CudaRng::new(0, dev.clone()).unwrap();
        let b_rng = CudaRng::new(0, dev).unwrap();

        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        b_rng.fill_with_uniform(&mut b_dev).unwrap();

        let a_host = a_dev.into_host().unwrap();
        let b_host = b_dev.into_host().unwrap();
        assert_eq!(a_host.as_ref(), b_host.as_ref());
    }

    #[test]
    fn test_different_seeds_neq() {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();

        let mut a_dev = dev.alloc_zeros::<[f32; 10]>().unwrap();
        let mut b_dev = a_dev.clone();

        let a_rng = CudaRng::new(0, dev.clone()).unwrap();
        let b_rng = CudaRng::new(1, dev).unwrap();

        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        b_rng.fill_with_uniform(&mut b_dev).unwrap();

        let a_host = a_dev.into_host().unwrap();
        let b_host = b_dev.into_host().unwrap();
        assert_ne!(a_host.as_ref(), b_host.as_ref());
    }

    const N: usize = 999;

    #[test]
    fn test_uniform_f32() {
        let a = gen_uniform::<Array<f32, N>>(0);
        for i in 0..N {
            assert!(0.0 < a[i] && a[i] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_f64() {
        let a = gen_uniform::<Array<f64, N>>(0);
        for i in 0..N {
            assert!(0.0 < a[i] && a[i] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_u32() {
        let a = gen_uniform::<[u32; N]>(0);
        for i in 0..N {
            assert!(a[i] > 0);
        }
    }

    #[test]
    fn test_normal_f32() {
        let a = gen_normal::<Array<f32, N>>(0, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_normal::<Array<f32, N>>(0, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }

    #[test]
    fn test_normal_f64() {
        let a = gen_normal::<Array<f64, N>>(0, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_normal::<Array<f64, N>>(0, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }

    #[test]
    fn test_log_normal_f32() {
        let a = gen_log_normal::<Array<f32, N>>(0, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_log_normal::<Array<f32, N>>(0, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }

    #[test]
    fn test_log_normal_f64() {
        let a = gen_log_normal::<Array<f64, N>>(0, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_log_normal::<Array<f64, N>>(0, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }
}
