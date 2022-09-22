use std::rc::Rc;

use crate::arrays::NumElements;
use crate::cudarc::CudaRc;
use crate::curand::{result, sys};

pub struct CudaRng {
    gen: sys::curandGenerator_t,
}

impl Default for CudaRng {
    fn default() -> Self {
        Self::new(0).unwrap()
    }
}

impl CudaRng {
    pub fn new(seed: u64) -> Result<Self, result::CurandError> {
        let gen = result::create_generator()?;
        let mut rng = Self { gen };
        rng.set_seed(seed)?;
        Ok(rng)
    }

    pub fn set_seed(&mut self, seed: u64) -> Result<(), result::CurandError> {
        unsafe { result::set_seed(self.gen, seed) }
    }

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
    #[test]
    fn test_seed_reproducible() {
        todo!();
    }

    #[test]
    fn test_uniform_f32() {
        todo!();
    }

    #[test]
    fn test_uniform_f64() {
        todo!();
    }

    #[test]
    fn test_uniform_u32() {
        todo!();
    }

    #[test]
    fn test_uniform_u64() {
        todo!();
    }

    #[test]
    fn test_normal_f32() {
        todo!();
    }

    #[test]
    fn test_normal_f64() {
        todo!();
    }

    #[test]
    fn test_log_normal_f32() {
        todo!();
    }

    #[test]
    fn test_log_normal_f64() {
        todo!();
    }
}
