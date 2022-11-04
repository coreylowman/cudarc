use core::{
    mem::{size_of, MaybeUninit},
    ops::{Deref, DerefMut},
};

use super::sys::*;

use crate::{
    cudarc::CudaUniquePtr,
    driver::sys::CUdeviceptr,
    prelude::{CudaRc, IntoKernelParam},
};

/// Wrapper around [sys::cublasStatus_t].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CublasError(pub cublasStatus_t);

pub type CublasResult<T> = Result<T, CublasError>;
impl cublasStatus_t {
    /// Transforms into a [Result] of [CublasError]
    pub fn result(self) -> CublasResult<()> {
        match self {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            _ => Err(CublasError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CublasError {}

pub type CublasValueType = f32;

struct CublasVector<const S: usize>(CudaRc<[CublasValueType; S]>);
impl<const S: usize> Deref for CublasVector<S> {
    type Target = CudaRc<[CublasValueType; S]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<const S: usize> CublasVector<S> {
    unsafe fn new(device: CudaRc<[CublasValueType; S]>, vector: &[CublasValueType; S]) -> CublasResult<Self> {
        let mut s = Self::uninit(device);
        s.set(vector).map(|_| s)
    }
    
    unsafe fn uninit(device: CudaRc<[CublasValueType; S]>) -> Self {
        Self(device)
    }

    unsafe fn set(&mut self, vector: &[CublasValueType; S]) -> CublasResult<()> {
        cublasSetVector(
            S as _,
            size_of::<CublasValueType>() as _,
            vector.as_ptr() as *const _,
            1,
            self.into_kernel_param(),
            1,
        )
        .result()
    }

    unsafe fn get(&self, out: &mut [CublasValueType; S]) -> CublasResult<()> {
        cublasGetVector(
            S as _,
            size_of::<CublasValueType>() as _,
            self.into_kernel_param(),
            1,
            out.as_mut_ptr() as *mut _,
            1,
        )
        .result()
    }
}

/// A cublas Matrix with `R` rows and `C` columns in COLUMN-MAJOR!!! format.
pub struct CublasMatrix<const R: usize, const C: usize>(CudaRc<[[CublasValueType; R]; C]>);
impl<const R: usize, const C: usize> Deref for CublasMatrix<R, C> {
    type Target = CudaRc<[[CublasValueType; R]; C]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<const R: usize, const C: usize> DerefMut for CublasMatrix<R, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<const R: usize, const C: usize> CublasMatrix<R, C> {
    unsafe fn new(device: CudaRc<[[CublasValueType; R]; C]>, matrix: &[[CublasValueType; R]; C]) -> CublasResult<Self> {
        let mut s = Self::uninit(device);
        s.set(matrix).map(|_| s)
    }

    unsafe fn uninit(device: CudaRc<[[CublasValueType; R]; C]>) -> Self {
        Self(device)
    }

    unsafe fn set(&mut self, matrix: &[[CublasValueType; R]; C]) -> CublasResult<()> {
        cublasSetMatrix(
            R as _,
            C as _,
            size_of::<CublasValueType>() as _,
            matrix.as_ptr() as *const _,
            R as _,
            self.into_kernel_param(),
            R as _,
        )
        .result()
    }

    unsafe fn get(&self, out: &mut [[CublasValueType; R]; C]) -> CublasResult<()> {
        cublasGetMatrix(
            R as _,
            C as _,
            size_of::<CublasValueType>() as _,
            self.into_kernel_param(),
            R as _,
            out.as_mut_ptr() as *mut _,
            R as _,
        )
        .result()
    }
}

struct Handle(cublasHandle_t);
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.0);
        }
    }
}

impl Handle {
    unsafe fn create() -> CublasResult<Self> {
        let mut handle: Self = std::mem::zeroed();
        cublasCreate_v2(&mut handle.0 as *mut _).result()?;
        Ok(handle)
    }

    unsafe fn vm<const R: usize, const C: usize>(
        &mut self,
        matrix: &CublasMatrix<R, C>,
        vector: &CublasVector<C>,
        out: &mut CublasVector<R>,
    ) -> CublasResult<()> {
        cublasSgemv_v2(
            self.0,
            cublasOperation_t::CUBLAS_OP_N,
            R as _,
            C as _,
            &1.0f32 as *const CublasValueType,
            matrix.into_kernel_param() as *const _,
            R as _,
            vector.into_kernel_param() as *const _,
            1,
            &0.0f32 as *const CublasValueType,
            out.into_kernel_param() as *mut _,
            1,
        )
        .result()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::CudaDeviceBuilder;

    use super::*;

    #[test]
    fn test_create_vector() {
        let h_vector = [1.0f32, 2.0, 3.0];
        let mut h_out = [0.0; 3];
        unsafe {
            let device = CudaDeviceBuilder::new(0).build().unwrap();
            let ptr = device.alloc().unwrap();
            let d_vector = CublasVector::new(ptr, &h_vector).unwrap();
            d_vector.get(&mut h_out).unwrap();
        }
        assert_eq!(h_vector, h_out);
    }

    #[test]
    fn test_create_matrix() {
        let h_matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut h_out = [[0.0; 3]; 2];
        unsafe {
            let device = CudaDeviceBuilder::new(0).build().unwrap();
            let ptr = device.alloc().unwrap();
            let d_matrix = CublasMatrix::new(ptr, &h_matrix).unwrap();
            d_matrix.get(&mut h_out).unwrap();
        }
        assert_eq!(h_matrix, h_out);
    }

    #[test]
    fn test_gemv() {
        let h_vector = [1.0, 2.0, 3.0];
        let h_matrix = [[1.0, 2.0], [0.5, 1.0], [1.0 / 3.0, 1.0]];
        let h_expected = [3.0, 7.0];

        let mut h_out = [0.0; 2];
        unsafe {
            let device = CudaDeviceBuilder::new(0).build().unwrap();
            let mut cublas = Handle::create().unwrap();

            let ptr_m = device.alloc().unwrap();
            let d_matrix = CublasMatrix::new(ptr_m, &h_matrix).unwrap();

            let ptr_v = device.alloc().unwrap();
            let d_vector = CublasVector::new(ptr_v, &h_vector).unwrap();

            let ptr_o = device.alloc().unwrap();
            let mut d_out = CublasVector::uninit(ptr_o);

            
            cublas.vm(&d_matrix, &d_vector, &mut d_out).unwrap();

            d_out.get(&mut h_out).unwrap();
        }
        assert_eq!(h_out, h_expected);
    }
}
