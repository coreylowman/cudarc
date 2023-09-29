//! Safe abstractions around [crate::cublaslt::result] for doing matmul.

use core::ffi::c_int;
use super::{result, result::CublasError, sys};
use crate::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DriverError};
use core::mem;
use std::println;
use std::sync::Arc;
use crate::driver::sys::CUdevice_attribute;

/// Wrapper around [sys::cublasLtHandle_t]
///
/// 1. Create with [CudaBlasLT::new()]
/// 2. Execute matmul kernel with matmul. f32 is supported. f16 and bf16 are supported
/// if feature `half` is activated
#[derive(Debug)]
pub struct CudaBlasLT {
    pub(crate) handle: sys::cublasLtHandle_t,
}

unsafe impl Send for CudaBlasLT {}

unsafe impl Sync for CudaBlasLT {}

impl CudaBlasLT {
    /// Creates a new cublasLt handle.
    pub fn new() -> Result<Self, CublasError> {
        let handle = result::create_handle()?;
        Ok(Self { handle })
    }

    /// Returns a reference to the underlying cublas handle.
    pub fn handle(&self) -> &sys::cublasLtHandle_t {
        &self.handle
    }
}

impl Drop for CudaBlasLT {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}

#[derive(Debug, Clone)]
pub struct Workspace {
    pub(crate) buffer: CudaSlice<u8>,
    pub(crate) size: usize,
}

impl Workspace {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, DriverError> {
        device.bind_to_thread()?;

        let major = device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let workspace_size = if major >= 9 { 33_554_432 } else { 4_194_304 };

        let buffer = unsafe { device.alloc::<u8>(workspace_size)? };
        Ok(Self { buffer, size: workspace_size })
    }
}

/// Configuration for [Matmul]
#[derive(Debug, Copy, Clone)]
pub struct MatmulConfig<T> {
    pub transa: bool,
    pub transb: bool,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub alpha: T,
    pub lda: i64,
    pub ldb: i64,
    pub beta: T,
    pub ldc: i64,
}

/// Matrix matrix multiplication with elements of type `T`.
pub trait Matmul<T> {
    unsafe fn matmul<A: DevicePtr<T>, B: DevicePtr<T>, C: DevicePtrMut<T>>(
        &self,
        cfg: MatmulConfig<T>,
        workspace: Workspace,
        stream: CudaStream,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError>;
}

impl Matmul<f32> for CudaBlasLT {
    unsafe fn matmul<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        cfg: MatmulConfig<f32>,
        mut workspace: Workspace,
        stream: CudaStream,
        a: &A, b: &B, c: &mut C,
    ) -> Result<(), CublasError> {
        let (a_rows, a_cols) = if cfg.transa {
            (cfg.k, cfg.m)
        } else {
            (cfg.m, cfg.k)
        };
        let (b_rows, b_cols) = if cfg.transb {
            (cfg.n, cfg.k)
        } else {
            (cfg.k, cfg.n)
        };

        let a_layout = result::create_matrix_layout(
            sys::cudaDataType_t::CUDA_R_32F,
            a_rows,
            a_cols,
            cfg.lda,
        )?;
        let b_layout = result::create_matrix_layout(
            sys::cudaDataType_t::CUDA_R_32F,
            b_rows,
            b_cols,
            cfg.ldb,
        )?;
        let c_layout = result::create_matrix_layout(
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.m,
            cfg.n,
            cfg.ldc,
        )?;

        let matmul_desc = result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            sys::cudaDataType_t::CUDA_R_32F,
        )?;

        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            (&cfg.transa) as *const _ as *const _,
            mem::size_of::<u32>(),
        )?;

        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            (&cfg.transb) as *const _ as *const _,
            mem::size_of::<u32>(),
        )?;

        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            (&sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT) as *const _ as *const _,
            mem::size_of::<u32>(),
        )?;

        let matmul_pref = result::create_matmul_pref()?;

        result::set_matmul_pref_attribute(
            matmul_pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&workspace.size) as *const _ as *const _,
            mem::size_of::<usize>(),
        )?;

        let heuristic = result::get_matmul_algo_heuristic(self.handle,
                                                          matmul_desc,
                                                          a_layout,
                                                          b_layout,
                                                          c_layout,
                                                          c_layout,
                                                          matmul_pref)?;

        result::matmul(
            self.handle,
            matmul_desc,
            (&cfg.alpha) as *const _ as *const _,
            (&cfg.beta) as *const _ as *const _,
            *a.device_ptr() as *const _,
            a_layout,
            *b.device_ptr() as *const _,
            b_layout,
            *c.device_ptr_mut() as *const _,
            c_layout,
            *c.device_ptr_mut() as *mut _,
            c_layout,
            (&heuristic.algo) as *const _,
            workspace.buffer.device_ptr_mut() as *mut _ as *mut _,
            workspace.size,
            stream.stream as *mut _,
        )
    }
}


#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use std::ffi::CString;
    use super::*;

    fn matmul_truth<T, const M: usize, const N: usize, const K: usize>(
        alpha: T,
        a: &[[T; K]; M],
        b: &[[T; N]; K],
        beta: T,
        c: &mut [[T; N]; M],
    ) where
        T: Copy + Clone + std::ops::AddAssign + std::ops::MulAssign + std::ops::Mul<T, Output=T>,
    {
        for m in 0..M {
            for n in 0..N {
                c[m][n] *= beta;
            }
        }
        for m in 0..M {
            for n in 0..N {
                for k in 0..K {
                    c[m][n] += alpha * a[m][k] * b[k][n];
                }
            }
        }
    }


    #[test]
    fn test_sgemm() {
        unsafe { sys::cublasLtLoggerSetLevel(4).result().unwrap() };
        unsafe { sys::cublasLtLoggerOpenFile(CString::new("log").unwrap().as_ptr()).result().unwrap() };

        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlasLT::new().unwrap();
        let workspace = Workspace::new(dev.clone()).unwrap();
        const M: usize = 3;
        const K: usize = 4;
        const N: usize = 5;
        let a: [[f32; K]; M] = [
            [-0.5944882, 1.8055636, 0.52204555, -0.00397902],
            [-0.38346434, -0.38013917, 0.4198623, -0.22479166],
            [-1.6661372, -0.4568837, -0.9043474, 0.39125723],
        ];
        let b: [[f32; N]; K] = [
            [1.1292169, -0.13450263, 0.62789696, -0.5685516, 0.21946938],
            [1.0585804, -0.39789402, 0.90205914, 0.989318, -0.3443096],
            [1.3412506, 0.3059701, -0.9714474, -0.36113533, -1.6809629],
            [3.4746711, -1.0930681, 0.16502666, -0.59988785, 0.41375792],
        ];
        let mut c: [[f32; N]; M] = [[0.0; N]; M];
        matmul_truth(1.0, &a, &b, 0.0, &mut c);

        #[rustfmt::skip]
        let a_dev = dev.htod_sync_copy::<f32>(&[
            -0.5944882, 1.8055636, 0.52204555, -0.00397902,
            -0.38346434, -0.38013917, 0.4198623, -0.22479166,
            -1.6661372, -0.4568837, -0.9043474, 0.39125723,
        ]).unwrap();
        #[rustfmt::skip]
        let b_dev = dev.htod_sync_copy::<f32>(&[
            1.1292169, -0.13450263, 0.62789696, -0.5685516, 0.21946938,
            1.0585804, -0.39789402, 0.90205914, 0.989318, -0.3443096,
            1.3412506, 0.3059701, -0.9714474, -0.36113533, -1.6809629,
            3.4746711, -1.0930681, 0.16502666, -0.59988785, 0.41375792,
        ]).unwrap();
        let mut c_dev = dev.alloc_zeros::<f32>(M * N).unwrap();
        unsafe {
            blas.matmul(
                MatmulConfig {
                    transa: false,
                    transb: false,
                    m: N as u64,
                    n: M as u64,
                    k: K as u64,
                    alpha: 1.0,
                    lda: N as i64,
                    ldb: K as i64,
                    beta: 0.0,
                    ldc: N as i64,
                },
                workspace,
                dev.fork_default_stream().unwrap(),
                &b_dev,
                &a_dev,
                &mut c_dev,
            )
        }
            .unwrap();

        let c_host = dev.sync_reclaim(c_dev).unwrap();
        for m in 0..M {
            for n in 0..N {
                assert!((c_host[m * N + n] - c[m][n]) <= 1e-6);
            }
        }
    }
}