//! Safe abstractions around [crate::cublas::result] for doing gemm and gemv.

use crate::cublas::{result, result::CublasError, sys};
use crate::device::{CudaDevice, CudaSlice};
use core::ffi::c_int;
use std::sync::Arc;

/// Wrapper around [sys::cublasHandle_t]
///
/// 1. Create with [CudaBlas::new()]
/// 2. Execute gemm/gemv kernels with [Gemv] and [Gemm]. Both f32 and f64 are supported
///    for both
///
/// Note: This maintains a instance of [`Arc<CudaDevice>`], so will prevent the device
/// from being dropped.
#[derive(Debug)]
pub struct CudaBlas {
    pub(crate) handle: sys::cublasHandle_t,
    pub(crate) device: Arc<CudaDevice>,
}

impl CudaBlas {
    /// Creates a new cublas handle and sets the stream to the `device`'s stream.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CublasError> {
        let handle = result::create_handle()?;
        let blas = Self { handle, device };
        unsafe { result::set_stream(handle, blas.device.cu_stream as *mut _) }?;
        Ok(blas)
    }
}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}

/// Matrix vector multiplication with elements of type `T`
pub trait Gemv<T> {
    unsafe fn gemv_async(
        &self,
        trans: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        alpha: T,
        a: &CudaSlice<T>,
        lda: c_int,
        x: &CudaSlice<T>,
        incx: c_int,
        beta: T,
        y: &mut CudaSlice<T>,
        incy: c_int,
    ) -> Result<(), CublasError>;
}

impl Gemv<f32> for CudaBlas {
    unsafe fn gemv_async(
        &self,
        trans: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        alpha: f32,
        a: &CudaSlice<f32>,
        lda: c_int,
        x: &CudaSlice<f32>,
        incx: c_int,
        beta: f32,
        y: &mut CudaSlice<f32>,
        incy: c_int,
    ) -> Result<(), CublasError> {
        result::sgemv(
            self.handle,
            trans,
            m,
            n,
            (&alpha) as *const _,
            a.cu_device_ptr as *const _,
            lda,
            x.cu_device_ptr as *const _,
            incx,
            (&beta) as *const _,
            y.cu_device_ptr as *mut _,
            incy,
        )
    }
}

impl Gemv<f64> for CudaBlas {
    unsafe fn gemv_async(
        &self,
        trans: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        alpha: f64,
        a: &CudaSlice<f64>,
        lda: c_int,
        x: &CudaSlice<f64>,
        incx: c_int,
        beta: f64,
        y: &mut CudaSlice<f64>,
        incy: c_int,
    ) -> Result<(), CublasError> {
        result::dgemv(
            self.handle,
            trans,
            m,
            n,
            (&alpha) as *const _,
            a.cu_device_ptr as *const _,
            lda,
            x.cu_device_ptr as *const _,
            incx,
            (&beta) as *const _,
            y.cu_device_ptr as *mut _,
            incy,
        )
    }
}

/// Matrix matrix multiplication with elements of type `T`
pub trait Gemm<T> {
    unsafe fn gemm_async(
        &self,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: T,
        a: &CudaSlice<T>,
        lda: c_int,
        b: &CudaSlice<T>,
        ldb: c_int,
        beta: T,
        c: &mut CudaSlice<T>,
        ldc: c_int,
    ) -> Result<(), CublasError>;
}

impl Gemm<f32> for CudaBlas {
    unsafe fn gemm_async(
        &self,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: &CudaSlice<f32>,
        lda: c_int,
        b: &CudaSlice<f32>,
        ldb: c_int,
        beta: f32,
        c: &mut CudaSlice<f32>,
        ldc: c_int,
    ) -> Result<(), CublasError> {
        result::sgemm(
            self.handle,
            transa,
            transb,
            m,
            n,
            k,
            (&alpha) as *const _,
            a.cu_device_ptr as *const _,
            lda,
            b.cu_device_ptr as *const _,
            ldb,
            (&beta) as *const _,
            c.cu_device_ptr as *mut _,
            ldc,
        )
    }
}

impl Gemm<f64> for CudaBlas {
    unsafe fn gemm_async(
        &self,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: &CudaSlice<f64>,
        lda: c_int,
        b: &CudaSlice<f64>,
        ldb: c_int,
        beta: f64,
        c: &mut CudaSlice<f64>,
        ldc: c_int,
    ) -> Result<(), CublasError> {
        result::dgemm(
            self.handle,
            transa,
            transb,
            m,
            n,
            k,
            (&alpha) as *const _,
            a.cu_device_ptr as *const _,
            lda,
            b.cu_device_ptr as *const _,
            ldb,
            (&beta) as *const _,
            c.cu_device_ptr as *mut _,
            ldc,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CudaDeviceBuilder;

    fn gemv_truth<T, const M: usize, const N: usize>(
        alpha: T,
        a: &[[T; N]; M],
        x: &[T; N],
        beta: T,
        y: &mut [T; M],
    ) where
        T: Copy + Clone + std::ops::AddAssign + std::ops::MulAssign + std::ops::Mul<T, Output = T>,
    {
        for m in 0..M {
            y[m] *= beta;
        }
        for m in 0..M {
            for n in 0..N {
                y[m] += alpha * a[m][n] * x[n];
            }
        }
    }

    fn gemm_truth<T, const M: usize, const N: usize, const K: usize>(
        alpha: T,
        a: &[[T; K]; M],
        b: &[[T; N]; K],
        beta: T,
        c: &mut [[T; N]; M],
    ) where
        T: Copy + Clone + std::ops::AddAssign + std::ops::MulAssign + std::ops::Mul<T, Output = T>,
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
    fn test_sgemv() {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();
        const M: usize = 2;
        const N: usize = 5;
        let a: [[f32; N]; M] = [
            [0.93147761, 0.10300648, -0.62077409, 1.52707517, 0.02598040],
            [0.16820757, -0.94463515, -1.38501012, 1.06005239, 1.51240087],
        ];
        #[rustfmt::skip]
        let b: [f32; N] = [-1.34419966, 1.39655411, -0.89106345, 0.21196432, -0.95535654];
        let mut c: [f32; M] = [1.0; M];
        gemv_truth(1.0, &a, &b, 0.0, &mut c);

        #[rustfmt::skip]
        let a_dev = dev.sync_copy(&[
            0.93147761, 0.10300648, -0.62077409, 1.52707517, 0.02598040,
            0.16820757, -0.94463515, -1.38501012, 1.06005239, 1.51240087,
        ]).unwrap();
        let b_dev = dev.sync_copy(&b).unwrap();
        let mut c_dev = dev.alloc_zeros_async(M).unwrap();
        unsafe {
            blas.gemv_async(
                sys::cublasOperation_t::CUBLAS_OP_T,
                N as i32,
                M as i32,
                1.0,
                &a_dev,
                N as i32,
                &b_dev,
                1,
                0.0,
                &mut c_dev,
                1,
            )
        }
        .unwrap();

        let c_host = dev.sync_release(c_dev).unwrap();
        for i in 0..M {
            assert!((c_host[i] - c[i]).abs() <= 1e-8);
        }
    }

    #[test]
    fn test_dgemv() {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();
        const M: usize = 8;
        const N: usize = 3;
        let a: [[f64; N]; M] = [
            [0.96151888, -0.36771390, 0.94069099],
            [2.20621538, -0.16479775, -1.78425562],
            [0.41080803, -0.56567699, -0.72781092],
            [-0.65718418, -0.14466463, 0.63984287],
            [0.20309605, 0.40480086, -1.57559848],
            [0.85628128, -0.51614553, -1.15904427],
            [-1.84258616, 0.24096519, -0.04563522],
            [-0.53364468, -1.07902217, 0.46823528],
        ];
        #[rustfmt::skip]
        let b: [f64; N] = [ 0.39745075, -1.06677043, -1.18272650];
        let mut c: [f64; M] = [1.0; M];
        gemv_truth(1.0, &a, &b, 0.0, &mut c);

        #[rustfmt::skip]
        let a_dev = dev.sync_copy(&[
            0.96151888, -0.36771390, 0.94069099,
            2.20621538, -0.16479775, -1.78425562,
            0.41080803, -0.56567699, -0.72781092,
            -0.65718418, -0.14466463, 0.63984287,
            0.20309605, 0.40480086, -1.57559848,
            0.85628128, -0.51614553, -1.15904427,
            -1.84258616, 0.24096519, -0.04563522,
            -0.53364468, -1.07902217, 0.46823528,
        ]).unwrap();
        let b_dev = dev.sync_copy(&b).unwrap();
        let mut c_dev = dev.alloc_zeros_async(M).unwrap();
        unsafe {
            blas.gemv_async(
                sys::cublasOperation_t::CUBLAS_OP_T,
                N as i32,
                M as i32,
                1.0,
                &a_dev,
                N as i32,
                &b_dev,
                1,
                0.0,
                &mut c_dev,
                1,
            )
        }
        .unwrap();

        let c_host = dev.sync_release(c_dev).unwrap();
        for i in 0..M {
            assert!((c_host[i] - c[i]).abs() <= 1e-8);
        }
    }

    #[test]
    fn test_sgemm() {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();
        const M: usize = 3;
        const K: usize = 4;
        const N: usize = 5;
        let a: [[f32; K]; M] = [
            [-0.59448820, 1.80556369, 0.52204555, -0.00397902],
            [-0.38346434, -0.38013917, 0.41986239, -0.22479166],
            [-1.66613722, -0.45688370, -0.90434748, 0.39125723],
        ];
        let b: [[f32; N]; K] = [
            [1.12921691, -0.13450263, 0.62789696, -0.56855160, 0.21946938],
            [1.05858040, -0.39789402, 0.90205914, 0.98931807, -0.34430960],
            [
                1.34125066,
                0.30597019,
                -0.97144741,
                -0.36113533,
                -1.68096292,
            ],
            [3.47467113, -1.09306812, 0.16502666, -0.59988785, 0.41375792],
        ];
        let mut c: [[f32; N]; M] = [[0.0; N]; M];
        gemm_truth(1.0, &a, &b, 0.0, &mut c);

        #[rustfmt::skip]
        let a_dev = dev.sync_copy::<f32>(&[
            -0.59448820, 1.80556369, 0.52204555, -0.00397902,
            -0.38346434, -0.38013917, 0.41986239, -0.22479166,
            -1.66613722, -0.45688370, -0.90434748, 0.39125723,
        ]).unwrap();
        #[rustfmt::skip]
        let b_dev = dev.sync_copy::<f32>(&[
            1.12921691, -0.13450263, 0.62789696, -0.56855160, 0.21946938,
            1.05858040, -0.39789402, 0.90205914, 0.98931807, -0.34430960,
            1.34125066, 0.30597019, -0.97144741, -0.36113533, -1.68096292,
            3.47467113, -1.09306812, 0.16502666, -0.59988785, 0.41375792,
        ]).unwrap();
        let mut c_dev = dev.alloc_zeros_async::<f32>(M * N).unwrap();
        unsafe {
            blas.gemm_async(
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                N as i32,
                M as i32,
                K as i32,
                1.0,
                &b_dev,
                N as i32,
                &a_dev,
                K as i32,
                0.0,
                &mut c_dev,
                N as i32,
            )
        }
        .unwrap();

        let c_host = dev.sync_release(c_dev).unwrap();
        for m in 0..M {
            for n in 0..N {
                assert!((c_host[m * N + n] - c[m][n]) <= 1e-6);
            }
        }
    }

    #[test]
    fn test_dgemm() {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();
        const M: usize = 4;
        const K: usize = 3;
        const N: usize = 2;
        let a: [[f64; K]; M] = [
            [-0.70925030, -1.01357541, -0.64827034],
            [2.18493467, -0.61584842, -1.43844327],
            [-1.34792593, 0.68840750, -0.48057214],
            [1.22180992, 1.16245157, 0.01253436],
        ];
        let b: [[f64; N]; K] = [
            [-0.72735474, 1.35931170],
            [1.71798307, -0.13296247],
            [0.26855612, -1.95189980],
        ];
        let mut c: [[f64; N]; M] = [[0.0; N]; M];
        gemm_truth(1.0, &a, &b, 0.0, &mut c);

        #[rustfmt::skip]
        let a_dev = dev.sync_copy::<f64>(&[
            -0.70925030, -1.01357541, -0.64827034,
            2.18493467, -0.61584842, -1.43844327,
            -1.34792593, 0.68840750, -0.48057214,
            1.22180992, 1.16245157, 0.01253436,
        ]).unwrap();
        #[rustfmt::skip]
        let b_dev = dev.sync_copy::<f64>(&[
            -0.72735474, 1.35931170,
            1.71798307, -0.13296247,
            0.26855612, -1.95189980,
        ]).unwrap();
        let mut c_dev = dev.alloc_zeros_async::<f64>(M * N).unwrap();
        unsafe {
            blas.gemm_async(
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                N as i32,
                M as i32,
                K as i32,
                1.0,
                &b_dev,
                N as i32,
                &a_dev,
                K as i32,
                0.0,
                &mut c_dev,
                N as i32,
            )
        }
        .unwrap();

        let c_host = dev.sync_release(c_dev).unwrap();
        for m in 0..M {
            for n in 0..N {
                assert!((c_host[m * N + n] - c[m][n]) <= 1e-10);
            }
        }
    }
}
