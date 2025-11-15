use super::{result, result::CublasError, sys};
use crate::cublas::CudaBlas;
use crate::driver::{DevicePtr, DevicePtrMut};
use core::ffi::c_int;

/// Configuration for [Gemv]
#[derive(Debug, Copy, Clone)]
pub struct GemvConfig<T> {
    pub trans: sys::cublasOperation_t,
    pub m: c_int,
    pub n: c_int,
    pub alpha: T,
    pub lda: c_int,
    pub incx: c_int,
    pub beta: T,
    pub incy: c_int,
}

/// Matrix vector multiplication with elements of type `T`
pub trait Gemv<T> {
    /// Matrix vector multiplication.
    ///
    /// # Safety
    /// This is unsafe because improper arguments may lead to invalid
    /// memory accesses.
    unsafe fn gemv<A: DevicePtr<T>, X: DevicePtr<T>, Y: DevicePtrMut<T>>(
        &self,
        cfg: GemvConfig<T>,
        a: &A,
        x: &X,
        y: &mut Y,
    ) -> Result<(), CublasError>;
}

impl Gemv<f32> for CudaBlas {
    unsafe fn gemv<A: DevicePtr<f32>, X: DevicePtr<f32>, Y: DevicePtrMut<f32>>(
        &self,
        cfg: GemvConfig<f32>,
        a: &A,
        x: &X,
        y: &mut Y,
    ) -> Result<(), CublasError> {
        let (a, _record_a) = a.device_ptr(&self.stream);
        let (x, _record_x) = x.device_ptr(&self.stream);
        let (y, _record_y) = y.device_ptr_mut(&self.stream);
        result::sgemv(
            self.handle,
            cfg.trans,
            cfg.m,
            cfg.n,
            (&cfg.alpha) as *const _,
            a as *const _,
            cfg.lda,
            x as *const _,
            cfg.incx,
            (&cfg.beta) as *const _,
            y as *mut _,
            cfg.incy,
        )
    }
}

impl Gemv<f64> for CudaBlas {
    unsafe fn gemv<A: DevicePtr<f64>, X: DevicePtr<f64>, Y: DevicePtrMut<f64>>(
        &self,
        cfg: GemvConfig<f64>,
        a: &A,
        x: &X,
        y: &mut Y,
    ) -> Result<(), CublasError> {
        let (a, _record_a) = a.device_ptr(&self.stream);
        let (x, _record_x) = x.device_ptr(&self.stream);
        let (y, _record_y) = y.device_ptr_mut(&self.stream);
        result::dgemv(
            self.handle,
            cfg.trans,
            cfg.m,
            cfg.n,
            (&cfg.alpha) as *const _,
            a as *const _,
            cfg.lda,
            x as *const _,
            cfg.incx,
            (&cfg.beta) as *const _,
            y as *mut _,
            cfg.incy,
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use crate::driver::CudaContext;

    use super::*;

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

    #[test]
    fn test_sgemv() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).unwrap();
        const M: usize = 2;
        const N: usize = 5;
        let a: [[f32; N]; M] = [
            [0.9314776, 0.10300648, -0.620774, 1.5270752, 0.0259804],
            [0.16820757, -0.94463515, -1.3850101, 1.0600523, 1.5124008],
        ];
        #[rustfmt::skip]
        let b: [f32; N] = [-1.3441996, 1.3965541, -0.89106345, 0.21196432, -0.95535654];
        let mut c: [f32; M] = [1.0; M];
        gemv_truth(1.0, &a, &b, 0.0, &mut c);

        #[rustfmt::skip]
        let a_dev = stream.clone_htod(&[
            0.9314776, 0.10300648, -0.620774, 1.527075, 0.0259804,
            0.16820757, -0.94463515, -1.3850101, 1.0600523, 1.5124008,
        ]).unwrap();
        let b_dev = stream.clone_htod(&b).unwrap();
        let mut c_dev = stream.alloc_zeros(M).unwrap();
        unsafe {
            blas.gemv(
                GemvConfig {
                    trans: sys::cublasOperation_t::CUBLAS_OP_T,
                    m: N as i32,
                    n: M as i32,
                    alpha: 1.0,
                    lda: N as i32,
                    incx: 1,
                    beta: 0.0,
                    incy: 1,
                },
                &a_dev,
                &b_dev,
                &mut c_dev,
            )
        }
        .unwrap();

        let c_host = stream.clone_dtoh(&c_dev).unwrap();
        for i in 0..M {
            assert!((c_host[i] - c[i]).abs() <= 1e-6);
        }
    }

    #[test]
    fn test_dgemv() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).unwrap();
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
        let a_dev = stream.clone_htod(&[
            0.96151888, -0.36771390, 0.94069099,
            2.20621538, -0.16479775, -1.78425562,
            0.41080803, -0.56567699, -0.72781092,
            -0.65718418, -0.14466463, 0.63984287,
            0.20309605, 0.40480086, -1.57559848,
            0.85628128, -0.51614553, -1.15904427,
            -1.84258616, 0.24096519, -0.04563522,
            -0.53364468, -1.07902217, 0.46823528,
        ]).unwrap();
        let b_dev = stream.clone_htod(&b).unwrap();
        let mut c_dev = stream.alloc_zeros(M).unwrap();
        unsafe {
            blas.gemv(
                GemvConfig {
                    trans: sys::cublasOperation_t::CUBLAS_OP_T,
                    m: N as i32,
                    n: M as i32,
                    alpha: 1.0,
                    lda: N as i32,
                    incx: 1,
                    beta: 0.0,
                    incy: 1,
                },
                &a_dev,
                &b_dev,
                &mut c_dev,
            )
        }
        .unwrap();

        let c_host = stream.clone_dtoh(&c_dev).unwrap();
        for i in 0..M {
            assert!((c_host[i] - c[i]).abs() <= 1e-8);
        }
    }
}
