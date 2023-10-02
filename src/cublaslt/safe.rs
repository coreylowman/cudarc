//! Safe abstractions around [crate::cublaslt::result] for doing matmul.

use super::{result, result::CublasError, sys};
use crate::driver::sys::CUstream;
use crate::driver::sys::{CUdevice_attribute, CUdeviceptr_v2};
use crate::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DriverError};
use core::mem;
use std::println;
use std::sync::Arc;

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

        let major =
            device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let workspace_size = if major >= 9 { 33_554_432 } else { 4_194_304 };

        let buffer = unsafe { device.alloc::<u8>(workspace_size)? };
        Ok(Self {
            buffer,
            size: workspace_size,
        })
    }
}

#[derive(Debug)]
pub enum Activation {
    Relu,
    Gelu,
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
        workspace: &mut Workspace,
        stream: CUstream,
        a: &A,
        b: &B,
        c: &mut C,
        bias: Option<&A>,
        act: Option<Activation>,
    ) -> Result<(), CublasError>;
}

impl Matmul<f32> for CudaBlasLT {
    unsafe fn matmul<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        cfg: MatmulConfig<f32>,
        workspace: &mut Workspace,
        stream: CUstream,
        a: &A,
        b: &B,
        c: &mut C,
        bias: Option<&A>,
        act: Option<Activation>,
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

        let a_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, a_rows, a_cols, cfg.lda)?;
        let b_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, b_rows, b_cols, cfg.ldb)?;
        let c_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, cfg.m, cfg.n, cfg.ldc)?;

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

        let epilogue = if let Some(bias) = bias {
            let epilogue = act
                .map(|act| match act {
                    Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
                    Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
                })
                .unwrap_or(sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS);

            println!("??? {:?}", a.device_ptr());
            println!("??? {:?}", bias.device_ptr());

            result::set_matmul_desc_attribute(
                matmul_desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                *bias.device_ptr() as *const _,
                mem::size_of::<f32>(),
            )?;

            println!("rip");

            epilogue
        } else if let Some(act) = act {
            match act {
                Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
                Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
            }
        } else {
            sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT
        };

        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            (&epilogue) as *const _ as *const _,
            mem::size_of::<sys::cublasLtMatmulDescAttributes_t>(),
        )?;

        let matmul_pref = result::create_matmul_pref()?;

        result::set_matmul_pref_attribute(
            matmul_pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&workspace.size) as *const _ as *const _,
            mem::size_of::<usize>(),
        )?;

        let heuristic = result::get_matmul_algo_heuristic(
            self.handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            matmul_pref,
        )?;

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
            stream as *mut _,
        )
    }
}

#[cfg(feature = "f16")]
impl Matmul<half::f16> for CudaBlasLT {
    unsafe fn matmul<
        A: DevicePtr<half::f16>,
        B: DevicePtr<half::f16>,
        C: DevicePtrMut<half::f16>,
    >(
        &self,
        cfg: MatmulConfig<half::f16>,
        workspace: &mut Workspace,
        stream: CUstream,
        a: &A,
        b: &B,
        c: &mut C,
        bias: Option<&A>,
        act: Option<Activation>,
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

        let a_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_16F, a_rows, a_cols, cfg.lda)?;
        let b_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_16F, b_rows, b_cols, cfg.ldb)?;
        let c_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_16F, cfg.m, cfg.n, cfg.ldc)?;

        let matmul_desc = result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
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

        let epilogue = if let Some(bias) = bias {
            let epilogue = act
                .map(|act| match act {
                    Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
                    Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
                })
                .unwrap_or(sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS);

            result::set_matmul_desc_attribute(
                matmul_desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                *bias.device_ptr() as *const _,
                mem::size_of::<CUdeviceptr_v2>(),
            )?;
            epilogue
        } else if let Some(act) = act {
            match act {
                Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
                Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
            }
        } else {
            sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT
        };

        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            (&epilogue) as *const _ as *const _,
            mem::size_of::<sys::cublasLtMatmulDescAttributes_t>(),
        )?;

        let matmul_pref = result::create_matmul_pref()?;

        result::set_matmul_pref_attribute(
            matmul_pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&workspace.size) as *const _ as *const _,
            mem::size_of::<usize>(),
        )?;

        let heuristic = result::get_matmul_algo_heuristic(
            self.handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            matmul_pref,
        )?;

        let alpha: f32 = cfg.alpha.to_f32();
        let beta: f32 = cfg.beta.to_f32();
        result::matmul(
            self.handle,
            matmul_desc,
            (&alpha) as *const _ as *const _,
            (&beta) as *const _ as *const _,
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
            stream as *mut _,
        )
    }
}

#[cfg(feature = "f16")]
impl Matmul<half::bf16> for CudaBlasLT {
    unsafe fn matmul<
        A: DevicePtr<half::bf16>,
        B: DevicePtr<half::bf16>,
        C: DevicePtrMut<half::bf16>,
    >(
        &self,
        cfg: MatmulConfig<half::bf16>,
        workspace: &mut Workspace,
        stream: CUstream,
        a: &A,
        b: &B,
        c: &mut C,
        bias: Option<&A>,
        act: Option<Activation>,
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
            sys::cudaDataType_t::CUDA_R_16BF,
            a_rows,
            a_cols,
            cfg.lda,
        )?;
        let b_layout = result::create_matrix_layout(
            sys::cudaDataType_t::CUDA_R_16BF,
            b_rows,
            b_cols,
            cfg.ldb,
        )?;
        let c_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_16BF, cfg.m, cfg.n, cfg.ldc)?;

        let matmul_desc = result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
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

        let epilogue = if let Some(bias) = bias {
            let epilogue = act
                .map(|act| match act {
                    Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
                    Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
                })
                .unwrap_or(sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS);

            result::set_matmul_desc_attribute(
                matmul_desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                *bias.device_ptr() as *const _,
                mem::size_of::<CUdeviceptr_v2>(),
            )?;
            epilogue
        } else if let Some(act) = act {
            match act {
                Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
                Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
            }
        } else {
            sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT
        };

        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            (&epilogue) as *const _ as *const _,
            mem::size_of::<sys::cublasLtMatmulDescAttributes_t>(),
        )?;

        let matmul_pref = result::create_matmul_pref()?;

        result::set_matmul_pref_attribute(
            matmul_pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&workspace.size) as *const _ as *const _,
            mem::size_of::<usize>(),
        )?;

        let heuristic = result::get_matmul_algo_heuristic(
            self.handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            matmul_pref,
        )?;

        let alpha: f32 = cfg.alpha.to_f32();
        let beta: f32 = cfg.beta.to_f32();
        result::matmul(
            self.handle,
            matmul_desc,
            (&alpha) as *const _ as *const _,
            (&beta) as *const _ as *const _,
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
            stream as *mut _,
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use std::ffi::CString;

    fn matmul_truth<T, const M: usize, const N: usize, const K: usize>(
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
    fn test_matmul_f32() {
        let logpath = CString::new("log_matmul_f32").unwrap();
        unsafe { sys::cublasLtLoggerSetLevel(4).result().unwrap() };
        unsafe {
            sys::cublasLtLoggerOpenFile(logpath.as_ptr())
                .result()
                .unwrap()
        };

        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlasLT::new().unwrap();
        let mut workspace = Workspace::new(dev.clone()).unwrap();
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
        #[rustfmt::skip]
        let bias_dev = dev.htod_sync_copy::<f32>(&[
            1.1292169, -0.13450263, 0.62789696, -0.5685516, 0.21946938,
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
                &mut workspace,
                dev.stream,
                &b_dev,
                &a_dev,
                &mut c_dev,
                Some(&bias_dev),
                None
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

    #[cfg(feature = "f16")]
    #[test]
    fn test_matmul_half() {
        let logpath = CString::new("log_matmul_half").unwrap();
        unsafe { sys::cublasLtLoggerSetLevel(4).result().unwrap() };
        unsafe {
            sys::cublasLtLoggerOpenFile(logpath.as_ptr())
                .result()
                .unwrap()
        };

        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlasLT::new().unwrap();
        let mut workspace = Workspace::new(dev.clone()).unwrap();
        const M: usize = 2;
        const K: usize = 4;
        const N: usize = 6;
        let a: [[half::f16; K]; M] = [
            [-0.5944882, 1.8055636, 0.52204555, -0.00397902],
            [-0.38346434, -0.38013917, 0.4198623, -0.22479166],
        ]
        .map(|r| r.map(half::f16::from_f32));
        let b: [[half::f16; N]; K] = [
            [
                1.1292169,
                -0.13450263,
                0.62789696,
                -0.5685516,
                0.21946938,
                -1.6661372,
            ],
            [
                1.0585804,
                -0.39789402,
                0.90205914,
                0.989318,
                -0.3443096,
                -0.4568837,
            ],
            [
                1.3412506,
                0.3059701,
                -0.9714474,
                -0.36113533,
                -1.6809629,
                -0.9043474,
            ],
            [
                3.4746711,
                -1.0930681,
                0.16502666,
                -0.59988785,
                0.41375792,
                0.39125723,
            ],
        ]
        .map(|r| r.map(half::f16::from_f32));
        let mut c: [[half::f16; N]; M] = [[0.0; N]; M].map(|r| r.map(half::f16::from_f32));
        matmul_truth(
            half::f16::from_f32(1.0),
            &a,
            &b,
            half::f16::from_f32(0.0),
            &mut c,
        );

        #[rustfmt::skip]
            let a_dev = dev.htod_sync_copy::<half::f16>(&[
            -0.5944882, 1.8055636, 0.52204555, -0.00397902,
            -0.38346434, -0.38013917, 0.4198623, -0.22479166,
        ].map(half::f16::from_f32)).unwrap();
        #[rustfmt::skip]
            let b_dev = dev.htod_sync_copy::<half::f16>(&[
            1.1292169, -0.13450263, 0.62789696, -0.5685516, 0.21946938, -1.6661372,
            1.0585804, -0.39789402, 0.90205914, 0.989318, -0.3443096, -0.4568837,
            1.3412506, 0.3059701, -0.9714474, -0.36113533, -1.6809629, -0.9043474,
            3.4746711, -1.0930681, 0.16502666, -0.59988785, 0.41375792, 0.39125723,
        ].map(half::f16::from_f32)).unwrap();
        let mut c_dev = dev.alloc_zeros::<half::f16>(M * N).unwrap();
        unsafe {
            blas.matmul(
                MatmulConfig {
                    transa: false,
                    transb: false,
                    m: N as u64,
                    n: M as u64,
                    k: K as u64,
                    alpha: half::f16::from_f32(1.0),
                    lda: N as i64,
                    ldb: K as i64,
                    beta: half::f16::from_f32(0.0),
                    ldc: N as i64,
                },
                &mut workspace,
                dev.stream,
                &b_dev,
                &a_dev,
                &mut c_dev,
                None,
                None,
            )
        }
        .unwrap();

        let c_host = dev.sync_reclaim(c_dev).unwrap();
        for m in 0..M {
            for n in 0..N {
                let found = c_host[m * N + n];
                let expected = c[m][n];
                assert!(
                    (found - expected) <= half::f16::from_f32(1e-2),
                    "found={found:?}, expected={expected:?}"
                );
            }
        }

        #[rustfmt::skip]
            let a_dev = dev.htod_sync_copy::<half::bf16>(&[
            -0.5944882, 1.8055636, 0.52204555, -0.00397902,
            -0.38346434, -0.38013917, 0.4198623, -0.22479166,
        ].map(half::bf16::from_f32)).unwrap();
        #[rustfmt::skip]
            let b_dev = dev.htod_sync_copy::<half::bf16>(&[
            1.1292169, -0.13450263, 0.62789696, -0.5685516, 0.21946938, -1.6661372,
            1.0585804, -0.39789402, 0.90205914, 0.989318, -0.3443096, -0.4568837,
            1.3412506, 0.3059701, -0.9714474, -0.36113533, -1.6809629, -0.9043474,
            3.4746711, -1.0930681, 0.16502666, -0.59988785, 0.41375792, 0.39125723,
        ].map(half::bf16::from_f32)).unwrap();
        let mut c_dev = dev.alloc_zeros::<half::bf16>(M * N).unwrap();
        unsafe {
            blas.matmul(
                MatmulConfig {
                    transa: false,
                    transb: false,
                    m: N as u64,
                    n: M as u64,
                    k: K as u64,
                    alpha: half::bf16::from_f32(1.0),
                    lda: N as i64,
                    ldb: K as i64,
                    beta: half::bf16::from_f32(0.0),
                    ldc: N as i64,
                },
                &mut workspace,
                dev.stream,
                &b_dev,
                &a_dev,
                &mut c_dev,
                None,
                None,
            )
        }
        .unwrap();
        let c_host = dev.sync_reclaim(c_dev).unwrap();
        for m in 0..M {
            for n in 0..N {
                let found = c_host[m * N + n];
                let expected = c[m][n];
                assert!(
                    (half::bf16::to_f32(found) - half::f16::to_f32(expected)) <= 1e-2,
                    "found={found:?}, expected={expected:?}"
                );
            }
        }
    }
}
