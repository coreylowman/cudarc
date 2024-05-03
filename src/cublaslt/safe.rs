//! Safe abstractions around [crate::cublaslt::result] for doing matmul.

use super::{result, result::CublasError, sys};
use crate::cublaslt::result::set_matrix_layout_attribute;
use crate::driver::sys::{CUdevice_attribute, CUdeviceptr, CUstream};
use crate::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DriverError};
use core::ffi::c_int;
use core::mem;
use std::sync::Arc;

/// Wrapper around [sys::cublasLtHandle_t]
///
/// 1. Create with [CudaBlasLT::new()]
/// 2. Execute matmul kernel with matmul. f32 is supported. f16 and bf16 are supported
/// if feature `half` is activated
///
/// Note: This maintains a instance of [`Arc<CudaDevice>`], so will prevent the device
/// from being dropped. Kernels will be launched on the device device default stream.
#[derive(Debug)]
pub struct CudaBlasLT {
    handle: sys::cublasLtHandle_t,
    workspace: Workspace,
    device: Arc<CudaDevice>,
}

unsafe impl Send for CudaBlasLT {}

unsafe impl Sync for CudaBlasLT {}

impl CudaBlasLT {
    /// Creates a new cublasLt handle.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CublasError> {
        let handle = result::create_handle()?;
        let workspace = Workspace::new(device.clone()).unwrap();

        Ok(Self {
            handle,
            workspace,
            device,
        })
    }
}

impl Drop for CudaBlasLT {
    fn drop(&mut self) {
        let handle = mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}

/// User owned CublasLt workspace buffer.
/// The workspace is initialised following the Nvidia recommendations:
///
/// 1. NVIDIA Hopper Architecture: 32 MiB
/// 2. Other: 4 MiB
#[derive(Debug, Clone)]
pub struct Workspace {
    pub(crate) buffer: CudaSlice<u8>,
    pub(crate) size: usize,
}

impl Workspace {
    /// Creates a CublasLt workspace buffer on the provided device
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

/// Available activation for kernel fusing in matmul
#[derive(Debug, Clone)]
pub enum Activation {
    Relu,
    Gelu,
}

/// MatrixLayout helper type
struct MatrixLayout {
    handle: sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    fn new(
        matrix_type: sys::cudaDataType,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> Result<Self, CublasError> {
        let handle = result::create_matrix_layout(matrix_type, rows, cols, ld)?;
        Ok(Self { handle })
    }

    fn set_batch(&self, size: c_int, stride: i64) -> Result<(), CublasError> {
        unsafe {
            // Set batch size
            set_matrix_layout_attribute(
                self.handle,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                (&size) as *const _ as *const _,
                mem::size_of::<c_int>(),
            )?;
            // Set batch stride
            set_matrix_layout_attribute(
                self.handle,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                (&stride) as *const _ as *const _,
                mem::size_of::<i64>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        // panic on failure
        unsafe {
            result::destroy_matrix_layout(self.handle).expect("Unable to destroy matrix layout")
        }
    }
}

enum Matrix {
    A,
    B,
    #[allow(dead_code)]
    C,
}

/// MatmulDesc helper type
struct MatmulDesc {
    handle: sys::cublasLtMatmulDesc_t,
}

impl MatmulDesc {
    fn new(
        compute_type: sys::cublasComputeType_t,
        scale_type: sys::cudaDataType,
    ) -> Result<Self, CublasError> {
        let handle = result::create_matmul_desc(compute_type, scale_type)?;
        Ok(Self { handle })
    }

    fn set_transpose(&self, transpose: bool, matrix: Matrix) -> Result<(), CublasError> {
        // Set transpose
        // 1 == T, 0 == N
        let transpose = transpose as i32;
        let attr = match matrix {
            Matrix::A => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            Matrix::B => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            Matrix::C => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSC,
        };

        unsafe {
            result::set_matmul_desc_attribute(
                self.handle,
                attr,
                (&transpose) as *const _ as *const _,
                mem::size_of::<u32>(),
            )?;
        }
        Ok(())
    }

    // Epilogue system can be leveraged to fuse add and activation operations
    fn set_epilogue(
        &self,
        act: Option<&Activation>,
        bias_ptr: Option<&CUdeviceptr>,
        stride_bias: Option<i64>,
    ) -> Result<(), CublasError> {
        let epilogue = if let Some(bias_ptr) = bias_ptr {
            let epilogue = act
                .map(|act| match act {
                    // Act + bias
                    Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
                    Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
                })
                // Only bias
                .unwrap_or(sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS);

            // Set bias CUdeviceptr in matmul_desc
            unsafe {
                result::set_matmul_desc_attribute(
                    self.handle,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    bias_ptr as *const CUdeviceptr as *const _,
                    mem::size_of::<CUdeviceptr>(),
                )?;
            }

            if let Some(stride_bias) = stride_bias {
                // Set bias batch stride
                unsafe {
                    result::set_matmul_desc_attribute(
                        self.handle,
                        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
                        (&stride_bias) as *const _ as *const _,
                        mem::size_of::<i64>(),
                    )?;
                }
            }
            epilogue
        } else if let Some(act) = act {
            // Only Act
            match act {
                Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
                Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
            }
        } else {
            // No epilogue
            sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT
        };

        // Set epilogue
        unsafe {
            result::set_matmul_desc_attribute(
                self.handle,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                (&epilogue) as *const _ as *const _,
                mem::size_of::<sys::cublasLtMatmulDescAttributes_t>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        unsafe { result::destroy_matmul_desc(self.handle).expect("Unable to destroy matmul desc") }
    }
}

/// MatmulPref helper type
struct MatmulPref {
    handle: sys::cublasLtMatmulPreference_t,
}

impl MatmulPref {
    fn new() -> Result<Self, CublasError> {
        let handle = result::create_matmul_pref()?;
        Ok(Self { handle })
    }

    fn set_workspace_size(&self, size: usize) -> Result<(), CublasError> {
        unsafe {
            // Set workspace size
            result::set_matmul_pref_attribute(
                self.handle,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&size) as *const _ as *const _,
                mem::size_of::<usize>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatmulPref {
    fn drop(&mut self) {
        unsafe { result::destroy_matmul_pref(self.handle).expect("Unable to destroy matmul pref") }
    }
}

/// [Matmul] super-trait
pub trait MatmulShared {
    /// Returns a reference to the underlying cublasLt handle.
    fn handle(&self) -> &sys::cublasLtHandle_t;

    /// Returns a reference to the underlying cublasLt workspace
    fn workspace(&self) -> &Workspace;

    /// Returns a reference to the underlying stream
    fn stream(&self) -> &CUstream;
}

/// Configuration for [Matmul]
#[derive(Debug, Copy, Clone)]
pub struct MatmulConfig {
    pub transa: bool,
    pub transb: bool,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub alpha: f32,
    pub lda: i64,
    pub ldb: i64,
    pub beta: f32,
    pub ldc: i64,
    pub stride_a: Option<i64>,
    pub stride_b: Option<i64>,
    pub stride_c: Option<i64>,
    pub stride_bias: Option<i64>,
    pub batch_size: Option<c_int>,
}

/// Matrix matrix multiplication with elements of type `T`.
pub trait Matmul<T>: MatmulShared {
    /// Underlying CUDA Type for `T`
    fn matrix_type() -> sys::cudaDataType;

    /// Underlying CUDA Compute Type for `T`
    fn compute_type() -> sys::cublasComputeType_t;

    /// Matrix matrix multiplication. See
    /// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)
    ///
    /// # Safety
    /// This is unsafe because improper arguments may lead to invalid
    /// memory accesses.
    unsafe fn matmul<I: DevicePtr<T>, O: DevicePtrMut<T>>(
        &self,
        cfg: MatmulConfig,
        a: &I,
        b: &I,
        c: &mut O,
        bias: Option<&I>,
        act: Option<&Activation>,
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

        // Creates matrix layouts
        let a_layout = MatrixLayout::new(Self::matrix_type(), a_rows, a_cols, cfg.lda)?;
        if let (Some(batch_size), Some(stride_a)) = (cfg.batch_size, cfg.stride_a) {
            a_layout.set_batch(batch_size, stride_a)?;
        }

        let b_layout = MatrixLayout::new(Self::matrix_type(), b_rows, b_cols, cfg.ldb)?;
        if let (Some(batch_size), Some(stride_b)) = (cfg.batch_size, cfg.stride_b) {
            b_layout.set_batch(batch_size, stride_b)?;
        }

        let c_layout = MatrixLayout::new(Self::matrix_type(), cfg.m, cfg.n, cfg.ldc)?;
        if let (Some(batch_size), Some(stride_c)) = (cfg.batch_size, cfg.stride_c) {
            c_layout.set_batch(batch_size, stride_c)?;
        }

        // Matmul description
        let matmul_desc = MatmulDesc::new(Self::compute_type(), sys::cudaDataType_t::CUDA_R_32F)?;

        // Set transa
        matmul_desc.set_transpose(cfg.transa, Matrix::A)?;
        // Set transb
        matmul_desc.set_transpose(cfg.transb, Matrix::B)?;

        // Epilogue system can be leveraged to fuse add and activation operations
        matmul_desc.set_epilogue(act, bias.map(|b| b.device_ptr()), cfg.stride_bias)?;

        // Create matmul heuristic search preferences
        let matmul_pref = MatmulPref::new()?;

        // Set workspace size
        matmul_pref.set_workspace_size(self.workspace().size)?;

        // Get heuristic given Config, bias, act and workspace size
        let heuristic = result::get_matmul_algo_heuristic(
            *self.handle(),
            matmul_desc.handle,
            a_layout.handle,
            b_layout.handle,
            c_layout.handle,
            c_layout.handle,
            matmul_pref.handle,
        )?;

        // Launch matmul kernel
        result::matmul(
            *self.handle(),
            matmul_desc.handle,
            (&cfg.alpha) as *const _ as *const _,
            (&cfg.beta) as *const _ as *const _,
            *a.device_ptr() as *const _,
            a_layout.handle,
            *b.device_ptr() as *const _,
            b_layout.handle,
            *c.device_ptr_mut() as *const _,
            c_layout.handle,
            *c.device_ptr_mut() as *mut _,
            c_layout.handle,
            (&heuristic.algo) as *const _,
            *self.workspace().buffer.device_ptr() as *const CUdeviceptr as *mut _,
            self.workspace().size,
            *self.stream() as *mut _,
        )
    }
}

impl MatmulShared for CudaBlasLT {
    fn handle(&self) -> &sys::cublasLtHandle_t {
        &self.handle
    }

    fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    fn stream(&self) -> &CUstream {
        &self.device.stream
    }
}

impl Matmul<f32> for CudaBlasLT {
    fn matrix_type() -> sys::cudaDataType {
        sys::cudaDataType_t::CUDA_R_32F
    }

    fn compute_type() -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    }
}

#[cfg(feature = "f16")]
impl Matmul<half::f16> for CudaBlasLT {
    fn matrix_type() -> sys::cudaDataType {
        sys::cudaDataType_t::CUDA_R_16F
    }

    fn compute_type() -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}

#[cfg(feature = "f16")]
impl Matmul<half::bf16> for CudaBlasLT {
    fn matrix_type() -> sys::cudaDataType {
        sys::cudaDataType_t::CUDA_R_16BF
    }

    fn compute_type() -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::sys::lib;
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
        unsafe { lib().cublasLtLoggerSetLevel(4).result().unwrap() };
        unsafe {
            lib()
                .cublasLtLoggerOpenFile(logpath.as_ptr())
                .result()
                .unwrap()
        };

        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlasLT::new(dev.clone()).unwrap();
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
            let bias = dev.alloc_zeros::<f32>(N).unwrap();

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
                    stride_a: None,
                    stride_b: None,
                    stride_c: None,
                    stride_bias: None,
                    batch_size: None,
                },
                &b_dev,
                &a_dev,
                &mut c_dev,
                Some(&bias),
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
                    (found - expected) <= 1e-6,
                    "found={found:?}, expected={expected:?}"
                );
            }
        }
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_matmul_half() {
        let logpath = CString::new("log_matmul_half").unwrap();
        unsafe { lib().cublasLtLoggerSetLevel(4).result().unwrap() };
        unsafe {
            lib()
                .cublasLtLoggerOpenFile(logpath.as_ptr())
                .result()
                .unwrap()
        };

        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlasLT::new(dev.clone()).unwrap();
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
        let bias = dev.alloc_zeros::<half::f16>(N).unwrap();
        let mut c_dev = dev.alloc_zeros::<half::f16>(M * N).unwrap();
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
                    stride_a: None,
                    stride_b: None,
                    stride_c: None,
                    stride_bias: None,
                    batch_size: None,
                },
                &b_dev,
                &a_dev,
                &mut c_dev,
                Some(&bias),
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
        let bias = dev.alloc_zeros::<half::bf16>(N).unwrap();
        let mut c_dev = dev.alloc_zeros::<half::bf16>(M * N).unwrap();
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
                    stride_a: None,
                    stride_b: None,
                    stride_c: None,
                    stride_bias: None,
                    batch_size: None,
                },
                &b_dev,
                &a_dev,
                &mut c_dev,
                Some(&bias),
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
