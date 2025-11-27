//! Grouped GEMM matrix multiplication operations.
//! Safe version of cublasGemmGroupedBatchedEx.
//! ref: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex
use std::{ffi::c_void, sync::Arc};

use crate::{
    cublas::{result::CublasError, sys, CudaBlas},
    driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr},
};

fn htod_copy<T: DeviceRepr>(stream: &Arc<CudaStream>, src: &[T]) -> CudaSlice<T> {
    let mut dst = unsafe { stream.alloc(src.len()).unwrap() };
    stream.memcpy_htod(src, &mut dst).unwrap();
    dst
}

pub trait GmmDType: DeviceRepr {
    type ComputeType: DeviceRepr + Copy;

    fn data_type() -> sys::cudaDataType_t;
    fn compute_type() -> sys::cublasComputeType_t;
}

#[cfg(feature = "f16")]
impl GmmDType for half::f16 {
    type ComputeType = f32;

    fn data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_16F
    }

    fn compute_type() -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}

#[cfg(feature = "f16")]
impl GmmDType for half::bf16 {
    type ComputeType = f32;

    fn data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_16BF
    }

    fn compute_type() -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}

pub struct GmmConfig<T: GmmDType> {
    /// transb for each group (len = group count)
    pub transb_array: Vec<sys::cublasOperation_t>,
    /// transa for each group (len = group count)
    pub transa_array: Vec<sys::cublasOperation_t>,

    /// m for each group (len = group count)
    pub m_array: Vec<usize>,
    /// n for each group (len = group count)
    pub n_array: Vec<usize>,
    /// k for each group (len = group count)
    pub k_array: Vec<usize>,

    /// alpha for each group, must be same as compute type
    /// for data type (len = group count)
    pub alpha_array: Vec<T::ComputeType>,

    /// beta for each group, must be same as compute type
    /// for data type (len = group count)
    pub beta_array: Vec<T::ComputeType>,

    /// A leading dim for each group (len = group count)
    pub lda_array: Vec<usize>,
    /// B leading dim for each group (len = group count)
    pub ldb_array: Vec<usize>,
    /// C leading dim for each group (len = group count)
    pub ldc_array: Vec<usize>,

    /// number of problems in each group (len = group count)
    pub group_size: Vec<usize>,
}

impl<T: GmmDType> GmmConfig<T> {
    pub fn problem_count(&self) -> usize {
        self.group_size.iter().sum()
    }

    pub fn group_count(&self) -> usize {
        self.group_size.len()
    }

    #[inline]
    fn validate(&self) {
        let group_count = self.group_count();

        debug_assert_eq!(self.transb_array.len(), group_count);
        debug_assert_eq!(self.transa_array.len(), group_count);
        debug_assert_eq!(self.m_array.len(), group_count);
        debug_assert_eq!(self.n_array.len(), group_count);
        debug_assert_eq!(self.k_array.len(), group_count);
        debug_assert_eq!(self.alpha_array.len(), group_count);
        debug_assert_eq!(self.beta_array.len(), group_count);
        debug_assert_eq!(self.lda_array.len(), group_count);
        debug_assert_eq!(self.ldb_array.len(), group_count);
        debug_assert_eq!(self.ldc_array.len(), group_count);
    }
}

pub trait Gmm<T: GmmDType> {
    /// Grouped matrix multiplication using raw device pointers.
    ///
    /// * `config` – sizes, leading dimensions, scalars, and counts per group.
    /// * `a_ptrs` – device pointer to pointers to matrices A for every problem (len = problem count).
    /// * `b_ptrs` – device pointer to pointers to matrices B for every problem (len = problem count).
    /// * `c_ptrs` – device pointer to pointers to output matrices C for every problem (len = problem count).
    fn gmm_raw(
        &self,
        config: GmmConfig<T>,
        a_ptrs: *const *const c_void,
        b_ptrs: *const *const c_void,
        c_ptrs: *mut *mut c_void,
    ) -> Result<(), CublasError>;

    /// Grouped matrix multiplication using device slices.
    ///
    /// This will incur a htod copy because the pointer to matrix pointers
    /// must be on-device.
    ///
    /// * `config` – sizes, leading dimensions, scalars, and counts per group.
    /// * `a_slices` – device slices for matrices A for every problem (len = problem count).
    /// * `b_slices` – device slices for matrices B for every problem (len = problem count).
    /// * `c_slices` – device slices for output matrices C for every problem (len = problem count).
    fn gmm(
        &self,
        config: GmmConfig<T>,
        a_slices: &[&CudaSlice<T>],
        b_slices: &[&CudaSlice<T>],
        c_slices: &[&CudaSlice<T>],
    ) -> Result<(), CublasError>;
}

impl<T: GmmDType> Gmm<T> for CudaBlas {
    fn gmm_raw(
        &self,
        config: GmmConfig<T>,
        a_ptrs: *const *const c_void,
        b_ptrs: *const *const c_void,
        c_ptrs: *mut *mut c_void,
    ) -> Result<(), CublasError> {
        config.validate();

        let cuda_dtype = T::data_type();
        let group_count = config.group_count();

        // For CUBLAS_COMPUTE_32F, alpha and beta must be f32
        let alpha_f32: Vec<T::ComputeType> = config
            .alpha_array
            .iter()
            .map(|x| *x as T::ComputeType)
            .collect();
        let beta_f32: Vec<T::ComputeType> = config
            .beta_array
            .iter()
            .map(|x| *x as T::ComputeType)
            .collect();

        let m_array: Vec<i32> = config.m_array.iter().map(|&x| x as i32).collect();
        let n_array: Vec<i32> = config.n_array.iter().map(|&x| x as i32).collect();
        let k_array: Vec<i32> = config.k_array.iter().map(|&x| x as i32).collect();
        let lda_array: Vec<i32> = config.lda_array.iter().map(|&x| x as i32).collect();
        let ldb_array: Vec<i32> = config.ldb_array.iter().map(|&x| x as i32).collect();
        let ldc_array: Vec<i32> = config.ldc_array.iter().map(|&x| x as i32).collect();
        let group_size: Vec<i32> = config.group_size.iter().map(|&x| x as i32).collect();

        unsafe {
            sys::cublasGemmGroupedBatchedEx(
                self.handle,
                config.transa_array.as_ptr(),
                config.transb_array.as_ptr(),
                m_array.as_ptr(),
                n_array.as_ptr(),
                k_array.as_ptr(),
                alpha_f32.as_ptr() as *const _,
                a_ptrs as *const _,
                cuda_dtype,
                lda_array.as_ptr(),
                b_ptrs as *const _,
                cuda_dtype,
                ldb_array.as_ptr(),
                beta_f32.as_ptr() as *const _,
                c_ptrs as *mut _,
                cuda_dtype,
                ldc_array.as_ptr(),
                group_count as i32,
                group_size.as_ptr(),
                T::compute_type(),
            )
            .result()?;
        };

        Ok(())
    }

    fn gmm(
        &self,
        config: GmmConfig<T>,
        a_slices: &[&CudaSlice<T>],
        b_slices: &[&CudaSlice<T>],
        c_slices: &[&CudaSlice<T>],
    ) -> Result<(), CublasError> {
        let (a_ptrs_host, _a_guard_vec): (Vec<u64>, Vec<_>) =
            a_slices.iter().map(|s| s.device_ptr(&self.stream)).unzip();

        let (b_ptrs_host, _b_guard_vec): (Vec<u64>, Vec<_>) =
            b_slices.iter().map(|s| s.device_ptr(&self.stream)).unzip();

        let (c_ptrs_host, _c_guard_vec): (Vec<u64>, Vec<_>) =
            c_slices.iter().map(|s| s.device_ptr(&self.stream)).unzip();

        // TODO coalesce these allocations
        let a_ptrs_dev = htod_copy(&self.stream, &a_ptrs_host);
        let b_ptrs_dev = htod_copy(&self.stream, &b_ptrs_host);
        let c_ptrs_dev = htod_copy(&self.stream, &c_ptrs_host);

        let (a_ptrs, _a_guard) = a_ptrs_dev.device_ptr(&self.stream);
        let (b_ptrs, _b_guard) = b_ptrs_dev.device_ptr(&self.stream);
        let (c_ptrs, _c_guard) = c_ptrs_dev.device_ptr(&self.stream);

        Gmm::gmm_raw(
            self,
            config,
            a_ptrs as *const *const c_void,
            b_ptrs as *const *const c_void,
            c_ptrs as *mut *mut c_void,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{CudaContext, DevicePtr};

    #[test]
    #[cfg(feature = "f16")]
    fn test_grouped_gemm_raw_f16() {
        use half::f16;
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let handle = CudaBlas::new(stream.clone()).unwrap();

        // Group 0: 2 problems, 2x2
        // Problem 0
        // A = | 1 2 |  B = | 5 6 |
        //     | 3 4 |      | 7 8 |
        // C = | 19 22 |
        //     | 43 50 |
        // (Column-Major storage)
        let a0_host = [1.0, 3.0, 2.0, 4.0].map(f16::from_f32);
        let b0_host = [5.0, 7.0, 6.0, 8.0].map(f16::from_f32);
        // Problem 1
        // A = | 5 6 |  B = | 9 10 |
        //     | 7 8 |      | 11 12 |
        // C = | 111 122 |
        //     | 151 166 |
        let a1_host = [5.0, 7.0, 6.0, 8.0].map(f16::from_f32);
        let b1_host = [9.0, 11.0, 10.0, 12.0].map(f16::from_f32);

        // Group 1: 1 problem, 3x3
        // Problem 2
        let a2_host = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0].map(f16::from_f32);
        let b2_host = [4.0, 7.0, 10.0, 5.0, 8.0, 11.0, 6.0, 9.0, 12.0].map(f16::from_f32);

        let a0_dev = htod_copy(&stream, &a0_host);
        let b0_dev = htod_copy(&stream, &b0_host);
        let a1_dev = htod_copy(&stream, &a1_host);
        let b1_dev = htod_copy(&stream, &b1_host);
        let a2_dev = htod_copy(&stream, &a2_host);
        let b2_dev = htod_copy(&stream, &b2_host);

        let c0_dev = stream.alloc_zeros::<f16>(4).unwrap();
        let c1_dev = stream.alloc_zeros::<f16>(4).unwrap();
        let c2_dev = stream.alloc_zeros::<f16>(9).unwrap();

        let a_ptrs_host: Vec<u64> = vec![
            a0_dev.device_ptr(&stream).0,
            a1_dev.device_ptr(&stream).0,
            a2_dev.device_ptr(&stream).0,
        ];
        let b_ptrs_host: Vec<u64> = vec![
            b0_dev.device_ptr(&stream).0,
            b1_dev.device_ptr(&stream).0,
            b2_dev.device_ptr(&stream).0,
        ];
        let c_ptrs_host: Vec<u64> = vec![
            c0_dev.device_ptr(&stream).0,
            c1_dev.device_ptr(&stream).0,
            c2_dev.device_ptr(&stream).0,
        ];

        let a_ptrs_dev = htod_copy(&stream, &a_ptrs_host);
        let b_ptrs_dev = htod_copy(&stream, &b_ptrs_host);
        let c_ptrs_dev = htod_copy(&stream, &c_ptrs_host);

        let config = GmmConfig {
            transb_array: vec![sys::cublasOperation_t::CUBLAS_OP_N; 2],
            transa_array: vec![sys::cublasOperation_t::CUBLAS_OP_N; 2],
            m_array: vec![2, 3],
            n_array: vec![2, 3],
            k_array: vec![2, 3],
            alpha_array: vec![1.0; 2],
            beta_array: vec![0.0; 2],
            lda_array: vec![2, 3],
            ldb_array: vec![2, 3],
            ldc_array: vec![2, 3],
            group_size: vec![2, 1],
        };

        (&handle as &dyn Gmm<f16>)
            .gmm_raw(
                config,
                a_ptrs_dev.device_ptr(&stream).0 as *const *const c_void,
                b_ptrs_dev.device_ptr(&stream).0 as *const *const c_void,
                c_ptrs_dev.device_ptr(&stream).0 as *mut *mut c_void,
            )
            .unwrap();

        let c0_host = stream.memcpy_dtov(&c0_dev).unwrap();
        let c1_host = stream.memcpy_dtov(&c1_dev).unwrap();
        let c2_host = stream.memcpy_dtov(&c2_dev).unwrap();

        let expected_c0 = [19.0, 43.0, 22.0, 50.0].map(f16::from_f32);
        let expected_c1 = [111.0, 151.0, 122.0, 166.0].map(f16::from_f32);
        let expected_c2 =
            [48.0, 111.0, 174.0, 54.0, 126.0, 198.0, 60.0, 141.0, 222.0].map(f16::from_f32);

        assert_eq!(c0_host, expected_c0);
        assert_eq!(c1_host, expected_c1);
        assert_eq!(c2_host, expected_c2);
    }

    #[test]
    #[cfg(feature = "f16")]
    fn test_grouped_gemm_raw_bf16() {
        use half::bf16;
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let handle = CudaBlas::new(stream.clone()).unwrap();

        // Group 0: 2 problems, 2x2
        // Problem 0
        // A = | 1 2 |  B = | 5 6 |
        //     | 3 4 |      | 7 8 |
        // C = | 19 22 |
        //     | 43 50 |
        // (Column-Major storage)
        let a0_host = [1.0, 3.0, 2.0, 4.0].map(bf16::from_f32);
        let b0_host = [5.0, 7.0, 6.0, 8.0].map(bf16::from_f32);
        // Problem 1
        // A = | 5 6 |  B = | 9 10 |
        //     | 7 8 |      | 11 12 |
        // C = | 111 122 |
        //     | 151 166 |
        let a1_host = [5.0, 7.0, 6.0, 8.0].map(bf16::from_f32);
        let b1_host = [9.0, 11.0, 10.0, 12.0].map(bf16::from_f32);

        // Group 1: 1 problem, 3x3
        // Problem 2
        let a2_host = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0].map(bf16::from_f32);
        let b2_host = [4.0, 7.0, 10.0, 5.0, 8.0, 11.0, 6.0, 9.0, 12.0].map(bf16::from_f32);

        let a0_dev = htod_copy(&stream, &a0_host);
        let b0_dev = htod_copy(&stream, &b0_host);
        let a1_dev = htod_copy(&stream, &a1_host);
        let b1_dev = htod_copy(&stream, &b1_host);
        let a2_dev = htod_copy(&stream, &a2_host);
        let b2_dev = htod_copy(&stream, &b2_host);

        let c0_dev = stream.alloc_zeros::<bf16>(4).unwrap();
        let c1_dev = stream.alloc_zeros::<bf16>(4).unwrap();
        let c2_dev = stream.alloc_zeros::<bf16>(9).unwrap();

        let a_ptrs_host: Vec<u64> = vec![
            a0_dev.device_ptr(&stream).0,
            a1_dev.device_ptr(&stream).0,
            a2_dev.device_ptr(&stream).0,
        ];
        let b_ptrs_host: Vec<u64> = vec![
            b0_dev.device_ptr(&stream).0,
            b1_dev.device_ptr(&stream).0,
            b2_dev.device_ptr(&stream).0,
        ];
        let c_ptrs_host: Vec<u64> = vec![
            c0_dev.device_ptr(&stream).0,
            c1_dev.device_ptr(&stream).0,
            c2_dev.device_ptr(&stream).0,
        ];

        let a_ptrs_dev = htod_copy(&stream, &a_ptrs_host);
        let b_ptrs_dev = htod_copy(&stream, &b_ptrs_host);
        let c_ptrs_dev = htod_copy(&stream, &c_ptrs_host);

        let config = GmmConfig {
            transb_array: vec![sys::cublasOperation_t::CUBLAS_OP_N; 2],
            transa_array: vec![sys::cublasOperation_t::CUBLAS_OP_N; 2],
            m_array: vec![2, 3],
            n_array: vec![2, 3],
            k_array: vec![2, 3],
            alpha_array: vec![1.0; 2],
            beta_array: vec![0.0; 2],
            lda_array: vec![2, 3],
            ldb_array: vec![2, 3],
            ldc_array: vec![2, 3],
            group_size: vec![2, 1],
        };

        (&handle as &dyn Gmm<bf16>)
            .gmm_raw(
                config,
                a_ptrs_dev.device_ptr(&stream).0 as *const *const c_void,
                b_ptrs_dev.device_ptr(&stream).0 as *const *const c_void,
                c_ptrs_dev.device_ptr(&stream).0 as *mut *mut c_void,
            )
            .unwrap();

        let c0_host = stream.memcpy_dtov(&c0_dev).unwrap();
        let c1_host = stream.memcpy_dtov(&c1_dev).unwrap();
        let c2_host = stream.memcpy_dtov(&c2_dev).unwrap();

        let expected_c0 = [19.0, 43.0, 22.0, 50.0].map(bf16::from_f32);
        let expected_c1 = [111.0, 151.0, 122.0, 166.0].map(bf16::from_f32);
        let expected_c2 =
            [48.0, 111.0, 174.0, 54.0, 126.0, 198.0, 60.0, 141.0, 222.0].map(bf16::from_f32);

        assert_eq!(c0_host, expected_c0);
        assert_eq!(c1_host, expected_c1);
        assert_eq!(c2_host, expected_c2);
    }
}
