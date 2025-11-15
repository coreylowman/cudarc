use crate::cusolver::sys::*;
use crate::{
    cusolver::sys::cusolverEigMode_t,
    driver::{CudaContext, DevicePtr},
};
use core::ffi::c_int;
use core::mem::MaybeUninit;
use std::{vec, vec::Vec};

#[test]
#[allow(clippy::excessive_precision)]
fn test_ssyevd() {
    // equilvant code in python:
    // ```
    // import numpy as np
    // a = np.arange(25).reshape(5, 5).T  # cusolver uses column-major
    // e, v = np.linalg.eigh(a, UPLO='U')
    // print(e)
    // print(v.T)                         # cusolver uses column-major

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let n: usize = 5;
    let a: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
    let a = stream.clone_htod(&a).unwrap();
    let lda = n;
    let w = stream.alloc_zeros::<f32>(n).unwrap();
    let work = stream.alloc_zeros::<f32>(1024).unwrap();
    let lwork = 1024;
    let info = stream.alloc_zeros::<c_int>(1).unwrap();

    let handle = {
        let mut handle = MaybeUninit::uninit();
        unsafe {
            let stat = cusolverDnCreate(handle.as_mut_ptr());
            assert_eq!(stat, cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
            handle.assume_init()
        }
    };
    let jobz = cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
    let uplo = cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
    let n_ffi = n as c_int;
    let a_ffi = a.device_ptr(&stream).0 as *mut f32;
    let lda_ffi = lda as c_int;
    let w_ffi = w.device_ptr(&stream).0 as *mut f32;
    let work_ffi = work.device_ptr(&stream).0 as *mut f32;
    let lwork_ffi = lwork as c_int;
    let info_ffi = info.device_ptr(&stream).0 as *mut c_int;

    let stat = unsafe {
        cusolverDnSsyevd(
            handle, jobz, uplo, n_ffi, a_ffi, lda_ffi, w_ffi, work_ffi, lwork_ffi, info_ffi,
        )
    };
    assert_eq!(stat, cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);

    let a = stream.clone_dtoh(&a).unwrap();
    let w = stream.clone_dtoh(&w).unwrap();

    // reference value
    #[rustfmt::skip]
    let a_ref = vec![
        [ 0.66532203,  0.4597277 ,  0.13485761, -0.2250011 , -0.52648358],
        [-0.52969785,  0.16421032,  0.63734141,  0.2535813 , -0.47111317],
        [ 0.37231491, -0.60582469, -0.02526467,  0.6202977 , -0.33007653],
        [-0.20110065,  0.51839766, -0.63485056,  0.50534169, -0.18004701],
        [ 0.31258105,  0.35486015,  0.41465144,  0.49490561,  0.59958317],
    ];
    let a_ref = a_ref.into_iter().flatten().collect::<Vec<f32>>();

    #[rustfmt::skip]
    let w_ref = [
        -15.41730621,  -2.97513341,  -1.55469259,  -1.10724576, 81.05437797
    ];

    // eigenvalues may sign-flipped, so take absolute value to check
    for i in 0..n * n {
        assert!((a[i].abs() - a_ref[i].abs()).abs() < 1e-5);
    }
    for i in 0..n {
        assert!((w[i] - w_ref[i]).abs() < 1e-5);
    }
}
