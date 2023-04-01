//! A thin wrapper around [sys] providing [Result]s with [CudnnError].

use std::mem::MaybeUninit;

use super::sys;

pub type CudnnResult<T> = Result<T, CudnnError>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CudnnError(pub sys::cudnnStatus_t);

impl sys::cudnnStatus_t {
    /// Transforms into a [Result] of [CudnnError]
    pub fn result(self) -> Result<(), CudnnError> {
        match self {
            sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(CudnnError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CudnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CudnnError {}

pub fn get_version() -> usize {
    unsafe { sys::cudnnGetVersion() }
}

pub fn get_cudart_version() -> usize {
    unsafe { sys::cudnnGetCudartVersion() }
}

pub fn version_check() -> Result<(), CudnnError> {
    unsafe {
        sys::cudnnAdvInferVersionCheck().result()?;
        sys::cudnnAdvTrainVersionCheck().result()?;
        sys::cudnnCnnInferVersionCheck().result()?;
        sys::cudnnCnnTrainVersionCheck().result()?;
        sys::cudnnOpsInferVersionCheck().result()?;
        sys::cudnnOpsTrainVersionCheck().result()?;
    }
    Ok(())
}

/// Creates a handle to the cuDNN library. See
/// [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate)
pub fn create_handle() -> Result<sys::cudnnHandle_t, CudnnError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cudnnCreate(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// Destroys a handle previously created with [create_handle()]. See
/// [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroy)
///
/// # Safety
///
/// `handle` must not have been freed already.
pub unsafe fn destroy_handle(handle: sys::cudnnHandle_t) -> Result<(), CudnnError> {
    sys::cudnnDestroy(handle).result()
}

/// Sets the stream cuDNN will use. See
/// [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetStream)
///
/// # Safety
///
/// `handle` and `stream` must be valid.
pub unsafe fn set_stream(
    handle: sys::cudnnHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CudnnError> {
    sys::cudnnSetStream(handle, stream).result()
}

pub fn create_tensor_descriptor() -> Result<sys::cudnnTensorDescriptor_t, CudnnError> {
    let mut desc = MaybeUninit::uninit();
    unsafe {
        sys::cudnnCreateTensorDescriptor(desc.as_mut_ptr()).result()?;
        Ok(desc.assume_init())
    }
}

/// # Safety
/// TODO
pub unsafe fn set_tensor4d_descriptor_ex(
    tensor_desc: sys::cudnnTensorDescriptor_t,
    data_type: sys::cudnnDataType_t,
    [n, c, h, w]: [std::ffi::c_int; 4],
    [n_stride, c_stride, h_stride, w_stride]: [std::ffi::c_int; 4],
) -> Result<(), CudnnError> {
    sys::cudnnSetTensor4dDescriptorEx(
        tensor_desc,
        data_type,
        n,
        c,
        h,
        w,
        n_stride,
        c_stride,
        h_stride,
        w_stride,
    )
    .result()
}

/// # Safety
/// TODO
pub unsafe fn set_tensornd_descriptor(
    tensor_desc: sys::cudnnTensorDescriptor_t,
    data_type: sys::cudnnDataType_t,
    num_dims: ::std::os::raw::c_int,
    dims: *const ::std::os::raw::c_int,
    strides: *const ::std::os::raw::c_int,
) -> Result<(), CudnnError> {
    sys::cudnnSetTensorNdDescriptor(tensor_desc, data_type, num_dims, dims, strides).result()
}

/// # Safety
/// TODO
pub unsafe fn destroy_tensor_descriptor(
    desc: sys::cudnnTensorDescriptor_t,
) -> Result<(), CudnnError> {
    sys::cudnnDestroyTensorDescriptor(desc).result()
}

pub fn create_filter_descriptor() -> Result<sys::cudnnFilterDescriptor_t, CudnnError> {
    let mut desc = MaybeUninit::uninit();
    unsafe {
        sys::cudnnCreateFilterDescriptor(desc.as_mut_ptr()).result()?;
        Ok(desc.assume_init())
    }
}

/// # Safety
/// TODO
pub unsafe fn set_filter4d_descriptor(
    filter_desc: sys::cudnnFilterDescriptor_t,
    data_type: sys::cudnnDataType_t,
    format: sys::cudnnTensorFormat_t,
    [k, c, h, w]: [::std::os::raw::c_int; 4],
) -> Result<(), CudnnError> {
    sys::cudnnSetFilter4dDescriptor(filter_desc, data_type, format, k, c, h, w).result()
}

/// # Safety
/// TODO
pub unsafe fn destroy_filter_descriptor(
    desc: sys::cudnnFilterDescriptor_t,
) -> Result<(), CudnnError> {
    sys::cudnnDestroyFilterDescriptor(desc).result()
}

pub fn create_convolution_descriptor() -> Result<sys::cudnnConvolutionDescriptor_t, CudnnError> {
    let mut desc = MaybeUninit::uninit();
    unsafe {
        sys::cudnnCreateConvolutionDescriptor(desc.as_mut_ptr()).result()?;
        Ok(desc.assume_init())
    }
}

/// # Safety
/// TODO
#[allow(clippy::too_many_arguments)]
pub unsafe fn set_convolution2d_descriptor(
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    pad_h: ::std::os::raw::c_int,
    pad_w: ::std::os::raw::c_int,
    u: ::std::os::raw::c_int,
    v: ::std::os::raw::c_int,
    dilation_h: ::std::os::raw::c_int,
    dilation_w: ::std::os::raw::c_int,
    mode: sys::cudnnConvolutionMode_t,
    compute_type: sys::cudnnDataType_t,
) -> Result<(), CudnnError> {
    sys::cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h,
        pad_w,
        u,
        v,
        dilation_h,
        dilation_w,
        mode,
        compute_type,
    )
    .result()
}

/// # Safety
/// TODO
pub unsafe fn destroy_convolution_descriptor(
    desc: sys::cudnnConvolutionDescriptor_t,
) -> Result<(), CudnnError> {
    sys::cudnnDestroyConvolutionDescriptor(desc).result()
}

/// # Safety
/// TODO
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_convolution_forward_algorithm(
    handle: sys::cudnnHandle_t,
    src: sys::cudnnTensorDescriptor_t,
    filter: sys::cudnnFilterDescriptor_t,
    conv: sys::cudnnConvolutionDescriptor_t,
    dest: sys::cudnnTensorDescriptor_t,
    requested_algo_count: ::std::os::raw::c_int,
    returned_algo_count: *mut ::std::os::raw::c_int,
    perf_results: *mut sys::cudnnConvolutionFwdAlgoPerf_t,
) -> Result<(), CudnnError> {
    sys::cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        src,
        filter,
        conv,
        dest,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )
    .result()
}

/// Returns size in **bytes**
/// # Safety
/// TODO
pub unsafe fn get_convolution_forward_workspace_size(
    handle: sys::cudnnHandle_t,
    x: sys::cudnnTensorDescriptor_t,
    w: sys::cudnnFilterDescriptor_t,
    conv: sys::cudnnConvolutionDescriptor_t,
    y: sys::cudnnTensorDescriptor_t,
    algo: sys::cudnnConvolutionFwdAlgo_t,
) -> Result<usize, CudnnError> {
    let mut size_in_bytes = [0];
    sys::cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        x,
        w,
        conv,
        y,
        algo,
        size_in_bytes.as_mut_ptr(),
    )
    .result()?;
    Ok(size_in_bytes[0])
}

/// # Safety
/// TODO
#[allow(clippy::too_many_arguments)]
pub unsafe fn convolution_forward(
    handle: sys::cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    x_desc: sys::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    w_desc: sys::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    algo: sys::cudnnConvolutionFwdAlgo_t,
    work_space: *mut ::core::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const ::core::ffi::c_void,
    y_desc: sys::cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> Result<(), CudnnError> {
    sys::cudnnConvolutionForward(
        handle,
        alpha,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        y_desc,
        y,
    )
    .result()
}

/// # Safety
/// TODO
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_convolution_backward_data_algorithm(
    handle: sys::cudnnHandle_t,
    w_desc: sys::cudnnFilterDescriptor_t,
    dy_desc: sys::cudnnTensorDescriptor_t,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    dx_desc: sys::cudnnTensorDescriptor_t,
    requested_algo_count: ::std::os::raw::c_int,
    returned_algo_count: *mut ::std::os::raw::c_int,
    perf_results: *mut sys::cudnnConvolutionBwdDataAlgoPerf_t,
) -> Result<(), CudnnError> {
    sys::cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle,
        w_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )
    .result()
}

/// Returns size in **bytes**
/// # Safety
/// TODO
pub unsafe fn get_convolution_backward_data_workspace_size(
    handle: sys::cudnnHandle_t,
    w_desc: sys::cudnnFilterDescriptor_t,
    dy_desc: sys::cudnnTensorDescriptor_t,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    dx_desc: sys::cudnnTensorDescriptor_t,
    algo: sys::cudnnConvolutionBwdDataAlgo_t,
) -> Result<usize, CudnnError> {
    let mut size_in_bytes = [0];
    sys::cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        w_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        algo,
        size_in_bytes.as_mut_ptr(),
    )
    .result()?;
    Ok(size_in_bytes[0])
}
/// # Safety
/// TODO
#[allow(clippy::too_many_arguments)]
pub unsafe fn convolution_backward_data(
    handle: sys::cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    w_desc: sys::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    dy_desc: sys::cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    algo: sys::cudnnConvolutionBwdDataAlgo_t,
    work_space: *mut ::core::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const ::core::ffi::c_void,
    dx_desc: sys::cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> Result<(), CudnnError> {
    sys::cudnnConvolutionBackwardData(
        handle,
        alpha,
        w_desc,
        w,
        dy_desc,
        dy,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        dx_desc,
        dx,
    )
    .result()
}

/// # Safety
/// TODO
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_convolution_backward_filter_algorithm(
    handle: sys::cudnnHandle_t,
    src_desc: sys::cudnnTensorDescriptor_t,
    diff_desc: sys::cudnnTensorDescriptor_t,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    grad_desc: sys::cudnnFilterDescriptor_t,
    requested_algo_count: ::std::os::raw::c_int,
    returned_algo_count: *mut ::std::os::raw::c_int,
    perf_results: *mut sys::cudnnConvolutionBwdFilterAlgoPerf_t,
) -> Result<(), CudnnError> {
    sys::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle,
        src_desc,
        diff_desc,
        conv_desc,
        grad_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )
    .result()
}

/// Returns size in **bytes**
/// # Safety
/// TODO
pub unsafe fn get_convolution_backward_filter_workspace_size(
    handle: sys::cudnnHandle_t,
    x_desc: sys::cudnnTensorDescriptor_t,
    dy_desc: sys::cudnnTensorDescriptor_t,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    grad_desc: sys::cudnnFilterDescriptor_t,
    algo: sys::cudnnConvolutionBwdFilterAlgo_t,
) -> Result<usize, CudnnError> {
    let mut size_in_bytes = [0];
    sys::cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle,
        x_desc,
        dy_desc,
        conv_desc,
        grad_desc,
        algo,
        size_in_bytes.as_mut_ptr(),
    )
    .result()?;
    Ok(size_in_bytes[0])
}

/// # Safety
/// TODO
#[allow(clippy::too_many_arguments)]
pub unsafe fn convolution_backward_filter(
    handle: sys::cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    x_desc: sys::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dy_desc: sys::cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    algo: sys::cudnnConvolutionBwdFilterAlgo_t,
    work_space: *mut ::core::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const ::core::ffi::c_void,
    dw_desc: sys::cudnnFilterDescriptor_t,
    dw: *mut ::core::ffi::c_void,
) -> Result<(), CudnnError> {
    sys::cudnnConvolutionBackwardFilter(
        handle,
        alpha,
        x_desc,
        x,
        dy_desc,
        dy,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        dw_desc,
        dw,
    )
    .result()
}
