use super::core::*;
use crate::{
    cudnn::{result, result::CudnnError, sys},
    driver::{DevicePtr, DevicePtrMut},
};

use std::{marker::PhantomData, sync::Arc};

/// A descriptor of the filters for conv2d operation. Create with [`Cudnn::create_4d_filter()`]
#[derive(Debug)]
pub struct FilterDescriptor<T> {
    pub(crate) desc: sys::cudnnFilterDescriptor_t,
    #[allow(unused)]
    pub(crate) handle: Arc<Cudnn>,
    pub(crate) marker: PhantomData<T>,
}

impl Cudnn {
    /// Create a filter 4d descriptor.
    pub fn create_4d_filter<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        format: sys::cudnnTensorFormat_t,
        dims: [std::ffi::c_int; 4],
    ) -> Result<FilterDescriptor<T>, CudnnError> {
        let desc = result::create_filter_descriptor()?;
        let desc = FilterDescriptor {
            desc,
            handle: self.clone(),
            marker: PhantomData,
        };
        unsafe { result::set_filter4d_descriptor(desc.desc, T::DATA_TYPE, format, dims) }?;
        Ok(desc)
    }
}

impl<T> Drop for FilterDescriptor<T> {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::destroy_filter_descriptor(desc) }.unwrap()
        }
    }
}

/// A descriptor for a conv2d operation holding stride, padding, and dilation.
#[derive(Debug)]
pub struct Conv2dDescriptor<T> {
    pub(crate) desc: sys::cudnnConvolutionDescriptor_t,
    pub(crate) handle: Arc<Cudnn>,
    pub(crate) marker: PhantomData<T>,
}

impl Cudnn {
    /// Creates a conv2d descriptor.
    /// - `pad` is the padding to apply to height and width of image
    /// - `stride` is the kernel strides
    /// - `dilation` is the kernel dilation
    /// - `mode` - CROSS_CORRELATION is standard convolution
    pub fn create_conv2d<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        pad: [std::ffi::c_int; 2],
        stride: [std::ffi::c_int; 2],
        dilation: [std::ffi::c_int; 2],
        mode: sys::cudnnConvolutionMode_t,
    ) -> Result<Conv2dDescriptor<T>, CudnnError> {
        let [pad_h, pad_w] = pad;
        let [stride_h, stride_w] = stride;
        let [dilation_h, dilation_w] = dilation;
        let desc = result::create_convolution_descriptor()?;
        let desc = Conv2dDescriptor {
            desc,
            handle: self.clone(),
            marker: PhantomData,
        };
        unsafe {
            result::set_convolution2d_descriptor(
                desc.desc,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                mode,
                T::DATA_TYPE,
            )
        }?;
        Ok(desc)
    }
}

impl<T> Conv2dDescriptor<T> {
    /// Set's the math type for this convolution. Refer to [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetConvolutionMathType)
    /// for more information.
    pub fn set_math_type(&mut self, math_type: sys::cudnnMathType_t) -> Result<(), CudnnError> {
        unsafe { result::set_convolution_math_type(self.desc, math_type) }
    }

    /// Set's the group count for this convolution. Refer to [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetConvolutionGroupCount)
    /// for more information.
    pub fn set_group_count(&mut self, group_count: i32) -> Result<(), CudnnError> {
        unsafe { result::set_convolution_group_count(self.desc, group_count) }
    }
}

impl<T> Drop for Conv2dDescriptor<T> {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::destroy_convolution_descriptor(desc) }.unwrap()
        }
    }
}

/// The convolution 2d forward operation. Pass in references to descriptors
/// directly, and then call:
/// 1. [`Conv2dForward::pick_algorithm()`] to use cudnn heuristics to select the algorithm
/// 2. [`Conv2dForward::get_workspace_size()`] to get required workspace size.
/// 3. [`Conv2dForward::launch()`] to execute it
#[derive(Debug)]
pub struct Conv2dForward<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> {
    /// Conv parameters
    pub conv: &'a Conv2dDescriptor<C>,
    /// Input image descriptor
    pub x: &'a TensorDescriptor<X>,
    /// Filter descriptor
    pub w: &'a FilterDescriptor<X>,
    /// Output image descriptor
    pub y: &'a TensorDescriptor<Y>,
}

impl<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> Conv2dForward<'a, X, C, Y> {
    /// Picks the fastest algorithm from all available cuDNN algorithms based on cudnn heuristics.
    pub fn pick_algorithm(&self) -> Result<sys::cudnnConvolutionFwdAlgo_t, CudnnError> {
        const NUM_ALGOS: usize = 8;
        debug_assert_eq!(
            sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT as u32,
            NUM_ALGOS as u32
        );
        let mut returned_count = [0; 1];
        let mut perf_results = [Default::default(); NUM_ALGOS];
        unsafe {
            result::get_convolution_forward_algorithm(
                self.conv.handle.handle,
                self.x.desc,
                self.w.desc,
                self.conv.desc,
                self.y.desc,
                NUM_ALGOS as std::ffi::c_int,
                returned_count.as_mut_ptr(),
                perf_results.as_mut_ptr(),
            )
        }?;
        assert!(returned_count[0] > 0);
        perf_results[0].status.result()?;
        Ok(perf_results[0].algo)
    }

    /// Returns size in **bytes** to execute the selected algorithm.
    pub fn get_workspace_size(
        &self,
        algo: sys::cudnnConvolutionFwdAlgo_t,
    ) -> Result<usize, CudnnError> {
        unsafe {
            result::get_convolution_forward_workspace_size(
                self.conv.handle.handle,
                self.x.desc,
                self.w.desc,
                self.conv.desc,
                self.y.desc,
                algo,
            )
        }
    }

    /// Launches the operation.
    ///
    /// - `img` is the input image
    /// - `filter` is the convolution kernels
    /// - `y` is the output
    ///
    /// # Safety
    /// The img/filter/y arguments must match the data type/layout specified in the
    /// descriptors in `self.
    pub unsafe fn launch<Workspace, Img, Filter, Dst>(
        &self,
        algo: sys::cudnnConvolutionFwdAlgo_t,
        workspace: Option<&mut Workspace>,
        (alpha, beta): (Y, Y),
        img: &Img,
        filter: &Filter,
        y: &mut Dst,
    ) -> Result<(), CudnnError>
    where
        Workspace: DevicePtrMut<u8>,
        Img: DevicePtr<X>,
        Filter: DevicePtr<X>,
        Dst: DevicePtrMut<Y>,
    {
        let (num_bytes, workspace_ptr) = match workspace {
            Some(w) => (
                w.num_bytes(),
                *w.device_ptr_mut() as *mut u8 as *mut std::ffi::c_void,
            ),
            None => (0, std::ptr::null_mut()),
        };
        let alpha = alpha.into_scaling_parameter();
        let beta = beta.into_scaling_parameter();
        result::convolution_forward(
            self.conv.handle.handle,
            (&alpha) as *const Y::Scalar as *const std::ffi::c_void,
            self.x.desc,
            *img.device_ptr() as *const X as *const std::ffi::c_void,
            self.w.desc,
            *filter.device_ptr() as *const X as *const std::ffi::c_void,
            self.conv.desc,
            algo,
            workspace_ptr,
            num_bytes,
            (&beta) as *const Y::Scalar as *const std::ffi::c_void,
            self.y.desc,
            *y.device_ptr_mut() as *mut Y as *mut std::ffi::c_void,
        )
    }
}

/// The convolution 2d backward operation for the input image. Pass in references to descriptors
/// directly, and then call:
/// 1. [`Conv2dBackwardData::pick_algorithm()`] to use cudnn heuristics to select the algorithm
/// 2. [`Conv2dBackwardData::get_workspace_size()`] to get required workspace size.
/// 3. [`Conv2dBackwardData::launch()`] to execute it
#[derive(Debug)]
pub struct Conv2dBackwardData<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> {
    /// Conv descriptor
    pub conv: &'a Conv2dDescriptor<C>,
    /// Input image descriptor
    pub dx: &'a TensorDescriptor<X>,
    /// Filter descriptor
    pub w: &'a FilterDescriptor<X>,
    /// Output image descriptor
    pub dy: &'a TensorDescriptor<Y>,
}

impl<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> Conv2dBackwardData<'a, X, C, Y> {
    /// Picks the fastest algorithm from all available cuDNN algorithms based on cudnn heuristics.
    pub fn pick_algorithm(&self) -> Result<sys::cudnnConvolutionBwdDataAlgo_t, CudnnError> {
        const NUM_ALGOS: usize = 6;
        debug_assert_eq!(
            sys::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT as u32,
            NUM_ALGOS as u32
        );
        let mut returned_count = [0; 1];
        let mut perf_results = [Default::default(); NUM_ALGOS];
        unsafe {
            result::get_convolution_backward_data_algorithm(
                self.conv.handle.handle,
                self.w.desc,
                self.dy.desc,
                self.conv.desc,
                self.dx.desc,
                NUM_ALGOS as std::ffi::c_int,
                returned_count.as_mut_ptr(),
                perf_results.as_mut_ptr(),
            )
        }?;
        assert!(returned_count[0] > 0);
        perf_results[0].status.result()?;
        Ok(perf_results[0].algo)
    }

    /// Returns size in **bytes** to execute the selected algorithm.
    pub fn get_workspace_size(
        &self,
        algo: sys::cudnnConvolutionBwdDataAlgo_t,
    ) -> Result<usize, CudnnError> {
        unsafe {
            result::get_convolution_backward_data_workspace_size(
                self.conv.handle.handle,
                self.w.desc,
                self.dy.desc,
                self.conv.desc,
                self.dx.desc,
                algo,
            )
        }
    }

    /// Launches the operation.
    ///
    /// - `dx` is the gradient of the input image to populate
    /// - `filter` is the convolution kernels
    /// - `dy` is the gradient of the output image
    ///
    /// # Safety
    /// The arguments must match the data type/layout specified in the
    /// descriptors in `self.
    pub unsafe fn launch<Workspace, Img, Filter, Dst>(
        &self,
        algo: sys::cudnnConvolutionBwdDataAlgo_t,
        workspace: Option<&mut Workspace>,
        (alpha, beta): (Y, Y),
        dx: &mut Img,
        filter: &Filter,
        dy: &Dst,
    ) -> Result<(), CudnnError>
    where
        Workspace: DevicePtrMut<u8>,
        Img: DevicePtrMut<X>,
        Filter: DevicePtr<X>,
        Dst: DevicePtr<Y>,
    {
        let (num_bytes, workspace_ptr) = match workspace {
            Some(w) => (
                w.num_bytes(),
                *w.device_ptr_mut() as *mut u8 as *mut std::ffi::c_void,
            ),
            None => (0, std::ptr::null_mut()),
        };
        let alpha = alpha.into_scaling_parameter();
        let beta = beta.into_scaling_parameter();
        result::convolution_backward_data(
            self.conv.handle.handle,
            (&alpha) as *const Y::Scalar as *const std::ffi::c_void,
            self.w.desc,
            *filter.device_ptr() as *const X as *const std::ffi::c_void,
            self.dy.desc,
            *dy.device_ptr() as *const Y as *const std::ffi::c_void,
            self.conv.desc,
            algo,
            workspace_ptr,
            num_bytes,
            (&beta) as *const Y::Scalar as *const std::ffi::c_void,
            self.dx.desc,
            *dx.device_ptr_mut() as *mut X as *mut std::ffi::c_void,
        )
    }
}

/// The convolution 2d backward operation for the filters. Pass in references to descriptors
/// directly, and then call:
/// 1. [`Conv2dBackwardFilter::pick_algorithm()`] to use cudnn heuristics to select the algorithm
/// 2. [`Conv2dBackwardFilter::get_workspace_size()`] to get required workspace size.
/// 3. [`Conv2dBackwardFilter::launch()`] to execute it
#[derive(Debug)]
pub struct Conv2dBackwardFilter<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> {
    /// Conv descriptor
    pub conv: &'a Conv2dDescriptor<C>,
    /// Input image descriptor
    pub x: &'a TensorDescriptor<X>,
    /// Filter descriptor
    pub dw: &'a FilterDescriptor<X>,
    /// Output image descriptor
    pub dy: &'a TensorDescriptor<Y>,
}

impl<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> Conv2dBackwardFilter<'a, X, C, Y> {
    /// Picks the fastest algorithm from all available cuDNN algorithms based on cudnn heuristics.
    pub fn pick_algorithm(&self) -> Result<sys::cudnnConvolutionBwdFilterAlgo_t, CudnnError> {
        const NUM_ALGOS: usize = 7;
        debug_assert_eq!(
            sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT as u32,
            NUM_ALGOS as u32
        );
        let mut returned_count = [0; 1];
        let mut perf_results = [Default::default(); NUM_ALGOS];
        unsafe {
            result::get_convolution_backward_filter_algorithm(
                self.conv.handle.handle,
                self.x.desc,
                self.dy.desc,
                self.conv.desc,
                self.dw.desc,
                NUM_ALGOS as std::ffi::c_int,
                returned_count.as_mut_ptr(),
                perf_results.as_mut_ptr(),
            )
        }?;
        assert!(returned_count[0] > 0);
        perf_results[0].status.result()?;
        Ok(perf_results[0].algo)
    }

    /// Returns size in **bytes** to execute the selected algorithm.
    pub fn get_workspace_size(
        &self,
        algo: sys::cudnnConvolutionBwdFilterAlgo_t,
    ) -> Result<usize, CudnnError> {
        unsafe {
            result::get_convolution_backward_filter_workspace_size(
                self.conv.handle.handle,
                self.x.desc,
                self.dy.desc,
                self.conv.desc,
                self.dw.desc,
                algo,
            )
        }
    }

    /// Launches the operation.
    ///
    /// - `x` is the input image
    /// - `dfilter` is the gradient of the convolution kernels
    /// - `dy` is the gradient of the output image
    ///
    /// # Safety
    /// The arguments must match the data type/layout specified in the
    /// descriptors in `self.
    pub unsafe fn launch<Workspace, Img, Filter, Dst>(
        &self,
        algo: sys::cudnnConvolutionBwdFilterAlgo_t,
        workspace: Option<&mut Workspace>,
        (alpha, beta): (Y, Y),
        x: &Img,
        dfilter: &mut Filter,
        dy: &Dst,
    ) -> Result<(), CudnnError>
    where
        Workspace: DevicePtrMut<u8>,
        Img: DevicePtr<X>,
        Filter: DevicePtrMut<X>,
        Dst: DevicePtr<Y>,
    {
        let (num_bytes, workspace_ptr) = workspace
            .map(|x| (x.num_bytes(), *x.device_ptr_mut() as *mut std::ffi::c_void))
            .unwrap_or((0, std::ptr::null_mut()));
        let alpha = alpha.into_scaling_parameter();
        let beta = beta.into_scaling_parameter();
        result::convolution_backward_filter(
            self.conv.handle.handle,
            (&alpha) as *const Y::Scalar as *const std::ffi::c_void,
            self.x.desc,
            *x.device_ptr() as *const _,
            self.dy.desc,
            *dy.device_ptr() as *const _,
            self.conv.desc,
            algo,
            workspace_ptr,
            num_bytes,
            (&beta) as *const Y::Scalar as *const std::ffi::c_void,
            self.dw.desc,
            *dfilter.device_ptr_mut() as *mut _,
        )
    }
}
