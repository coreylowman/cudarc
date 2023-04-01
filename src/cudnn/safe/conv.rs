use super::core::*;
use crate::{
    cudnn::{result, result::CudnnError, sys},
    driver::{DevicePtr, DevicePtrMut},
};

use std::{marker::PhantomData, sync::Arc};

#[derive(Debug)]
pub struct FilterDescriptor<T> {
    pub(crate) desc: sys::cudnnFilterDescriptor_t,
    #[allow(unused)]
    pub(crate) handle: Arc<Cudnn>,
    pub(crate) marker: PhantomData<T>,
}

impl Cudnn {
    pub fn create_filter4d<T: CudnnDataType>(
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

#[derive(Debug)]
pub struct Conv2dDescriptor<T> {
    pub(crate) desc: sys::cudnnConvolutionDescriptor_t,
    pub(crate) handle: Arc<Cudnn>,
    pub(crate) marker: PhantomData<T>,
}

impl Cudnn {
    pub fn create_conv2d<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        pad_h: std::ffi::c_int,
        pad_w: std::ffi::c_int,
        stride_h: std::ffi::c_int,
        stride_w: std::ffi::c_int,
        dilation_h: std::ffi::c_int,
        dilation_w: std::ffi::c_int,
        mode: sys::cudnnConvolutionMode_t,
    ) -> Result<Conv2dDescriptor<T>, CudnnError> {
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

impl<T> Drop for Conv2dDescriptor<T> {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::destroy_convolution_descriptor(desc) }.unwrap()
        }
    }
}

#[derive(Debug)]
pub struct Conv2dForward<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> {
    pub conv: &'a Conv2dDescriptor<C>,
    pub img: &'a TensorDescriptor<X>,
    pub filter: &'a FilterDescriptor<X>,
    pub y: &'a TensorDescriptor<Y>,
}

impl<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> Conv2dForward<'a, X, C, Y> {
    /// Generics:
    /// - `MAX_NUM_CHOICES` - number of algorithms to look at
    pub fn pick_algorithm<const MAX_NUM_CHOICES: usize>(
        &self,
    ) -> Result<sys::cudnnConvolutionFwdAlgo_t, CudnnError> {
        let mut returned_count = [0; MAX_NUM_CHOICES];
        let mut perf_results = [Default::default(); MAX_NUM_CHOICES];
        unsafe {
            result::get_convolution_forward_algorithm(
                self.conv.handle.handle,
                self.img.desc,
                self.filter.desc,
                self.conv.desc,
                self.y.desc,
                MAX_NUM_CHOICES as std::ffi::c_int,
                returned_count.as_mut_ptr(),
                perf_results.as_mut_ptr(),
            )
        }?;
        assert!(returned_count[0] > 0);
        perf_results[0].status.result()?;
        Ok(perf_results[0].algo)
    }

    /// Returns size in **bytes**.
    pub fn get_workspace_size(
        &self,
        algo: sys::cudnnConvolutionFwdAlgo_t,
    ) -> Result<usize, CudnnError> {
        unsafe {
            result::get_convolution_forward_workspace_size(
                self.conv.handle.handle,
                self.img.desc,
                self.filter.desc,
                self.conv.desc,
                self.y.desc,
                algo,
            )
        }
    }

    pub unsafe fn launch<Workspace, Img, Filter, Dst>(
        &self,
        algo: sys::cudnnConvolutionFwdAlgo_t,
        workspace: Option<&mut Workspace>,
        alpha: Y,
        img: &Img,
        filter: &Filter,
        beta: Y,
        y: &mut Dst,
    ) -> Result<(), CudnnError>
    where
        Workspace: DevicePtrMut<u8>,
        Img: DevicePtr<X>,
        Filter: DevicePtr<X>,
        Dst: DevicePtrMut<Y>,
    {
        let (num_bytes, workspace_ptr) = workspace
            .map(|x| (x.num_bytes(), *x.device_ptr_mut() as *mut std::ffi::c_void))
            .unwrap_or((0, std::ptr::null_mut()));
        result::convolution_forward(
            self.conv.handle.handle,
            (&alpha) as *const _ as *const std::ffi::c_void,
            self.img.desc,
            *img.device_ptr() as *const std::ffi::c_void,
            self.filter.desc,
            *filter.device_ptr() as *const std::ffi::c_void,
            self.conv.desc,
            algo,
            workspace_ptr,
            num_bytes,
            (&beta) as *const _ as *const std::ffi::c_void,
            self.y.desc,
            *y.device_ptr_mut() as *mut std::ffi::c_void,
        )
    }
}

#[derive(Debug)]
pub struct Conv2dBackwardData<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> {
    pub conv: &'a Conv2dDescriptor<C>,
    pub dx: &'a TensorDescriptor<X>,
    pub filter: &'a FilterDescriptor<X>,
    pub dy: &'a TensorDescriptor<Y>,
}

impl<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> Conv2dBackwardData<'a, X, C, Y> {
    /// Generics:
    /// - `MAX_NUM_CHOICES` - number of algorithms to look at
    pub fn pick_algorithm<const MAX_NUM_CHOICES: usize>(
        &self,
    ) -> Result<sys::cudnnConvolutionBwdDataAlgo_t, CudnnError> {
        let mut returned_count = [0; MAX_NUM_CHOICES];
        let mut perf_results = [Default::default(); MAX_NUM_CHOICES];
        unsafe {
            result::get_convolution_backward_data_algorithm(
                self.conv.handle.handle,
                self.filter.desc,
                self.dy.desc,
                self.conv.desc,
                self.dx.desc,
                MAX_NUM_CHOICES as std::ffi::c_int,
                returned_count.as_mut_ptr(),
                perf_results.as_mut_ptr(),
            )
        }?;
        assert!(returned_count[0] > 0);
        perf_results[0].status.result()?;
        Ok(perf_results[0].algo)
    }

    /// Returns size in **bytes**.
    pub fn get_workspace_size(
        &self,
        algo: sys::cudnnConvolutionBwdDataAlgo_t,
    ) -> Result<usize, CudnnError> {
        unsafe {
            result::get_convolution_backward_data_workspace_size(
                self.conv.handle.handle,
                self.filter.desc,
                self.dy.desc,
                self.conv.desc,
                self.dx.desc,
                algo,
            )
        }
    }

    pub unsafe fn launch<Workspace, Img, Filter, Dst>(
        &self,
        algo: sys::cudnnConvolutionBwdDataAlgo_t,
        workspace: Option<&mut Workspace>,
        alpha: Y,
        dx: &mut Img,
        filter: &Filter,
        beta: Y,
        dy: &Dst,
    ) -> Result<(), CudnnError>
    where
        Workspace: DevicePtrMut<u8>,
        Img: DevicePtrMut<X>,
        Filter: DevicePtr<X>,
        Dst: DevicePtr<Y>,
    {
        let (num_bytes, workspace_ptr) = workspace
            .map(|x| (x.num_bytes(), *x.device_ptr_mut() as *mut std::ffi::c_void))
            .unwrap_or((0, std::ptr::null_mut()));
        result::convolution_backward_data(
            self.conv.handle.handle,
            (&alpha) as *const _ as *const std::ffi::c_void,
            self.filter.desc,
            *filter.device_ptr() as *const _,
            self.dy.desc,
            *dy.device_ptr() as *const _,
            self.conv.desc,
            algo,
            workspace_ptr,
            num_bytes,
            (&beta) as *const _ as *const std::ffi::c_void,
            self.dx.desc,
            *dx.device_ptr_mut() as *mut _,
        )
    }
}

#[derive(Debug)]
pub struct Conv2dBackwardFilter<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> {
    pub conv: &'a Conv2dDescriptor<C>,
    pub x: &'a TensorDescriptor<X>,
    pub dfilter: &'a FilterDescriptor<X>,
    pub dy: &'a TensorDescriptor<Y>,
}

impl<'a, X: CudnnDataType, C: CudnnDataType, Y: CudnnDataType> Conv2dBackwardFilter<'a, X, C, Y> {
    /// Generics:
    /// - `MAX_NUM_CHOICES` - number of algorithms to look at
    pub fn pick_algorithm<const MAX_NUM_CHOICES: usize>(
        &self,
    ) -> Result<sys::cudnnConvolutionBwdFilterAlgo_t, CudnnError> {
        let mut returned_count = [0; MAX_NUM_CHOICES];
        let mut perf_results = [Default::default(); MAX_NUM_CHOICES];
        unsafe {
            result::get_convolution_backward_filter_algorithm(
                self.conv.handle.handle,
                self.x.desc,
                self.dy.desc,
                self.conv.desc,
                self.dfilter.desc,
                MAX_NUM_CHOICES as std::ffi::c_int,
                returned_count.as_mut_ptr(),
                perf_results.as_mut_ptr(),
            )
        }?;
        assert!(returned_count[0] > 0);
        perf_results[0].status.result()?;
        Ok(perf_results[0].algo)
    }

    /// Returns size in **bytes**.
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
                self.dfilter.desc,
                algo,
            )
        }
    }

    pub unsafe fn launch<Workspace, Img, Filter, Dst>(
        &self,
        algo: sys::cudnnConvolutionBwdFilterAlgo_t,
        workspace: Option<&mut Workspace>,
        alpha: Y,
        x: &Img,
        dfilter: &mut Filter,
        beta: Y,
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
        result::convolution_backward_filter(
            self.conv.handle.handle,
            (&alpha) as *const _ as *const std::ffi::c_void,
            self.x.desc,
            *x.device_ptr() as *const _,
            self.dy.desc,
            *dy.device_ptr() as *const _,
            self.conv.desc,
            algo,
            workspace_ptr,
            num_bytes,
            (&beta) as *const _ as *const std::ffi::c_void,
            self.dfilter.desc,
            *dfilter.device_ptr_mut() as *mut _,
        )
    }
}
