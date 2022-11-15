#![allow(incomplete_features)]
#![feature(generic_const_exprs, generic_arg_infer)]

use std::time::{SystemTime, UNIX_EPOCH};

use cudarc::{cudarc::BuildError, curand::result::CurandError};

use cudarc::prelude::*;

#[derive(Debug)]
enum CudarcError {
    CudaCudnnError(CudaCudnnError),
    BuildError(BuildError),
    CurandError(CurandError)
}
macro_rules! impl_error {
    ($err:ident) => {
        impl From<$err> for CudarcError {
            fn from(value: $err) -> Self {
                Self::$err(value)
            }
        }
    }
}
impl From<CudaError> for CudarcError {
    fn from(value: CudaError) -> Self {
        Self::CudaCudnnError(value.into())
    }
}
impl_error!(BuildError);
impl_error!(CudaCudnnError);
impl_error!(CurandError);

const EPOCH: usize = 100;
const LR: f32 = 0.001;

fn main() -> Result<(), CudarcError> {
    // TODO don't clone into new device, just alloc uninit (no need to copy non-initialized data)
    let device = CudaDeviceBuilder::new(0).with_cudnn_modules().build()?;
    let cudnn_handle = CudnnHandle::create(&device)?;
    let rng = CudaRng::new(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(), device.clone())?;

    let mut x = unsafe {Tensor4D::alloc_uninit(&device)}?;
    rng.fill_with_normal(x.get_data_ref_mut().get_data_ref_mut(), 0.0, 1.0)?;

    let mut filter0 = unsafe{Filter::alloc_uninit(&device)}?;
    rng.fill_with_normal(filter0.get_data_ref_mut().get_data_ref_mut(), 0.0, 1.0)?;

    let mut yconv0 = unsafe{Tensor4D::alloc_uninit(&device)}?;
    let mut ya0 = yconv0.clone_into_new(&device)?;
    let mut yb0 = yconv0.clone_into_new(&device)?;
    let mut bias0 = unsafe {Tensor4D::<_, 1, _, _, _>::alloc_uninit(&device)}.unwrap();
    rng.fill_with_normal(bias0.get_data_ref_mut().get_data_ref_mut(), 0.0, 1.0)?;
    let mut yp0 = unsafe{Tensor4D::alloc_uninit(&device)}?;
    let mut ybn0 = yp0.clone_into_new(&device)?;

    let mut filter1 = unsafe{Filter::alloc_uninit(&device)}?;
    rng.fill_with_normal(filter1.get_data_ref_mut().get_data_ref_mut(), 0.0, 1.0)?;
    
    let mut yconv1 = unsafe{Tensor4D::alloc_uninit(&device)}?;
    let mut ya1 = yconv1.clone_into_new(&device)?;
    let mut yb1 = yconv1.clone_into_new(&device)?;
    let mut bias1 = unsafe {Tensor4D::<_, 1, _, _, _>::alloc_uninit(&device)}.unwrap();
    rng.fill_with_normal(bias1.get_data_ref_mut().get_data_ref_mut(), 0.0, 1.0)?;
    let mut yp1 = unsafe{Tensor4D::alloc_uninit(&device)}?;
    let mut ybn1 = yp1.clone_into_new(&device)?;

    let mut filter2 = unsafe{Filter::alloc_uninit(&device)}?;
    rng.fill_with_normal(filter2.get_data_ref_mut().get_data_ref_mut(), 0.0, 1.0)?;
    
    let mut yconv2 = unsafe{Tensor4D::alloc_uninit(&device)}?;
    let mut yb2 = yconv2.clone_into_new(&device)?;
    let mut bias1 = unsafe {Tensor4D::<_, 1, _, _, _>::alloc_uninit(&device)}.unwrap();
    rng.fill_with_normal(bias2.get_data_ref_mut().get_data_ref_mut(), 0.0, 1.0)?;

    let mut ysoft = yb2.clone_into_new(&device)?;

    let conv0 = Convolution2DForward::<f32, 28, 28, 0, 0, 10, 1, 16, 3, 3, 1, 1>::create(x.get_descriptor_rc(), filter0.get_descriptor_rc(), yconv0.get_descriptor_rc())?;
    let pool = MaxPooling2D::<2, 2, 0, 0, 2, 2>::create()?;
    let mut bn0 = BatchNormalizationForwardPerImage::create(&device, &cudnn_handle)?;

    let conv1 = Convolution2DForward::<f32, 13, 13, 0, 0, 10, 16, 32, 3, 3, 1, 1>::create(ybn0.get_descriptor_rc(), filter1.get_descriptor_rc(), yconv1.get_descriptor_rc())?;
    let mut bn1 = BatchNormalizationForwardPerImage::create(&device, &cudnn_handle)?;

    let conv2 = Convolution2DForward::<f32, 5, 5, 0, 0, 10, 32, 10, 5, 5, 1, 1>::create(ybn1.get_descriptor_rc(), filter2.get_descriptor_rc(), yconv2.get_descriptor_rc())?;
    let softmax = Softmax;
    let add = TensorOperation::<f32, OperationAdd>::create()?;
    let relu = Activation::<Relu>::create()?;

    let mut conv0_forward = AlgorithmWithWorkspace::create(&cudnn_handle, conv0, device.clone())?;
    let mut conv1_forward = AlgorithmWithWorkspace::create(&cudnn_handle, conv1, device.clone())?;
    let mut conv2_forward = AlgorithmWithWorkspace::create(&cudnn_handle, conv2, device.clone())?;
    
    conv0_forward.execute(&cudnn_handle, x.get_data_ref(), filter0.get_data_ref(), yconv0.get_data_ref_mut())?;
    add.execute(&cudnn_handle, &bias0, &yconv0, &mut yb0)?;
    relu.forward(&cudnn_handle, &yb0, &mut ya0)?;
    pool.forward(&cudnn_handle, &ya0, &mut yp0)?;
    bn0.train(&cudnn_handle, &yp0, &mut ybn0)?;
    
    conv1_forward.execute(&cudnn_handle, ybn0.get_data_ref(), filter1.get_data_ref(), yconv1.get_data_ref_mut())?;
    add.execute(&cudnn_handle, &bias1, &yconv1, &mut yb1)?;
    relu.forward(&cudnn_handle, &yb1, &mut ya1)?;
    pool.forward(&cudnn_handle, &ya1, &mut yp1)?;
    bn1.train(&cudnn_handle, &yp1, &mut ybn1)?;
    
    conv2_forward.execute(&cudnn_handle, ybn1.get_data_ref(), filter2.get_data_ref(), yconv2.get_data_ref_mut())?;
    add.execute(&cudnn_handle, &bias2, &yconv2, &mut yb2)?;
    softmax.forward(&cudnn_handle, &yb2, &mut ysoft)?;

    println!("{:?}", ysoft.get_data().as_host()?);
    Ok(())
}