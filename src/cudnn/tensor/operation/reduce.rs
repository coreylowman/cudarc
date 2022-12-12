use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::null_mut;

use alloc::rc::Rc;

use crate::cudnn::sys::*;
use crate::driver::sys::{cuMemAllocAsync, cuMemFreeAsync, CUdeviceptr};
use crate::prelude::*;

pub struct ReduceDescriptor(cudnnReduceTensorDescriptor_t);

impl ReduceDescriptor {
    pub fn create() -> CudaCudnnResult<Self> {
        let mut descriptor = MaybeUninit::uninit();
        unsafe {
            cudnnCreateReduceTensorDescriptor(descriptor.as_mut_ptr()).result()?;
            Ok(Self(descriptor.assume_init()))
        }
    }
}
impl Drop for ReduceDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyReduceTensorDescriptor(self.0) }
            .result()
            .unwrap();
    }
}
pub struct TensorReduction<
    O,
    T,
    const N_IN: usize,
    const C_IN: usize,
    const H_IN: usize,
    const W_IN: usize,
    const N_OUT: usize,
    const C_OUT: usize,
    const H_OUT: usize,
    const W_OUT: usize,
> {
    descriptor: ReduceDescriptor,
    op: PhantomData<O>,
    data_type: PhantomData<T>,
    device: Rc<CudaDevice>,
    a: Rc<TensorDescriptor<T, N_IN, C_IN, H_IN, W_IN>>,
    out: Rc<TensorDescriptor<T, N_OUT, C_OUT, H_OUT, W_OUT>>,
    workspace_allocation: CUdeviceptr,
    workspace_size: usize,
}
pub trait ReduceOp {
    fn get_op() -> cudnnReduceTensorOp_t;
}
macro_rules! impl_reduce_op {
    ($struct:ident : $op:ident) => {
        pub struct $struct;
        impl ReduceOp for $struct {
            fn get_op() -> cudnnReduceTensorOp_t {
                cudnnReduceTensorOp_t::$op
            }
        }
    };
}
impl_reduce_op!(ReduceOperationAdd: CUDNN_REDUCE_TENSOR_ADD);
impl_reduce_op!(ReduceOperationMul: CUDNN_REDUCE_TENSOR_MUL);
impl_reduce_op!(ReduceOperationMin: CUDNN_REDUCE_TENSOR_MIN);
impl_reduce_op!(ReduceOperationMax: CUDNN_REDUCE_TENSOR_MAX);
impl_reduce_op!(ReduceOperationAMax: CUDNN_REDUCE_TENSOR_AMAX);
impl_reduce_op!(ReduceOperationAvg: CUDNN_REDUCE_TENSOR_AVG);
impl_reduce_op!(ReduceOperationNorm1: CUDNN_REDUCE_TENSOR_NORM1);
impl_reduce_op!(ReduceOperationNorm2: CUDNN_REDUCE_TENSOR_NORM2);
impl_reduce_op!(ReduceOperationMulNoZeros: CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS);

impl<
        O: ReduceOp,
        T: TensorDataType,
        const N_IN: usize,
        const C_IN: usize,
        const H_IN: usize,
        const W_IN: usize,
        const N_OUT: usize,
        const C_OUT: usize,
        const H_OUT: usize,
        const W_OUT: usize,
    > TensorReduction<O, T, N_IN, C_IN, H_IN, W_IN, N_OUT, C_OUT, H_OUT, W_OUT>
where
    AssertTrue<{ is_either(N_OUT, N_IN, 1) }>: ConstTrue,
    AssertTrue<{ is_either(C_OUT, C_IN, 1) }>: ConstTrue,
    AssertTrue<{ is_either(H_OUT, H_IN, 1) }>: ConstTrue,
    AssertTrue<{ is_either(W_OUT, W_IN, 1) }>: ConstTrue,
{
    pub fn create(
        device: Rc<CudaDevice>,
        cudnn_handle: &CudnnHandle,
        a: Rc<TensorDescriptor<T, N_IN, C_IN, H_IN, W_IN>>,
        out: Rc<TensorDescriptor<T, N_OUT, C_OUT, H_OUT, W_OUT>>,
    ) -> CudaCudnnResult<Self> {
        let descriptor = ReduceDescriptor::create()?;
        unsafe {
            cudnnSetReduceTensorDescriptor(
                descriptor.0,
                O::get_op(),
                T::get_data_type(),
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_NO_INDICES,
                // random value
                cudnnIndicesType_t::CUDNN_8BIT_INDICES,
            )
            .result()?;
            let mut size = MaybeUninit::uninit();
            cudnnGetReductionWorkspaceSize(
                cudnn_handle.get_handle(),
                descriptor.0,
                a.get_descriptor(),
                out.get_descriptor(),
                size.as_mut_ptr(),
            )
            .result()?;
            let workspace_size = size.assume_init();
            let mut allocation = MaybeUninit::uninit();
            cuMemAllocAsync(allocation.as_mut_ptr(), workspace_size, device.cu_stream).result()?;
            Ok(Self {
                descriptor,
                op: PhantomData,
                data_type: PhantomData,
                device,
                a,
                out,
                workspace_allocation: allocation.assume_init(),
                workspace_size,
            })
        }
    }

    pub fn execute(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4DData<T, N_IN, C_IN, H_IN, W_IN>,
        out: &mut Tensor4DData<T, N_OUT, C_OUT, H_OUT, W_OUT>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnReduceTensor(
                cudnn_handle.get_handle(),
                self.descriptor.0,
                null_mut(),
                0,
                self.workspace_allocation as *mut _,
                self.workspace_size,
                &T::ONE as *const _ as *const _,
                self.a.get_descriptor(),
                a.get_data_ptr(),
                &T::ZERO as *const _ as *const _,
                self.out.get_descriptor(),
                out.get_data_ptr_mut(),
            )
        }
        .result()
    }
}
impl<
        O,
        T,
        const N_IN: usize,
        const C_IN: usize,
        const H_IN: usize,
        const W_IN: usize,
        const N_OUT: usize,
        const C_OUT: usize,
        const H_OUT: usize,
        const W_OUT: usize,
    > Drop for TensorReduction<O, T, N_IN, C_IN, H_IN, W_IN, N_OUT, C_OUT, H_OUT, W_OUT>
{
    fn drop(&mut self) {
        unsafe { cuMemFreeAsync(self.workspace_allocation, self.device.cu_stream) }
            .result()
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use crate::prelude::*;

    fn prepare() -> (
        Rc<CudaDevice>,
        CudnnHandle,
        Tensor4D<f64, 1, 1, 2, 3>,
        Tensor4D<f64, 1, 1, 1, 3>,
    ) {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn_handle = CudnnHandle::create(&device).unwrap();
        let a =
            Tensor4D::alloc_with(&device, [[[[1.0, 0.0, 2.0], [-1.0, -3.0, f64::NAN]]]]).unwrap();
        let out = unsafe { Tensor4D::alloc_uninit(&device) }.unwrap();
        (device, cudnn_handle, a, out)
    }

    #[test]
    fn test_reduce_add() {
        let (device, cudnn_handle, a, mut out) = prepare();
        let reduce = TensorReduction::<ReduceOperationAdd, _, _, _, _, _, _, _, _, _>::create(
            device.clone(),
            &cudnn_handle,
            a.get_descriptor_rc(),
            out.get_descriptor_rc(),
        )
        .unwrap();
        reduce
            .execute(&cudnn_handle, a.get_data_ref(), out.get_data_ref_mut())
            .unwrap();

        let output = out.get_data_ref().as_host().unwrap()[0][0][0];
        assert!(output[0].abs() < f64::EPSILON);
        assert!((output[1] - -3.0).abs() < f64::EPSILON);
        assert!(output[2].is_nan());
    }

    #[test]
    fn test_reduce_mul() {
        let (device, cudnn_handle, a, mut out) = prepare();
        let reduce = TensorReduction::<ReduceOperationMul, _, _, _, _, _, _, _, _, _>::create(
            device.clone(),
            &cudnn_handle,
            a.get_descriptor_rc(),
            out.get_descriptor_rc(),
        )
        .unwrap();
        reduce
            .execute(&cudnn_handle, a.get_data_ref(), out.get_data_ref_mut())
            .unwrap();

        let output = out.get_data_ref().as_host().unwrap()[0][0][0];
        assert!((output[0] - -1.0).abs() < f64::EPSILON);
        assert!(output[1].abs() < f64::EPSILON);
        assert!(output[2].is_nan());
    }

    #[test]
    fn test_reduce_min() {
        let (device, cudnn_handle, a, mut out) = prepare();
        let reduce = TensorReduction::<ReduceOperationMin, _, _, _, _, _, _, _, _, _>::create(
            device.clone(),
            &cudnn_handle,
            a.get_descriptor_rc(),
            out.get_descriptor_rc(),
        )
        .unwrap();
        reduce
            .execute(&cudnn_handle, a.get_data_ref(), out.get_data_ref_mut())
            .unwrap();

        let output = out.get_data_ref().as_host().unwrap()[0][0][0];
        assert!((output[0] - -1.0).abs() < f64::EPSILON);
        assert!((output[1] - -3.0).abs() < f64::EPSILON);
        assert!(output[2].is_nan());
    }

    #[test]
    fn test_reduce_max() {
        let (device, cudnn_handle, a, mut out) = prepare();
        let reduce = TensorReduction::<ReduceOperationMax, _, _, _, _, _, _, _, _, _>::create(
            device.clone(),
            &cudnn_handle,
            a.get_descriptor_rc(),
            out.get_descriptor_rc(),
        )
        .unwrap();
        reduce
            .execute(&cudnn_handle, a.get_data_ref(), out.get_data_ref_mut())
            .unwrap();

        let output = out.get_data_ref().as_host().unwrap()[0][0][0];
        assert!((output[0] - 1.0).abs() < f64::EPSILON);
        assert!(output[1].abs() < f64::EPSILON);
        assert!(output[2].is_nan());
    }

    #[test]
    fn test_reduce_amax() {
        let (device, cudnn_handle, a, mut out) = prepare();
        let reduce = TensorReduction::<ReduceOperationAMax, _, _, _, _, _, _, _, _, _>::create(
            device.clone(),
            &cudnn_handle,
            a.get_descriptor_rc(),
            out.get_descriptor_rc(),
        )
        .unwrap();
        reduce
            .execute(&cudnn_handle, a.get_data_ref(), out.get_data_ref_mut())
            .unwrap();

        let output = out.get_data_ref().as_host().unwrap()[0][0][0];
        assert!((output[0] - 1.0).abs() < f64::EPSILON);
        assert!((output[1] - 3.0).abs() < f64::EPSILON);
        assert!(output[2].is_nan());
    }

    #[test]
    fn test_reduce_avg() {
        let (device, cudnn_handle, a, mut out) = prepare();
        let reduce = TensorReduction::<ReduceOperationAvg, _, _, _, _, _, _, _, _, _>::create(
            device.clone(),
            &cudnn_handle,
            a.get_descriptor_rc(),
            out.get_descriptor_rc(),
        )
        .unwrap();
        reduce
            .execute(&cudnn_handle, a.get_data_ref(), out.get_data_ref_mut())
            .unwrap();

        let output = out.get_data_ref().as_host().unwrap()[0][0][0];
        assert!(output[0].abs() < f64::EPSILON);
        assert!((output[1] - -1.5).abs() < f64::EPSILON);
        assert!(output[2].is_nan());
    }

    #[test]
    fn test_reduce_norm1() {
        let (device, cudnn_handle, a, mut out) = prepare();
        let reduce = TensorReduction::<ReduceOperationNorm1, _, _, _, _, _, _, _, _, _>::create(
            device.clone(),
            &cudnn_handle,
            a.get_descriptor_rc(),
            out.get_descriptor_rc(),
        )
        .unwrap();
        reduce
            .execute(&cudnn_handle, a.get_data_ref(), out.get_data_ref_mut())
            .unwrap();

        let output = out.get_data_ref().as_host().unwrap()[0][0][0];
        assert!((output[0] - 2.0).abs() < f64::EPSILON);
        assert!((output[1] - 3.0).abs() < f64::EPSILON);
        assert!(output[2].is_nan());
    }

    #[test]
    fn test_reduce_norm2() {
        let (device, cudnn_handle, a, mut out) = prepare();
        let reduce = TensorReduction::<ReduceOperationNorm2, _, _, _, _, _, _, _, _, _>::create(
            device.clone(),
            &cudnn_handle,
            a.get_descriptor_rc(),
            out.get_descriptor_rc(),
        )
        .unwrap();
        reduce
            .execute(&cudnn_handle, a.get_data_ref(), out.get_data_ref_mut())
            .unwrap();

        let output = out.get_data_ref().as_host().unwrap()[0][0][0];
        assert!((output[0] - 2.0f64.sqrt()).abs() < f64::EPSILON);
        assert!((output[1] - 3.0) < f64::EPSILON);
        assert!(output[2].is_nan());
    }

    #[test]
    fn test_reduce_mul_no_zeros() {
        let (device, cudnn_handle, a, mut out) = prepare();
        let reduce =
            TensorReduction::<ReduceOperationMulNoZeros, _, _, _, _, _, _, _, _, _>::create(
                device.clone(),
                &cudnn_handle,
                a.get_descriptor_rc(),
                out.get_descriptor_rc(),
            )
            .unwrap();
        reduce
            .execute(&cudnn_handle, a.get_data_ref(), out.get_data_ref_mut())
            .unwrap();

        let output = out.get_data_ref().as_host().unwrap()[0][0][0];
        assert!((output[0] - -1.0).abs() < f64::EPSILON);
        assert!((output[1] - -3.0).abs() < f64::EPSILON);
        assert!(output[2].is_nan());
    }
}
