mod descriptor;
mod division;
mod mode;

pub use descriptor::*;
pub use division::*;
pub use mode::*;

use core::ffi::c_void;
use core::marker::PhantomData;

use crate::cudnn::sys::*;
use crate::prelude::*;

const NAN_PROPAGATION: cudnnNanPropagation_t = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;

/// A [TensorOperation] for type `T` and mode `O`.
pub struct TensorOperation<T, O> {
    descriptor: TensorOperationDescriptor,
    op: PhantomData<O>,
    data_type: PhantomData<T>,
}
impl<T: TensorDataType, O: TensorOperationMode> TensorOperation<T, O> {
    /// Creates a new [TensorOperation] for type `T` and mode `O`.
    pub fn create() -> CudaCudnnResult<Self> {
        let descriptor = TensorOperationDescriptor::create()?;
        unsafe {
            cudnnSetOpTensorDescriptor(
                descriptor.0,
                O::get_tensor_operation(),
                T::get_data_type(),
                NAN_PROPAGATION,
            )
        }
        .result()?;
        Ok(Self {
            descriptor,
            op: PhantomData,
            data_type: PhantomData,
        })
    }
}
/// This trait only exists so other trait can have default methods. Might get
/// removed in the future.
pub trait _Operation<T> {
    /// # Safety
    /// All pointers must be valid.
    #[allow(clippy::too_many_arguments)]
    unsafe fn execute_op(
        &self,
        cudnn_handle: &CudnnHandle,
        a_desc: cudnnTensorDescriptor_t,
        a: *const c_void,
        a_scale: &T,
        b: *const c_void,
        b_scale: &T,
        out: *mut c_void,
    ) -> CudaCudnnResult<()>;
}
impl<T: TensorDataType, O> _Operation<T> for TensorOperation<T, O> {
    unsafe fn execute_op(
        &self,
        cudnn_handle: &CudnnHandle,
        a_desc: cudnnTensorDescriptor_t,
        a: *const c_void,
        a_scale: &T,
        b: *const c_void,
        b_scale: &T,
        out: *mut c_void,
    ) -> CudaCudnnResult<()> {
        cudnnOpTensor(
            cudnn_handle.get_handle(),
            self.descriptor.0,
            a_scale as *const _ as *const _,
            a_desc,
            a,
            b_scale as *const _ as *const _,
            a_desc,
            b,
            &T::ZERO as *const _ as *const _,
            a_desc,
            out,
        )
        .result()
    }
}
/// A trait for single parameter [TensorOperationMode]s ([OperationSqrt]
/// and [OperationNot]).
///
/// # Safety
/// Implementing this trait must make sure that the [TensorOperationMode]
/// supports exactly one parameter.
pub unsafe trait SingleParameterOp<
    T: TensorDataType,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
>: _Operation<T>
{
    /// Executes the [TensorOperation] on the tensor `a`, scaling `a` with
    /// `scale` before running the operation.
    fn execute_with_scale(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        scale: &T,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            self.execute_op(
                cudnn_handle,
                a.get_descriptor(),
                a.get_data_ptr(),
                scale,
                a.get_data_ptr(),
                &T::ZERO,
                out.get_data_ptr_mut(),
            )
        }
    }

    /// Executes the [TensorOperation] in place on the tensor `a` (overwriting
    /// `a`), scaling `a` with `scale` before running the operation.
    fn execute_in_place_with_scale(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &mut Tensor4D<T, N, C, H, W>,
        scale: &T,
    ) -> CudaCudnnResult<()> {
        unsafe {
            self.execute_op(
                cudnn_handle,
                a.get_descriptor(),
                a.get_data_ptr(),
                scale,
                a.get_data_ptr(),
                &T::ZERO,
                a.get_data_ptr_mut(),
            )
        }
    }

    /// Executes the [TensorOperation] on the tensor `a`.
    fn execute(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        self.execute_with_scale(cudnn_handle, a, &T::ONE, out)
    }

    /// Executes the [TensorOperation] in place on the tensor `a` (overwriting
    /// `a`).
    fn execute_in_place(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        self.execute_in_place_with_scale(cudnn_handle, a, &T::ONE)
    }
}
/// A trait for multi parameter [TensorOperationMode]s (except [OperationSqrt]
/// and [OperationNot]).
///
/// # Safety
/// Implementing this trait must make sure that the [TensorOperationMode]
/// supports exactly 2 parameters.
pub unsafe trait MultiParameterOp<
    T: TensorDataType,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
>: _Operation<T>
{
    /// Executes the [TensorOperation] on the tensor `a` and `b`, scaling `a`
    /// with `a_scale` and `b` with `b_scale before running the operation.
    fn execute_with_scale(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        a_scale: &T,
        b: &Tensor4D<T, N, C, H, W>,
        b_scale: &T,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            self.execute_op(
                cudnn_handle,
                a.get_descriptor(),
                a.get_data_ptr(),
                a_scale,
                b.get_data_ptr(),
                b_scale,
                out.get_data_ptr_mut(),
            )
        }
    }

    /// Executes the [TensorOperation] on the tensor `a` and `b` in place
    /// (overwriting `a`), scaling `a` with `a_scale` and `b` with `b_scale
    /// before running the operation.
    fn execute_in_place_with_scale(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &mut Tensor4D<T, N, C, H, W>,
        a_scale: &T,
        b: &Tensor4D<T, N, C, H, W>,
        b_scale: &T,
    ) -> CudaCudnnResult<()> {
        unsafe {
            self.execute_op(
                cudnn_handle,
                a.get_descriptor(),
                a.get_data_ptr(),
                a_scale,
                b.get_data_ptr(),
                b_scale,
                a.get_data_ptr_mut(),
            )
        }
    }

    /// Executes the [TensorOperation] on the tensor `a` and `b`.
    fn execute(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        b: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        self.execute_with_scale(cudnn_handle, a, &T::ONE, b, &T::ONE, out)
    }

    /// Executes the [TensorOperation] on the tensor `a` and `b` in place
    /// (overwriting `a`).
    fn execute_in_place(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &mut Tensor4D<T, N, C, H, W>,
        b: &Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        self.execute_in_place_with_scale(cudnn_handle, a, &T::ONE, b, &T::ONE)
    }
}

macro_rules! unsafe_impl_op {
    ($op:ty : $type:ident) => {
        unsafe impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
            $type<T, N, C, H, W> for TensorOperation<T, $op>
        {
        }
    };
}
unsafe_impl_op!(OperationSqrt: SingleParameterOp);
unsafe_impl_op!(OperationNot: SingleParameterOp);
unsafe_impl_op!(OperationAdd: MultiParameterOp);
unsafe_impl_op!(OperationMul: MultiParameterOp);
unsafe_impl_op!(OperationMin: MultiParameterOp);
unsafe_impl_op!(OperationMax: MultiParameterOp);

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    fn get_information<O: TensorOperationMode>() -> (
        CudnnHandle,
        TensorOperation<f64, O>,
        Tensor4D<f64, 1, 1, 1, 6>,
        Tensor4D<f64, 1, 1, 1, 6>,
        Tensor4D<f64, 1, 1, 1, 6>,
    ) {
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn = CudnnHandle::create(&cuda).unwrap();
        let op = TensorOperation::create().unwrap();
        let a = Tensor4D::alloc_with(&cuda, [[[[
            1.0,
            2.0,
            -1.0,
            0.0,
            f64::NAN,
            f64::NEG_INFINITY,
        ]]]])
        .unwrap();
        let b =
            Tensor4D::alloc_with(&cuda, [[[[3.0, 0.0, -2.0, f64::INFINITY, 0.0, 0.4]]]]).unwrap();
        let out = unsafe { Tensor4D::alloc_uninit(&cuda) }.unwrap();
        (cudnn, op, a, b, out)
    }

    #[test]
    fn test_add() {
        let (cudnn, op, a, b, mut out) = get_information::<OperationAdd>();
        op.execute_with_scale(&cudnn, &a, &0.5, &b, &2.0, &mut out)
            .unwrap();
        let output = out.get_data().as_host().unwrap();
        assert_eq!(output[0][0][0][0], 6.5);
        assert_eq!(output[0][0][0][1], 1.0);
        assert_eq!(output[0][0][0][2], -4.5);
        assert_eq!(output[0][0][0][3], f64::INFINITY);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::NEG_INFINITY);
    }

    #[test]
    fn test_mul() {
        let (cudnn, op, mut a, b, _) = get_information::<OperationMul>();
        op.execute_in_place(&cudnn, &mut a, &b).unwrap();
        let output = a.get_data().as_host().unwrap();
        assert_eq!(output[0][0][0][0], 3.0);
        assert_eq!(output[0][0][0][1], 0.0);
        assert_eq!(output[0][0][0][2], 2.0);
        assert!(output[0][0][0][3].is_nan());
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::NEG_INFINITY);
    }

    #[test]
    fn test_min() {
        let (cudnn, op, mut a, b, _) = get_information::<OperationMin>();
        op.execute_in_place_with_scale(&cudnn, &mut a, &2.5, &b, &1.0)
            .unwrap();
        let output = a.get_data().as_host().unwrap();
        assert_eq!(output[0][0][0][0], 2.5);
        assert_eq!(output[0][0][0][1], 0.0);
        assert_eq!(output[0][0][0][2], -2.5);
        assert_eq!(output[0][0][0][3], 0.0);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::NEG_INFINITY);
    }

    #[test]
    fn test_max() {
        let (cudnn, op, a, b, mut out) = get_information::<OperationMax>();
        op.execute(&cudnn, &a, &b, &mut out).unwrap();
        let output = out.get_data().as_host().unwrap();
        assert_eq!(output[0][0][0][0], 3.0);
        assert_eq!(output[0][0][0][1], 2.0);
        assert_eq!(output[0][0][0][2], -1.0);
        assert_eq!(output[0][0][0][3], f64::INFINITY);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], 0.4);
    }

    #[test]
    fn test_sqrt() {
        let (cudnn, op, a, _, mut out) = get_information::<OperationSqrt>();
        op.execute(&cudnn, &a, &mut out).unwrap();
        let output = out.get_data().as_host().unwrap();
        assert_eq!(output[0][0][0][0], 1.0);
        assert_eq!(output[0][0][0][1], 2.0f64.sqrt());
        assert!(output[0][0][0][2].is_nan());
        assert_eq!(output[0][0][0][3], 0.0);
        assert!(output[0][0][0][4].is_nan());
        assert!(output[0][0][0][5].is_nan());
    }

    #[test]
    fn test_not() {
        let (cudnn, op, mut a, ..) = get_information::<OperationNot>();
        op.execute_in_place_with_scale(&cudnn, &mut a, &-10.0)
            .unwrap();
        let output = a.get_data().as_host().unwrap();
        assert_eq!(output[0][0][0][0], 11.0);
        assert_eq!(output[0][0][0][1], 21.0);
        assert_eq!(output[0][0][0][2], -9.0);
        assert_eq!(output[0][0][0][3], 1.0);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::NEG_INFINITY);
    }
}