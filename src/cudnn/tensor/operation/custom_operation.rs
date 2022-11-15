use crate::prelude::*;

pub trait KernelOperand<T: TensorDataType>
where
    for<'a> &'a T: IntoKernelParam,
{
    const FUNCTION_NAME_NO_SCALE: &'static str;
    const FUNCTION_NAME_WITH_SCALE: &'static str;
}
macro_rules! impl_kernel_op {
    ($type:ident : $for:ty, $base:literal) => {
        impl KernelOperand<$type> for $for {
            const FUNCTION_NAME_NO_SCALE: &'static str = concat!($base, "_", stringify!($type));
            const FUNCTION_NAME_WITH_SCALE: &'static str =
                concat!($base, "_with_scale_", stringify!($type));
        }
    };
}

/// if (<there are tokens>) {<expand to this>} else {<expand to this>}
macro_rules! if_then_else {
    (if ($($t:tt)+) {$($u:tt)*} else {$($e:tt)*}) => {
        $($u)*
    };
    (if () {$($u:tt)*} else {$($e:tt)*}) => {
        $($e)*
    };
}

/// This implements a tensor operation for type `operation` with the cudnn base
/// name `base` and additional trait bounds on. If this invocation starts with
/// an `@<token>:`, then this will be a single
macro_rules! impl_tensor_operation {
    (@single_parameter $operation:ty: $base:tt) => {
        impl_tensor_operation!(@ @single_parameter $operation: $base);
    };
    (@multi_parameter $operation:ty: $base:tt) => {
        impl_tensor_operation!(@ $operation: $base);
    };
    (@ $(@ $t:tt)? $operation:ty : $base:literal) => {
        impl_kernel_op!(f32: $operation, $base);
        impl_kernel_op!(f64: $operation, $base);

        impl<T: TensorDataType>
            Operation<T, ::alloc::rc::Rc<CudaDevice>> for $operation
        where
            Self: KernelOperand<T>,
            for<'a> &'a T: IntoKernelParam,
        {
            type MetaType = u32;
            // TODO something else?
            type Parameter = ::core::ffi::c_void;

            fn get_param_by_tensor<
                const N: usize,
                const C: usize,
                const H: usize,
                const W: usize,
            >(
                t: &Tensor4DData<T, N, C, H, W>,
            ) -> *const Self::Parameter {
                t.into_kernel_param()
            }

            fn get_param_mut_by_tensor<
                const N: usize,
                const C: usize,
                const H: usize,
                const W: usize,
            >(
                t: &mut Tensor4DData<T, N, C, H, W>,
            ) -> *mut Self::Parameter {
                t.into_kernel_param()
            }

            fn get_meta_type_by_tensor<
                const N: usize,
                const C: usize,
                const H: usize,
                const W: usize,
            >(
                t: &mut Tensor4D<T, N, C, H, W>,
            ) -> Self::MetaType {
                t.get_numel()
            }

            #[allow(unused_variables)]
            unsafe fn execute_op(
                &self,
                handle: &::alloc::rc::Rc<CudaDevice>,
                numel: Self::MetaType,
                a: *const Self::Parameter,
                a_scale: &T,
                b: *const Self::Parameter,
                b_scale: &T,
                out: *mut Self::Parameter,
            ) -> CudaCudnnResult<()> {
                let no_scale = a_scale == &T::ONE && b_scale == &T::ONE;
                let func = handle
                    .get_module(CUSTOM_KERNEL_MODULE)
                    .and_then(|m| {
                        m.get_fn(if no_scale {
                            Self::FUNCTION_NAME_NO_SCALE
                        } else {
                            Self::FUNCTION_NAME_WITH_SCALE
                        })
                    })
                    .ok_or(CudaCudnnError::CudaError(CudaError(
                        crate::driver::sys::CUresult::CUDA_ERROR_NOT_FOUND,
                    )))?;
                let cfg = LaunchConfig::for_num_elems(numel);
                if no_scale {
                    let param = if_then_else!(if ($($t)?) {(out, a, &numel)} else {(out, a, b, &numel)});
                    handle.launch_cuda_function(func, cfg, param)
                } else {
                    let param_a_scale = handle.take(::alloc::rc::Rc::new(*a_scale))?;
                    let param = if_then_else!(if ($($t)?) {(out, a, &param_a_scale, &numel)} else {(out, a, &param_a_scale, b, &handle.take(::alloc::rc::Rc::new(*b_scale))?, &numel)});
                    handle.launch_cuda_function(
                        func,
                        cfg,
                        param,
                    )
                }
                .into_cuda_cudnn_result()
            }
        }
        if_then_else! {
            if ($($t)?) {
                unsafe impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
            SingleParameterOp<T, ::alloc::rc::Rc<CudaDevice>, N, C, H, W> for $operation where Self: KernelOperand<T>,
            for<'a> &'a T: IntoKernelParam
        {
        }
            } else {
                unsafe impl<
                T: TensorDataType,
                const N_IN_OUT: usize,
                const C_IN_OUT: usize,
                const H_IN_OUT: usize,
                const W_IN_OUT: usize,
                const N_IN_B: usize,
                const C_IN_B: usize,
                const H_IN_B: usize,
                const W_IN_B: usize,
            >
            MultiParameterOp<
                T,
                ::alloc::rc::Rc<CudaDevice>,
                N_IN_OUT,
                C_IN_OUT,
                H_IN_OUT,
                W_IN_OUT,
                N_IN_B,
                C_IN_B,
                H_IN_B,
                W_IN_B,
            > for $operation
        where
            AssertEither<N_IN_OUT, N_IN_B, 1>: IsEither,
            AssertEither<C_IN_OUT, C_IN_B, 1>: IsEither,
            AssertEither<H_IN_OUT, H_IN_B, 1>: IsEither,
            AssertEither<W_IN_OUT, W_IN_B, 1>: IsEither,
            Self: KernelOperand<T>,
            for<'a> &'a T: IntoKernelParam
        {
        }
            }
        }
    };
}
