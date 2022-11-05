use crate::prelude::*;

use super::sys::*;

/// Functions for a matrix-vector multiplication. (<https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv>)
pub trait Gemv<T, M, const S: usize, const C: usize>: Sized + CublasTensor {
    type InputVector;

    /// Calculates the matrix-vector multiplication of `matrix` and `vector`.
    /// `add_to_output` decides weather the result should be added (if true)
    /// or set (if false) to `self`.
    ///
    /// # Safety
    /// Only if `add_to_output` is true `self` must be initialized.
    fn gemv(
        &mut self,
        cublas_handle: &CublasHandle,
        matrix: &M,
        vector: &Self::InputVector,
        add_to_output: bool,
    ) -> CublasResult<()>;

    /// Creates a new [CublasVector] in the `allocation`
    /// from the matrix-vector multiplication of `matrix` and `vector`.
    ///
    /// # Safety
    /// This function is safe if `allocation` is uninitialized.
    fn from_gemv(
        allocation: CudaRc<Self::Value>,
        cublas_handle: &CublasHandle,
        matrix: &M,
        vector: &Self::InputVector,
    ) -> CublasResult<Self> {
        let mut s = unsafe { Self::uninit(allocation) };
        s.gemv(cublas_handle, matrix, vector, false)?;
        Ok(s)
    }

    /// Adds the matrix-vector multiplication of `matrix` and `vector` to
    /// `self`.
    ///
    /// # Safety
    /// `self` must be initialized.
    fn add_gemv(
        &mut self,
        cublas_handle: &CublasHandle,
        matrix: &M,
        vector: &Self::InputVector,
    ) -> CublasResult<()> {
        self.gemv(cublas_handle, matrix, vector, true)
    }
}
/// Implements [Gemv] for different element types ([f32], [f64])
/// and different matrix "states" ([CublasMatrix], [CublasMatrixTransposed])
macro_rules! impl_gemv {
    (
        $type:ty,
        $struct:ident,
        $op:expr,
        $row:ident,
        $col:ident,
        $out:ident,
        $in:ident,
        $zero:literal,
        $one:literal,
        $cublas_fn:ident
    ) => {
        impl<const $row: usize, const $col: usize>
            Gemv<$type, $struct<$type, $row, $col>, $row, $col> for CublasVector<$type, $out>
        {
            type InputVector = CublasVector<$type, $in>;

            fn gemv(
                &mut self,
                cublas_handle: &CublasHandle,
                matrix: &$struct<$type, $row, $col>,
                vector: &Self::InputVector,
                add_to_output: bool,
            ) -> CublasResult<()> {
                unsafe {
                    $cublas_fn(
                        cublas_handle.0,
                        $op,
                        // m = col and n = row, as the matrix is transposed by default
                        $col as _,
                        $row as _,
                        &$one as *const _,
                        matrix.get_device_pointer() as *const _,
                        $col as _,
                        vector.get_device_pointer() as *const _,
                        1,
                        &(if add_to_output { $one } else { $zero }) as *const _,
                        self.get_device_pointer_mut() as *mut _,
                        1,
                    )
                }
                .result()
            }
        }
    };
    ($type:ty : $zero:literal, $one:literal, $cublas_fn:ident) => {
        impl_gemv!(
            $type,
            CublasMatrix,
            cublasOperation_t::CUBLAS_OP_T,
            R,
            C,
            R,
            C,
            $zero,
            $one,
            $cublas_fn
        );
        impl_gemv!(
            $type,
            CublasMatrixTransposed,
            cublasOperation_t::CUBLAS_OP_N,
            R,
            C,
            C,
            R,
            $zero,
            $one,
            $cublas_fn
        );
    };
}
impl_gemv!(f32: 0.0f32, 1.0f32, cublasSgemv_v2);
impl_gemv!(f64: 0.0f64, 1.0f64, cublasDgemv_v2);

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_add_sgemv() {
        let h_vector = [1.0, 2.0, 3.0];
        let h_matrix = [[1.0, 2.0], [0.5, 1.0], [1.0 / 3.0, 1.0]];
        let h_expected = [4.0, 9.0];

        // type-hinting `f32` in `h_vector` causes inference bugs
        let mut h_out = [1.0f32, 2.0];
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let cublas_handle = CublasHandle::create().unwrap();

        let ptr_m = unsafe { device.alloc() }.unwrap();
        let d_matrix = CublasMatrix::new(ptr_m, &h_matrix).unwrap();

        let ptr_v = unsafe { device.alloc() }.unwrap();
        let d_vector = CublasVector::new(ptr_v, &h_vector).unwrap();

        let ptr_o = unsafe { device.alloc() }.unwrap();
        let mut d_out = CublasVector::new(ptr_o, &h_out).unwrap();
        d_out
            .add_gemv(&cublas_handle, &d_matrix.transposed(), &d_vector)
            .unwrap();

        d_out.copy_to(&mut h_out).unwrap();
        assert_eq!(h_out, h_expected);
    }

    #[test]
    fn test_dgemv_transposed() {
        let h_vector = [1.0, 2.0, 3.0];
        let h_matrix = [[1.0, 0.5, 1.0 / 3.0], [2.0, 1.0, 1.0]];
        let h_expected = [3.0, 7.0];

        let mut h_out = [0.0; 2];
        unsafe {
            let device = CudaDeviceBuilder::new(0).build().unwrap();
            let cublas_handle = CublasHandle::create().unwrap();

            let ptr_m = device.alloc().unwrap();
            let d_matrix = CublasMatrix::new(ptr_m, &h_matrix).unwrap();

            let ptr_v = device.alloc().unwrap();
            let d_vector = CublasVector::new(ptr_v, &h_vector).unwrap();

            let ptr_o = device.alloc().unwrap();
            let d_out =
                CublasVector::from_gemv(ptr_o, &cublas_handle, &d_matrix, &d_vector).unwrap();

            d_out.copy_to(&mut h_out).unwrap();
        }
        assert_eq!(h_out, h_expected);
    }
}
