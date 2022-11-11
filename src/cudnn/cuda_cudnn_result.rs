use crate::prelude::*;

/// This implements `From<CudaError>` and `From<CudnnError>` so using the
/// `?`-operator coerces into `CudaCudnnError`.
#[derive(Debug, Clone)]
pub enum CudaCudnnError {
    CudaError(CudaError),
    CudnnError(CudnnError),
}
impl From<CudaError> for CudaCudnnError {
    fn from(error: CudaError) -> Self {
        Self::CudaError(error)
    }
}
impl From<CudnnError> for CudaCudnnError {
    fn from(error: CudnnError) -> Self {
        Self::CudnnError(error)
    }
}
pub trait IntoCudaCudnnResult<T> {
    fn into_cuda_cudnn_result(self) -> CudaCudnnResult<T>;
}
impl<T> IntoCudaCudnnResult<T> for Result<T, CudaError> {
    fn into_cuda_cudnn_result(self) -> CudaCudnnResult<T> {
        self.map_err(Into::into)
    }
}
pub type CudaCudnnResult<T> = Result<T, CudaCudnnError>;
