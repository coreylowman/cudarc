use crate::prelude::*;

/// The default return type of all fallible cudnn operations. This has a
/// [CudaCudnnError] as the Error type as some operations, like allocations,
/// actually require cuda (and not cudnn) api calls which return a [CudaError]
/// if an error occurred.
pub type CudaCudnnResult<T> = Result<T, CudaCudnnError>;

/// This either holds a [CudaError] or a [CudnnError]. Since `604007` this is
/// the default return type of all cudnn functions.
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
