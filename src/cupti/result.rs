use core::ffi::CStr;

use super::sys;

/// Wrapper around an erroneous `CUptiResult`. See
/// NVIDIA's [CUDA Runtime API](https://docs.nvidia.com/cupti/api/group__CUPTI__RESULT__API.html?highlight=CUptiResult#_CPPv411CUptiResult)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CuptiError(pub sys::CUptiResult);

impl sys::CUptiResult {
    #[inline]
    pub fn result(self) -> Result<(), CuptiError> {
        match self {
            sys::CUptiResult::CUPTI_SUCCESS => Ok(()),
            _ => Err(CuptiError(self)),
        }
    }
}

impl CuptiError {
    /// Gets the error string for this error.
    ///
    /// See [cuptiGetErrorMessage() docs](https://docs.nvidia.com/cupti/api/group__CUPTI__RESULT__API.html?highlight=cuptiGetErrorMessage#_CPPv420cuptiGetErrorMessage11CUptiResultPPKc)
    pub fn error_string(&self) -> Result<&CStr, CuptiError> {
        let mut err_str = std::ptr::null();
        unsafe {
            sys::cuptiGetErrorMessage(self.0, &mut err_str).result()?;
            Ok(CStr::from_ptr(err_str))
        }
    }
}

impl std::fmt::Debug for CuptiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_str = self.error_string().unwrap();
        f.debug_tuple("CuptiError")
            .field(&self.0)
            .field(&err_str)
            .finish()
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CuptiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CuptiError {}

/// Initialize a callback subscriber with a callback function and user data.
///
/// See [cuptiSubscribe() docs](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html?highlight=cuptiSubscribe#_CPPv414cuptiSubscribeP22CUpti_SubscriberHandle18CUpti_CallbackFuncPv)
///
/// # Safety
/// Subscriber handle must exist.
/// Callback function must exist.
pub unsafe fn subscribe(
    subscriber: *mut sys::CUpti_SubscriberHandle,
    callback: sys::CUpti_CallbackFunc,
    userdata: *mut ::std::os::raw::c_void,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiSubscribe(subscriber, callback, userdata) }.result()
}

/// Unregister a callback subscriber.
///
/// See [cuptiUnsubscribe() docs](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html?highlight=cuptiSubscribe#group__cupti__callback__api_1ga20b68c9c33f129179b56687a17356682)
///
/// # Safety
/// Subscriber must exist
pub unsafe fn unsubscribe(subscriber: sys::CUpti_SubscriberHandle) -> Result<(), CuptiError> {
    unsafe { sys::cuptiUnsubscribe(subscriber) }.result()
}

pub mod activity {
    //! Functions of the Activity API

    use super::super::{result::CuptiError, sys};

    /// Registers callback functions with CUPTI for activity buffer handling.
    ///
    /// See [cuptiActivityRegisterCallbacks() docs](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityRegisterCallbacks#_CPPv430cuptiActivityRegisterCallbacks32CUpti_BuffersCallbackRequestFunc33CUpti_BuffersCallbackCompleteFunc)
    ///
    /// Safety:
    /// Callback functions must exist.
    pub unsafe fn register_callbacks(
        func_buffer_requested: sys::CUpti_BuffersCallbackRequestFunc,
        func_buffer_completed: sys::CUpti_BuffersCallbackCompleteFunc,
    ) -> Result<(), CuptiError> {
        unsafe { sys::cuptiActivityRegisterCallbacks(func_buffer_requested, func_buffer_completed) }
            .result()
    }

    /// Request to deliver activity records via the buffer completion callback.
    ///
    /// See [cuptiActivityFlushAll() docs](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityRegisterCallbacks#_CPPv421cuptiActivityFlushAll8uint32_t)
    pub fn flush_all(flag: u32) -> Result<(), CuptiError> {
        unsafe { sys::cuptiActivityFlushAll(flag) }.result()
    }

    /// Enable collection of a specific kind of activity record.
    ///
    /// See [cuptiActivityEnable()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnable#_CPPv419cuptiActivityEnable18CUpti_ActivityKind)
    pub fn enable(kind: sys::CUpti_ActivityKind) -> Result<(), CuptiError> {
        unsafe { sys::cuptiActivityEnable(kind) }.result()
    }

    /// Iterate over the activity records in a buffer.
    ///
    /// See [cuptiActivityGetNextRecord](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnable#group__cupti__activity__api_1gab397f490a0df4a1633ea7b6e2420294f)
    ///
    /// Safety:
    /// Buffer must exist.
    /// Record pointer must exist.
    pub unsafe fn get_next_record(
        buffer: *mut u8,
        valid_buffer_size_bytes: usize,
        record: *mut *mut sys::CUpti_Activity,
    ) -> Result<(), CuptiError> {
        unsafe { sys::cuptiActivityGetNextRecord(buffer, valid_buffer_size_bytes, record) }.result()
    }
}
