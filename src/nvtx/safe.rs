use super::sys::{
    nvtxInitialize, nvtxMarkA, nvtxNameCategoryA, nvtxNameCuContextA, nvtxNameCuDeviceA,
    nvtxNameCuEventA, nvtxNameCuStreamA, nvtxNameOsThreadA, nvtxRangeEnd, nvtxRangePop,
    nvtxRangePushA, nvtxRangeStartA, CUcontext, CUevent, CUstream,
};
use std::ffi::CString;

pub fn range_push(message: &str) -> i32 {
    unsafe {
        let message = CString::new(message).unwrap();
        nvtxRangePushA(message.as_ptr())
    }
}

pub fn range_pop() -> i32 {
    unsafe { nvtxRangePop() }
}

pub fn range_start(message: &str) -> u64 {
    unsafe {
        let message = CString::new(message).unwrap();
        nvtxRangeStartA(message.as_ptr())
    }
}

pub fn range_end(range_id: u64) {
    unsafe {
        nvtxRangeEnd(range_id);
    }
}

pub fn mark(message: &str) {
    unsafe {
        let message = CString::new(message).unwrap();
        nvtxMarkA(message.as_ptr());
    }
}

pub fn name_category(category: u32, name: &str) {
    unsafe {
        let name = CString::new(name).unwrap();
        nvtxNameCategoryA(category, name.as_ptr());
    }
}

pub fn name_os_thread(os_thread_id: u32, name: &str) {
    unsafe {
        let name = CString::new(name).unwrap();
        nvtxNameOsThreadA(os_thread_id, name.as_ptr());
    }
}

pub fn name_cu_device(cu_device: i32, name: &str) {
    unsafe {
        let name = CString::new(name).unwrap();
        nvtxNameCuDeviceA(cu_device, name.as_ptr());
    }
}

pub fn name_cu_context(cu_context: CUcontext, name: &str) {
    unsafe {
        let name = CString::new(name).unwrap();
        nvtxNameCuContextA(cu_context, name.as_ptr());
    }
}

pub fn name_cu_stream(cu_stream: CUstream, name: &str) {
    unsafe {
        let name = CString::new(name).unwrap();
        nvtxNameCuStreamA(cu_stream, name.as_ptr());
    }
}

pub fn name_cu_event(cu_event: CUevent, name: &str) {
    unsafe {
        let name = CString::new(name).unwrap();
        nvtxNameCuEventA(cu_event, name.as_ptr());
    }
}

pub fn initialize() {
    unsafe {
        // according to the doc, reserved "must be zero or NULL."
        let reserved = std::ptr::null_mut();
        nvtxInitialize(reserved);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_push() {
        range_push("Test Range");
        // Add assertions here
    }

    #[test]
    fn test_range_pop() {
        range_pop();
        // Add assertions here
    }

    #[test]
    fn test_name_cuda_event() {
        let cuda_event: CUevent = std::ptr::null_mut();
        name_cu_event(cuda_event, "Test Event");
        // Add assertions here
    }

    #[test]
    fn test_initialize() {
        initialize();
        // Add assertions here
    }

    // Add more tests here
}
