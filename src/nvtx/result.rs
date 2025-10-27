use std::ffi::{c_int, CString};

use super::sys;

/// Force initialization - this is optional to call.
pub fn initialize() {
    unsafe { sys::nvtxInitialize(std::ptr::null()) };
}

/// Create a new domain. See [cuda docs](https://nvidia.github.io/NVTX/doxygen/group___d_o_m_a_i_n_s.html#ga2bbf44a48a4a46bf8900bd886524d87d)
pub fn domain_create<S: AsRef<str>>(name: S) -> sys::nvtxDomainHandle_t {
    let name = name.as_ref();
    let c_string = CString::new(name).expect("Found null bytes in string");
    let c_str = c_string.as_c_str();
    unsafe { sys::nvtxDomainCreateA(c_str.as_ptr()) }
}

/// Destroy an existing domain. See [cuda docs](https://nvidia.github.io/NVTX/doxygen/group___d_o_m_a_i_n_s.html#ga58b2508b5bbdfdd3cf30e4eaeb15a885)
/// # Safety
/// Ensure domain has not be freed already, and created with [domain_create]
pub unsafe fn domain_destroy(domain: sys::nvtxDomainHandle_t) {
    sys::nvtxDomainDestroy(domain)
}

/// # Safety
/// Ensure domain hasn't been destroyed.
pub unsafe fn domain_mark(
    domain: sys::nvtxDomainHandle_t,
    event_attrib: &sys::nvtxEventAttributes_t,
) {
    sys::nvtxDomainMarkEx(domain, event_attrib)
}

/// # Safety
/// Ensure domain hasn't been destroyed
pub unsafe fn domain_range_start(
    domain: sys::nvtxDomainHandle_t,
    event_attrib: &sys::nvtxEventAttributes_t,
) -> sys::nvtxRangeId_t {
    sys::nvtxDomainRangeStartEx(domain, event_attrib)
}

/// # Safety
/// Ensure domain hasn't been destroyed
pub unsafe fn domain_range_end(domain: sys::nvtxDomainHandle_t, id: sys::nvtxRangeId_t) {
    sys::nvtxDomainRangeEnd(domain, id);
}

/// # Safety
/// Ensure domain hasn't been destroyed
pub unsafe fn domain_range_push(
    domain: sys::nvtxDomainHandle_t,
    event_attrib: &sys::nvtxEventAttributes_t,
) -> c_int {
    sys::nvtxDomainRangePushEx(domain, event_attrib)
}

/// # Safety
/// Ensure domain hasn't been destroyed
pub unsafe fn domain_range_pop(domain: sys::nvtxDomainHandle_t) -> c_int {
    sys::nvtxDomainRangePop(domain)
}

/// # Safety
/// Ensure domain hasn't been destroyed
pub unsafe fn domain_name_category<S: AsRef<str>>(
    domain: sys::nvtxDomainHandle_t,
    category: u32,
    name: S,
) {
    let name = name.as_ref();
    let c_string = CString::new(name).expect("Found null bytes in str");
    let c_str = c_string.as_c_str();
    sys::nvtxDomainNameCategoryA(domain, category, c_str.as_ptr())
}

pub fn mark<S: AsRef<str>>(message: S) {
    let message = message.as_ref();
    let c_string = CString::new(message).expect("Found null bytes in str");
    let c_str = c_string.as_c_str();
    unsafe { sys::nvtxMarkA(c_str.as_ptr()) }
}

pub fn mark_ex(event_attrib: &sys::nvtxEventAttributes_t) {
    unsafe { sys::nvtxMarkEx(event_attrib) }
}

pub fn name_os_thread<S: AsRef<str>>(thread_id: u32, name: S) {
    let name = name.as_ref();
    let c_string = CString::new(name).expect("Found null bytes in str");
    let c_str = c_string.as_c_str();
    unsafe { sys::nvtxNameOsThreadA(thread_id, c_str.as_ptr()) }
}

pub fn name_category<S: AsRef<str>>(category: u32, name: S) {
    let name = name.as_ref();
    let c_string = CString::new(name).expect("Found null bytes in str");
    let c_str = c_string.as_c_str();
    unsafe { sys::nvtxNameCategoryA(category, c_str.as_ptr()) }
}

pub fn range_start<S: AsRef<str>>(message: S) -> u64 {
    let message = message.as_ref();
    let c_string = CString::new(message).expect("Found null bytes in str");
    let c_str = c_string.as_c_str();
    unsafe { sys::nvtxRangeStartA(c_str.as_ptr()) }
}

pub fn range_start_ex(event_attrib: &sys::nvtxEventAttributes_t) -> u64 {
    unsafe { sys::nvtxRangeStartEx(event_attrib) }
}

/// # Safety
/// Ensure range hasn't been ended already. Was not able to verify that this is actually unsafe.
pub unsafe fn range_end(id: u64) {
    sys::nvtxRangeEnd(id)
}

pub fn range_push<S: AsRef<str>>(message: S) -> c_int {
    let message = message.as_ref();
    let c_string = CString::new(message).expect("Found null bytes in string");
    let c_str = c_string.as_c_str();
    unsafe { sys::nvtxRangePushA(c_str.as_ptr()) }
}

pub fn range_push_ex(event_attrib: &sys::nvtxEventAttributes_t) -> c_int {
    unsafe { sys::nvtxRangePushEx(event_attrib) }
}

pub fn range_pop() -> c_int {
    unsafe { sys::nvtxRangePop() }
}
