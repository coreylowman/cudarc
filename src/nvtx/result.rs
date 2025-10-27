use std::ffi::{c_int, CString};

use super::sys;

pub fn initialize() {
    unsafe { sys::nvtxInitialize(std::ptr::null()) };
}

pub fn domain_create<S: AsRef<str>>(name: S) -> sys::nvtxDomainHandle_t {
    let name = name.as_ref();
    let c_string = CString::new(name).unwrap();
    let c_str = c_string.as_c_str();
    unsafe { sys::nvtxDomainCreateA(c_str.as_ptr()) }
}

pub unsafe fn domain_destroy(domain: sys::nvtxDomainHandle_t) {
    sys::nvtxDomainDestroy(domain)
}

pub unsafe fn domain_mark(
    domain: sys::nvtxDomainHandle_t,
    event_attrib: &sys::nvtxEventAttributes_t,
) {
    sys::nvtxDomainMarkEx(domain, event_attrib)
}

pub unsafe fn domain_range_start(
    domain: sys::nvtxDomainHandle_t,
    event_attrib: &sys::nvtxEventAttributes_t,
) -> sys::nvtxRangeId_t {
    sys::nvtxDomainRangeStartEx(domain, event_attrib)
}

pub unsafe fn domain_range_end(domain: sys::nvtxDomainHandle_t, id: sys::nvtxRangeId_t) {
    sys::nvtxDomainRangeEnd(domain, id);
}

pub unsafe fn domain_range_push(
    domain: sys::nvtxDomainHandle_t,
    event_attrib: &sys::nvtxEventAttributes_t,
) -> c_int {
    sys::nvtxDomainRangePushEx(domain, event_attrib)
}

pub unsafe fn domain_range_pop(domain: sys::nvtxDomainHandle_t) -> c_int {
    sys::nvtxDomainRangePop(domain)
}

pub unsafe fn domain_name_category<S: AsRef<str>>(
    domain: sys::nvtxDomainHandle_t,
    category: u32,
    name: S,
) {
    let name = name.as_ref();
    let c_string = CString::new(name).unwrap();
    let c_str = c_string.as_c_str();
    sys::nvtxDomainNameCategoryA(domain, category, c_str.as_ptr())
}

pub unsafe fn mark<S: AsRef<str>>(message: S) {
    let message = message.as_ref();
    let c_string = CString::new(message).unwrap();
    let c_str = c_string.as_c_str();
    sys::nvtxMarkA(c_str.as_ptr())
}

pub unsafe fn mark_ex(event_attrib: &sys::nvtxEventAttributes_t) {
    sys::nvtxMarkEx(event_attrib)
}

pub unsafe fn name_os_thread<S: AsRef<str>>(thread_id: u32, name: S) {
    let name = name.as_ref();
    let c_string = CString::new(name).unwrap();
    let c_str = c_string.as_c_str();
    sys::nvtxNameOsThreadA(thread_id, c_str.as_ptr())
}

pub unsafe fn name_category<S: AsRef<str>>(category: u32, name: S) {
    let name = name.as_ref();
    let c_string = CString::new(name).unwrap();
    let c_str = c_string.as_c_str();
    sys::nvtxNameCategoryA(category, c_str.as_ptr())
}

pub unsafe fn range_start<S: AsRef<str>>(message: S) -> u64 {
    let message = message.as_ref();
    let c_string = CString::new(message).unwrap();
    let c_str = c_string.as_c_str();
    sys::nvtxRangeStartA(c_str.as_ptr())
}

pub unsafe fn range_start_ex(event_attrib: &sys::nvtxEventAttributes_t) -> u64 {
    sys::nvtxRangeStartEx(event_attrib)
}

pub unsafe fn range_end(id: u64) {
    sys::nvtxRangeEnd(id)
}

pub unsafe fn range_push<S: AsRef<str>>(message: S) -> c_int {
    let message = message.as_ref();
    let c_string = CString::new(message).unwrap();
    let c_str = c_string.as_c_str();
    sys::nvtxRangePushA(c_str.as_ptr())
}

pub unsafe fn range_push_ex(event_attrib: &sys::nvtxEventAttributes_t) -> c_int {
    sys::nvtxRangePushEx(event_attrib)
}

pub unsafe fn range_pop() -> c_int {
    sys::nvtxRangePop()
}
