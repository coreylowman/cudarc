use std::ffi::CString;

use super::{result, sys};

/// Create a range guard. Short hand for `Event::message(msg).range()`.
///
/// Range is started on creation, and stopped when the returned [Range] is dropped.
pub fn scoped_range<S: AsRef<str>>(msg: S) -> Range {
    Event::message(msg).range()
}

/// Mark an instant. Short hand for `Event::message(msg).mark()`
pub fn mark<S: AsRef<str>>(msg: S) {
    Event::message(msg).mark()
}

/// Builder struct; create with [`Event::message()`], and then you can set optional fields on it using:
/// - [`Event::category()`]
/// - [`Event::argb()`]
/// - [`Event::payload()`]
///
/// Finalized with [`Event::mark()`] (for marking an instant) or [`Event::range()`] (for marking a range).
///
/// Example [`Event::mark()`] usage:
/// ```no_run
/// Event::message("Hello world").mark();
/// ```
///
/// Example [`Event::range()`] usage:
/// ```no_run
/// let range = Event::message("Hello_world").argb(0xffff0000).range();
/// // ... stuff you want to mark
/// drop(range);
/// ```
#[derive(Debug)]
pub struct Event {
    category: Option<u32>,
    argb: Option<u32>,
    payload: Option<Payload>,
    message: CString,
}

impl Event {
    /// Creates a new event builder struct with an associated message.
    pub fn message<S: AsRef<str>>(message: S) -> Self {
        let message = message.as_ref();
        Self {
            category: None,
            argb: None,
            payload: None,
            message: CString::new(message).expect("Message contained nul bytes"),
        }
    }

    /// Adds a category to the event. Name the category with [`name_category()`].
    pub fn category(&mut self, category: u32) -> &mut Self {
        self.category = Some(category);
        self
    }

    /// Sets the color for the event. e.g. red is `0xffff0000`.
    pub fn argb(&mut self, argb: u32) -> &mut Self {
        self.argb = Some(argb);
        self
    }

    /// Sets the payload value for the value. E.g. `Payload::I32(42)`.
    pub fn payload(&mut self, payload: Payload) -> &mut Self {
        self.payload = Some(payload);
        self
    }
}

/// Registers a name with the specified category.
/// Example:
/// ```no_run
/// name_category(1, "forward pass");
/// name_category(2, "backward pass");
/// // ...
/// Event::message("gemm").category(1).mark();
/// // ...
/// Event::message("gemm").category(2).mark();
/// ```
pub fn name_category<S: AsRef<str>>(category: u32, name: S) {
    result::name_category(category, name);
}

/// Value associated with an [Event]
#[derive(Debug, Copy, Clone)]
pub enum Payload {
    I32(i32),
    Int64(i64),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
}

impl Event {
    /// Mark an instant in nvtx.
    pub fn mark(self) {
        let event_attrib = sys::nvtxEventAttributes_t {
            version: self.cu_version(),
            size: self.cu_size(),
            category: self.cu_category(),
            colorType: self.cu_color_type(),
            color: self.cu_color_value(),
            payloadType: self.cu_payload_type(),
            reserved0: 0,
            payload: self.cu_payload_value(),
            messageType: sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII as u32 as i32,
            message: sys::nvtxMessageValue_t {
                ascii: self.message.as_ptr(),
            },
        };
        result::mark_ex(&event_attrib)
    }

    /// Start's a [Range] notation as soon as you call this [`result::range_start_ex()`] will be called.
    /// When the returned [Range] is dropped, [`result::range_end()`] will be called.
    ///
    /// Example usage:
    /// ```no_run
    /// let guard = Event::message("hello world").range();
    /// // ... do something
    /// drop(guard);
    /// ```
    pub fn range(self) -> Range {
        let event_attrib = sys::nvtxEventAttributes_t {
            version: self.cu_version(),
            size: self.cu_size(),
            category: self.cu_category(),
            colorType: self.cu_color_type(),
            color: self.cu_color_value(),
            payloadType: self.cu_payload_type(),
            reserved0: 0,
            payload: self.cu_payload_value(),
            messageType: sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII as u32 as i32,
            message: sys::nvtxMessageValue_t {
                ascii: self.message.as_ptr(),
            },
        };
        Range {
            id: result::range_start_ex(&event_attrib),
        }
    }
}

impl Event {
    pub fn cu_color_type(&self) -> i32 {
        let color_type = match self.argb {
            None => sys::nvtxColorType_t::NVTX_COLOR_UNKNOWN,
            Some(_) => sys::nvtxColorType_t::NVTX_COLOR_ARGB,
        };
        color_type as u32 as i32
    }

    pub fn cu_color_value(&self) -> u32 {
        self.argb.unwrap_or(0)
    }

    pub fn cu_payload_type(&self) -> i32 {
        let payload_type = match self.payload {
            None => sys::nvtxPayloadType_t::NVTX_PAYLOAD_UNKNOWN,
            Some(Payload::I32(_)) => sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT32,
            Some(Payload::Int64(_)) => sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT64,
            Some(Payload::U32(_)) => sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
            Some(Payload::U64(_)) => sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
            Some(Payload::F32(_)) => sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_FLOAT,
            Some(Payload::F64(_)) => sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_DOUBLE,
        };
        payload_type as u32 as i32
    }

    pub fn cu_payload_value(&self) -> sys::nvtxEventAttributes_v2_payload_t {
        match self.payload {
            None => sys::nvtxEventAttributes_v2_payload_t { iValue: 0 },
            Some(Payload::I32(v)) => sys::nvtxEventAttributes_v2_payload_t { iValue: v },
            Some(Payload::Int64(v)) => sys::nvtxEventAttributes_v2_payload_t { llValue: v },
            Some(Payload::U32(v)) => sys::nvtxEventAttributes_v2_payload_t { uiValue: v },
            Some(Payload::U64(v)) => sys::nvtxEventAttributes_v2_payload_t { ullValue: v },
            Some(Payload::F32(v)) => sys::nvtxEventAttributes_v2_payload_t { fValue: v },
            Some(Payload::F64(v)) => sys::nvtxEventAttributes_v2_payload_t { dValue: v },
        }
    }

    pub fn cu_category(&self) -> u32 {
        self.category.unwrap_or(0)
    }

    pub fn cu_version(&self) -> u16 {
        3
    }

    pub fn cu_size(&self) -> u16 {
        std::mem::size_of::<sys::nvtxEventAttributes_t>() as u16
    }
}

/// A guard class that calls [`result::range_end()`] when it is dropped. Create/start with [`Event::range()`].
#[derive(Debug)]
pub struct Range {
    id: u64,
}

impl Drop for Range {
    fn drop(&mut self) {
        unsafe { result::range_end(self.id) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvtx_mark() {
        mark("hello mark");
    }

    #[test]
    fn test_nvtx_range() {
        let _range = scoped_range("hello range");
    }
}
