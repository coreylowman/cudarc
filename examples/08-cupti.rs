use std::alloc::{alloc, dealloc, Layout};

use cudarc::cupti::{self, result::CuptiError};

// Same constants as used in cuda/extras/CUPTI/samples/common/helper_cupti_activity.h
const CUPTI_BUFFER_SIZE: usize = 8 * 1024 * 1024;
const CUPTI_BUFFER_ALIGN: usize = 8;

#[unsafe(no_mangle)]
pub(crate) extern "C" fn buffer_requested_callback(
    buffer: *mut *mut u8,
    size: *mut usize,
    max_num_records: *mut usize,
) {
    println!("cupti requested buffer");
    let layout = Layout::from_size_align(CUPTI_BUFFER_SIZE, CUPTI_BUFFER_ALIGN).unwrap();
    unsafe {
        // Safety: the memory allocated here is freed in buffer_complete_callback, which
        // cupti must call.
        let ptr = alloc(layout);
        *buffer = ptr;
        *size = CUPTI_BUFFER_SIZE;
        *max_num_records = 0; // means: fill this with as many records as possible
    }
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn buffer_complete_callback(
    _context: cupti::sys::CUcontext,
    stream_id: u32,
    buffer: *mut u8,
    size: usize,
    valid_size: usize,
) {
    println!("cupti completed buffer - stream id: {stream_id}, size: {size}, valid: {valid_size}");

    print_buffer(buffer, valid_size).expect("error printing buffer");

    let layout = Layout::from_size_align(CUPTI_BUFFER_SIZE, CUPTI_BUFFER_ALIGN).unwrap();
    unsafe {
        dealloc(buffer, layout);
    }
}

fn print_buffer(buffer_ptr: *mut u8, num_valid_bytes: usize) -> Result<(), CuptiError> {
    let mut record: *mut cupti::sys::CUpti_Activity = std::ptr::null_mut();

    loop {
        match unsafe {
            cupti::result::activity::get_next_record(buffer_ptr, num_valid_bytes, &mut record)
        } {
            Err(CuptiError(e)) => match e {
                cupti::sys::CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED
                | cupti::sys::CUptiResult::CUPTI_ERROR_INVALID_KIND => break,
                other => Err(CuptiError(other))?,
            },
            Ok(_) => {
                let record = unsafe { &*record };
                println!("cupti activity record kind: {}", record.kind as u32);
            }
        };
    }

    Ok(())
}

fn main() -> Result<(), CuptiError> {
    let mut subscriber_handle: cupti::sys::CUpti_SubscriberHandle = std::ptr::null_mut();

    // Step 1: subscribe to ensure no other subscribers are using CUPTI.
    unsafe {
        cupti::result::subscribe(&mut subscriber_handle, None, std::ptr::null_mut())?;
    }

    // Step 2: set activity API buffer callbacks
    cupti::result::activity::register_callbacks(
        Some(buffer_requested_callback),
        Some(buffer_complete_callback),
    )?;

    // Step 3: enable events of some kind
    cupti::result::activity::enable(cupti::sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME)?;

    // Step 4: do something with cuda
    let device_count =
        cudarc::runtime::result::device::get_count().expect("couldn't get device count");
    println!("device count: {device_count}");

    // Step 5: flush and observe the callbacks printing stuff on stdout
    //
    // This should print something like:
    // cupti requested buffer
    // Device count: 1
    // cupti completed buffer - stream id: 0, size: 8388608, valid: 40
    // cupti activity record kind: 5
    unsafe {
        cupti::result::activity::flush_all(std::mem::transmute(
            cupti::sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED,
        ))?;
    }

    // Step 6: clean up
    unsafe {
        cupti::result::unsubscribe(subscriber_handle)?;
    }

    Ok(())
}
