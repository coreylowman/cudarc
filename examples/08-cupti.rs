use std::alloc::{alloc, dealloc, Layout};

use cudarc::{
    cupti::{
        self,
        result::{activity, subscribe, unsubscribe, CuptiError},
        sys::{self, CUptiResult},
    },
    driver,
};

// Same constants as used in cuda/extras/CUPTI/samples/common/helper_cupti_activity.h
const CUPTI_BUFFER_SIZE: usize = 8 * 1024 * 1024;
const CUPTI_BUFFER_ALIGN: usize = 8;

#[unsafe(no_mangle)]
extern "C" fn buffer_requested_callback(
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
extern "C" fn buffer_complete_callback(
    _context: driver::sys::CUcontext,
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
        match unsafe { activity::get_next_record(buffer_ptr, num_valid_bytes, &mut record) } {
            Err(CuptiError(cupti_result)) => match cupti_result {
                CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED
                | CUptiResult::CUPTI_ERROR_INVALID_KIND => break,
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
    let mut subscriber_handle: sys::CUpti_SubscriberHandle = std::ptr::null_mut();

    // Step 1: subscribe to ensure there are no other CUPTI clients.
    // Also see: https://docs.nvidia.com/cupti/main/main.html#cupti-initialization
    unsafe {
        subscribe(&mut subscriber_handle, None, std::ptr::null_mut())?;
    }

    // Step 2: set activity API buffer callbacks
    // Also see: https://docs.nvidia.com/cupti/main/main.html#cupti-activity-api
    activity::register_callbacks(
        Some(buffer_requested_callback),
        Some(buffer_complete_callback),
    )?;

    // Step 3: enable events of some kind
    activity::enable(sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER)?;

    // Step 4: do something with CUDA
    cudarc::driver::result::init().expect("unable to init cuda");
    let device_count =
        cudarc::driver::result::device::get_count().expect("unable to get device count");
    println!("device count: {device_count}");

    // Step 5: flush and observe the callbacks printing stuff to stdout.
    //
    // This requires a transmute because of a rather confusing mix of the same
    // underlying values for different enum variants in the C header.
    //
    // This will print:
    // cupti requested buffer
    // cupti completed buffer - stream id: 0, size: 8388608, valid: 40
    // cupti activity record kind: 4
    activity::flush_all(unsafe {
        std::mem::transmute::<sys::CUpti_ActivityFlag, u32>(
            sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED,
        )
    })?;

    // Step 6: clean up
    activity::disable(sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER)?;
    unsafe {
        unsubscribe(subscriber_handle)?;
    }

    Ok(())
}
