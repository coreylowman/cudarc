#![cfg(feature = "nightly")]// nightly only due to experimental features (also improves linting linter)
#![feature(abi_ptx)]        // emitting ptx (unstable)
#![feature(stdsimd)]        // simd instructions (unstable)
#![no_std]                  // CUDA compatibility

use core::arch::nvptx::*;   // access to thread id, etc

#[panic_handler]            // CUDA compatibility (required?)
fn my_panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/*
    don't mangle fn name
    array: *mut i32     shared memory for writing output
    size: u32           size of array
*/

#[no_mangle]
pub unsafe extern "ptx-kernel" fn kernel(array: *mut i32, size: u32) {
    let thread_id: i32 = _thread_idx_x();
    let block_id: i32 = _block_idx_x();
    let grid_dim: i32 = _grid_dim_x();
    let n_threads: i32 = _block_dim_x() * _grid_dim_x();
    
    let index = thread_id as usize;
    if index < size as usize {
        let value: i32 = device(thread_id, block_id, n_threads, grid_dim);
        *array.offset(index as isize) = value;
    }
}

// an ordinary (no_std?) Rust fn that calculates using only thread info
pub fn device(thread_id: i32, block_id: i32, n_threads: i32, grid_dim: i32) -> i32 {
    // your device function implementation
    // use tid, ctaid, ntid, and nctaid to determine what to write to the array
    thread_id + block_id + n_threads + grid_dim + 100000
}
