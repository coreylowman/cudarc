#![feature(abi_ptx)]        // emitting ptx (unstable)
#![feature(stdsimd)]        // simd instructions (unstable)
#![no_std]                  // CUDA compatibility

mod device;

use core::arch::nvptx::*;   // access to thread id, etc

#[panic_handler]
fn my_panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/*
    don't mangle fn name
    array: *mut i32     shared memory for writing output
    size: u32           size of array
*/

#[no_mangle]
pub unsafe extern "ptx-kernel" fn square_kernel(input: *const f32, output: *mut f32, size: u32) {
    /* https://doc.rust-lang.org/stable/core/arch/nvptx/index.html */
    let thread_id: i32 = _thread_idx_x();
    let block_id: i32 = _block_idx_x();
    let grid_dim: i32 = _grid_dim_x();
    let n_threads: i32 = _block_dim_x() * _grid_dim_x();
    
    let index = thread_id as usize;
    if index < size as usize {
        let value = device::square(*input.offset(index as isize));
        *output.offset(index as isize) = value;
    }
}
