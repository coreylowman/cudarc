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
pub unsafe extern "ptx-kernel" fn square_kernel(input: *const f32, output: *mut f32, size: i32) {
    /* https://doc.rust-lang.org/stable/core/arch/nvptx/index.html */
    let thread_id: i32 = _thread_idx_x();
    let block_id: i32 = _block_idx_x();
    
    let block_dim: i32 = _block_dim_x();
    let grid_dim: i32 = _grid_dim_x();
    
    let n_threads = (block_dim * grid_dim) as u64;
    
    let thread_index = 
        thread_id + 
        block_id * block_dim
    ;

    if thread_index < size {
        let value = device::square(*input.offset(thread_index as isize));
        *output.offset(thread_index as isize) = value;
    }
}
