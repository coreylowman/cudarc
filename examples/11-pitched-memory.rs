use cudarc::driver::{result, sys, CudaContext, DriverError};
use std::mem;

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    println!("=== 2D Pitched Memory Example ===\n");

    // Create a 2D "image" in host memory
    // Let's say we have an 8x6 image of f32 values
    const WIDTH: usize = 8;
    const HEIGHT: usize = 6;
    const ELEM_SIZE: usize = mem::size_of::<f32>();

    let mut host_data = vec![0.0f32; WIDTH * HEIGHT];

    // Fill with a pattern: value = row * 10 + col
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            host_data[row * WIDTH + col] = (row * 10 + col) as f32;
        }
    }

    println!("Host data ({}x{}):", WIDTH, HEIGHT);
    for row in 0..HEIGHT {
        print!("  Row {}: [", row);
        for col in 0..WIDTH {
            print!("{:5.1}", host_data[row * WIDTH + col]);
            if col < WIDTH - 1 {
                print!(", ");
            }
        }
        println!("]");
    }
    println!();

    // Allocate pitched 2D memory on device
    let (device_ptr, pitch) =
        unsafe { result::malloc_pitched(WIDTH * ELEM_SIZE, HEIGHT, ELEM_SIZE as u32)? };

    println!("Allocated pitched memory:");
    println!(
        "  Requested width: {} bytes ({} elements)",
        WIDTH * ELEM_SIZE,
        WIDTH
    );
    println!(
        "  Actual pitch:    {} bytes ({:.1} elements)",
        pitch,
        pitch as f64 / ELEM_SIZE as f64
    );
    println!("  Height:          {} rows", HEIGHT);
    println!(
        "  Pitch overhead:  {} bytes per row\n",
        pitch - WIDTH * ELEM_SIZE
    );

    // Copy host data to device using 2D memcpy
    let copy_params = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
        srcHost: host_data.as_ptr() as *const _,
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: WIDTH * ELEM_SIZE, // Host data is tightly packed

        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: device_ptr,
        dstArray: std::ptr::null_mut(),
        dstPitch: pitch, // Device uses pitched memory

        WidthInBytes: WIDTH * ELEM_SIZE,
        Height: HEIGHT,
    };

    unsafe {
        result::memcpy_2d_async(&copy_params, stream.cu_stream())?;
    }

    stream.synchronize()?;
    println!("Copied host data to pitched device memory\n");

    // Allocate host memory to receive the data back
    let mut host_result = vec![0.0f32; WIDTH * HEIGHT];

    // Copy back from device to host
    let copy_back_params = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: device_ptr,
        srcArray: std::ptr::null_mut(),
        srcPitch: pitch, // Device uses pitched memory

        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
        dstHost: host_result.as_mut_ptr() as *mut _,
        dstDevice: 0,
        dstArray: std::ptr::null_mut(),
        dstPitch: WIDTH * ELEM_SIZE, // Host data is tightly packed

        WidthInBytes: WIDTH * ELEM_SIZE,
        Height: HEIGHT,
    };

    unsafe {
        result::memcpy_2d_async(&copy_back_params, stream.cu_stream())?;
    }

    stream.synchronize()?;
    println!("Copied device data back to host\n");

    // Verify the data
    println!("Verification:");
    let mut all_match = true;
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            let idx = row * WIDTH + col;
            if (host_data[idx] - host_result[idx]).abs() > 1e-6 {
                println!(
                    "  ✗ Mismatch at ({}, {}): expected {}, got {}",
                    row, col, host_data[idx], host_result[idx]
                );
                all_match = false;
            }
        }
    }

    if all_match {
        println!(" All {} elements match!", WIDTH * HEIGHT);
    }

    // Clean up
    unsafe {
        result::free_sync(device_ptr)?;
    }

    println!("\n=== 2D Sub-Region Copy Example ===\n");

    // Now demonstrate copying a sub-region (4x3 region starting at offset (2,1))
    let sub_width = 4;
    let sub_height = 3;
    let offset_x = 2;
    let offset_y = 1;

    let (device_ptr2, pitch2) =
        unsafe { result::malloc_pitched(WIDTH * ELEM_SIZE, HEIGHT, ELEM_SIZE as u32)? };

    // Copy a sub-region from host to device
    let sub_copy_params = sys::CUDA_MEMCPY2D {
        srcXInBytes: offset_x * ELEM_SIZE,
        srcY: offset_y,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
        srcHost: host_data.as_ptr() as *const _,
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: WIDTH * ELEM_SIZE,

        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: device_ptr2,
        dstArray: std::ptr::null_mut(),
        dstPitch: pitch2,

        WidthInBytes: sub_width * ELEM_SIZE,
        Height: sub_height,
    };

    unsafe {
        result::memcpy_2d_async(&sub_copy_params, stream.cu_stream())?;
    }

    stream.synchronize()?;

    println!(
        "Copied {}x{} sub-region starting at ({}, {}):",
        sub_width, sub_height, offset_x, offset_y
    );

    for row in offset_y..(offset_y + sub_height) {
        print!("  Original row {}: [", row);
        for col in offset_x..(offset_x + sub_width) {
            print!("{:5.1}", host_data[row * WIDTH + col]);
            if col < offset_x + sub_width - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    // Clean up
    unsafe {
        result::free_sync(device_ptr2)?;
    }

    println!("\n2D memory operations completed successfully!");

    Ok(())
}
