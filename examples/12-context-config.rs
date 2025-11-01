use cudarc::driver::{sys, CudaContext, DriverError};

fn main() -> Result<(), DriverError> {
    let dev = CudaContext::new(0)?;

    println!("=== Context Limits ===");

    // Query stack size limit
    let stack_size = dev.get_limit(sys::CUlimit::CU_LIMIT_STACK_SIZE)?;
    println!("Stack size per thread: {} bytes", stack_size);

    // Query malloc heap size
    let malloc_heap_size = dev.get_limit(sys::CUlimit::CU_LIMIT_MALLOC_HEAP_SIZE)?;
    println!(
        "Malloc heap size: {} bytes ({} MB)",
        malloc_heap_size,
        malloc_heap_size / 1024 / 1024
    );

    // Query printf buffer size
    let printf_fifo_size = dev.get_limit(sys::CUlimit::CU_LIMIT_PRINTF_FIFO_SIZE)?;
    println!(
        "Printf buffer size: {} bytes ({} KB)",
        printf_fifo_size,
        printf_fifo_size / 1024
    );

    // Query device runtime sync depth
    let dev_runtime_sync_depth = dev.get_limit(sys::CUlimit::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH)?;
    println!("Device runtime sync depth: {}", dev_runtime_sync_depth);

    // Query device runtime pending launch count
    let dev_runtime_pending_launch_count =
        dev.get_limit(sys::CUlimit::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT)?;
    println!(
        "Device runtime pending launch count: {}",
        dev_runtime_pending_launch_count
    );

    println!("\n=== Cache Configuration ===");

    // Query cache config
    let cache_config = dev.get_cache_config()?;
    let cache_str = match cache_config {
        sys::CUfunc_cache::CU_FUNC_CACHE_PREFER_NONE => "No preference",
        sys::CUfunc_cache::CU_FUNC_CACHE_PREFER_SHARED => "Prefer shared memory",
        sys::CUfunc_cache::CU_FUNC_CACHE_PREFER_L1 => "Prefer L1 cache",
        sys::CUfunc_cache::CU_FUNC_CACHE_PREFER_EQUAL => "Equal L1/shared",
    };
    println!("Current cache config: {:?} ({})", cache_config, cache_str);

    println!("\n=== Testing Limit Modification ===");

    // Demonstrate setting a limit - increase stack size
    let new_stack_size = stack_size * 2;
    println!("Setting stack size to {} bytes...", new_stack_size);
    dev.set_limit(sys::CUlimit::CU_LIMIT_STACK_SIZE, new_stack_size)?;

    let updated_stack_size = dev.get_limit(sys::CUlimit::CU_LIMIT_STACK_SIZE)?;
    println!("Updated stack size: {} bytes", updated_stack_size);

    // Restore original
    dev.set_limit(sys::CUlimit::CU_LIMIT_STACK_SIZE, stack_size)?;
    println!("Restored stack size to {} bytes", stack_size);

    println!("\n=== Testing Cache Config Modification ===");

    // Set cache config to prefer shared memory
    println!("Setting cache config to prefer shared memory...");
    dev.set_cache_config(sys::CUfunc_cache::CU_FUNC_CACHE_PREFER_SHARED)?;

    let updated_cache_config = dev.get_cache_config()?;
    println!("Updated cache config: {:?}", updated_cache_config);

    // Restore original
    dev.set_cache_config(cache_config)?;
    println!("Restored cache config to {:?}", cache_config);

    Ok(())
}
