use cudarc::driver::{CudaDevice, DeviceSlice, DriverError};

use std::thread;

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    let dev1 = dev.clone();
    let thread1 = thread::spawn(move || {
        // You'll need to call bind_to_thread on each thread that the
        // device is used on.
        dev1.bind_to_thread()?;
        dev1.alloc_zeros::<f32>(100)
    });

    let dev2 = dev;
    let thread2 = thread::spawn(move || {
        // Note we have the same call here.
        dev2.bind_to_thread()?;
        dev2.alloc_zeros::<f32>(100)
    });

    let a = thread1.join().unwrap()?;
    let b = thread2.join().unwrap()?;
    assert_eq!(a.len(), b.len());

    Ok(())
}
