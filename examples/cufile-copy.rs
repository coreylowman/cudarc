use std::{ffi::c_void, fs, os::fd::AsRawFd};

use cudarc::{
    cufile::{
        result::{driver_get_properties, driver_open, handle_register, read},
        sys::{
            cuFileDriverClose_v2, CUfileDescr_t, CUfileDescr_t__bindgen_ty_1, CUfileFileHandleType,
        },
    },
    driver::CudaContext,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 100000;
    let data: Vec<u8> = (0..N).flat_map(|x| (x as f32).to_le_bytes()).collect();
    let data_sz = data.len();
    let src_file = "/tmp/cufile_test.bin";
    fs::write(src_file, &data)?;

    unsafe {
        driver_open()?;

        let props = driver_get_properties()?;

        println!("props: {:#?}", props);

        let file = fs::File::open(src_file)?;
        let fd = file.as_raw_fd();

        let mut cuda_file = CUfileDescr_t::default();
        cuda_file.type_ = CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cuda_file.handle = CUfileDescr_t__bindgen_ty_1::default();
        cuda_file.handle.fd = fd;
        let fh = handle_register(&cuda_file)?;

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let cuda_buf: u64 = cudarc::driver::result::malloc_sync(data_sz)?;

        // stream.synchronize()?;

        let result = read(fh, cuda_buf as *mut c_void, data_sz, 0, 0)?;
        println!("cuFileRead result: {:?}", result);

        let mut verify_dst = vec![0; data_sz];

        cudarc::driver::result::memcpy_dtoh_sync(&mut verify_dst, cuda_buf)?;
        assert_eq!(verify_dst, data);

        let close_result = cuFileDriverClose_v2();
        println!("close result: {:?}", close_result);
    }

    Ok(())
}
