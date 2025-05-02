use std::{ffi::c_void, fs, os::fd::AsRawFd};

use cudarc::{
    cufile::sys::{
        cuFileDriverClose_v2, cuFileDriverGetProperties, cuFileDriverOpen, cuFileHandleRegister,
        cuFileRead, CUfileDescr_t, CUfileDescr_t__bindgen_ty_1, CUfileDrvProps,
        CUfileFileHandleType, CUfileHandle_t, CUfileOpError,
    },
    driver::{CudaContext, CudaSlice},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 100000;
    let data: Vec<u8> = (0..N).flat_map(|x| (x as f32).to_le_bytes()).collect();
    let data_sz = data.len();
    let src_file = "/tmp/cufile_test.bin";
    fs::write(src_file, &data)?;

    unsafe {
        let driver = cuFileDriverOpen();
        println!("driver result: {:?}", driver);

        if driver.err != CUfileOpError::CU_FILE_SUCCESS {
            panic!("failed to open cufile driver: {:?}", driver.cu_err);
        }

        let mut props = CUfileDrvProps::default();
        cuFileDriverGetProperties(&mut props);
        println!("props: {:#?}", props);

        let file = fs::File::open(src_file)?;
        let fd = file.as_raw_fd();

        let mut cudaFile = CUfileDescr_t::default();
        cudaFile.type_ = CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cudaFile.handle = CUfileDescr_t__bindgen_ty_1::default();
        cudaFile.handle.fd = fd;

        let mut fh: CUfileHandle_t = std::ptr::null_mut();
        let result = cuFileHandleRegister(&mut fh, &mut cudaFile);
        println!("cuFileHandleRegister result: {:?}, fh = {:?}", result, fh);

        if result.err != CUfileOpError::CU_FILE_SUCCESS {
            panic!("failed to register cufile handle: {:?}", result.cu_err);
        }

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let cuda_buf: u64 = cudarc::driver::result::malloc_sync(data_sz)?;

        stream.synchronize()?;

        let result = cuFileRead(fh, cuda_buf as *mut c_void, data_sz, 0, 0);
        println!("cuFileRead result: {:?}", result);

        let mut verify_dst = vec![0; data_sz];

        cudarc::driver::result::memcpy_dtoh_sync(&mut verify_dst, cuda_buf)?;
        assert_eq!(verify_dst, data);

        let close_result = cuFileDriverClose_v2();
        println!("close result: {:?}", close_result);
    }

    Ok(())
}
