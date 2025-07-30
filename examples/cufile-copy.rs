#[cfg(feature = "std")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    use cudarc::{cufile::safe::Cufile, driver::CudaContext};

    const N: usize = 100000;
    let data: Vec<u8> = (0..N).flat_map(|x| (x as f32).to_le_bytes()).collect();
    let data_sz = data.len();
    let src_file = "/tmp/cufile_test.bin";
    fs::write(src_file, &data)?;

    let cufile = Cufile::new()?;
    println!("{:?}", cufile.get_properties()?);

    let file = fs::File::open(src_file)?;
    let handle = cufile.register(file)?;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let mut buf = stream.alloc_zeros::<u8>(data_sz)?;
    stream.memcpy_ftod(&handle, 0, &mut buf)?;

    let verify_dst = stream.memcpy_dtov(&buf)?;
    assert_eq!(verify_dst, data);

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    println!("This example requires std")
}
