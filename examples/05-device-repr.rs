use cudarc::{driver::*, nvrtc::compile_ptx};

/// Here's the struct in rust, note that we have #[repr(C)]
/// here which allows us to pass it to cuda.
#[repr(C)]
struct MyCoolRustStruct {
    a: f32,
    b: f64,
    c: u32,
    d: usize,
}

/// We have to implement this to send it to cuda!
unsafe impl DeviceRepr for MyCoolRustStruct {}

const PTX_SRC: &str = "
// here's the same struct in cuda
struct MyCoolStruct {
    float a;
    double b;
    unsigned int c;
    size_t d;
};
extern \"C\" __global__ void my_custom_kernel(MyCoolStruct thing) {
    assert(thing.a == 1.0);
    assert(thing.b == 2.34);
    assert(thing.c == 57);
    assert(thing.d == 420);
}
";

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    let module = ctx.load_ptx(ptx, &["my_custom_kernel"])?;
    let f = module.get_func("my_custom_kernel").unwrap();

    // try changing some of these values to see a device assert
    let thing = MyCoolRustStruct {
        a: 1.0,
        b: 2.34,
        c: 57,
        d: 420,
    };

    let mut builder = stream.launch_builder(&f);
    // since MyCoolRustStruct implements DeviceRepr, we can pass it to launch.
    builder.arg(thing);
    unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }?;

    Ok(())
}
