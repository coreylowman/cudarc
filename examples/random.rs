use cudarc::prelude::*;
use cudarc::rng::CudaRng;

fn main() {
    let dev = CudaDeviceBuilder::new(0).build().unwrap();
    let rng = CudaRng::new(0, dev.clone()).unwrap();

    let mut t_dev = dev.alloc_zeros::<[f32; 10]>().unwrap();
    rng.fill_with_uniform(&mut t_dev).unwrap();
    let t_host = t_dev.into_host().unwrap();
    println!("Uniform: {:?}", t_host);

    let mut t_dev = dev.take(t_host).unwrap();
    rng.fill_with_normal(&mut t_dev, 0.0, 1.0).unwrap();
    let t_host = t_dev.into_host().unwrap();
    println!("Normal(0, 1): {:?}", t_host);

    let mut t_dev = dev.take(t_host).unwrap();
    rng.fill_with_log_normal(&mut t_dev, 0.0, 1.0).unwrap();
    let t_host = t_dev.into_host().unwrap();
    println!("LogNormal(0, 1): {:?}", t_host);
}
