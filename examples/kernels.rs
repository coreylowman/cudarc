use cudarc::{nvrtc::prelude::*, prelude::*};
use std::{marker::PhantomData, rc::Rc};

trait NumElements {
    const NUMEL: usize;
}
impl NumElements for f32 {
    const NUMEL: usize = 1;
}
impl<T: NumElements, const M: usize> NumElements for [T; M] {
    const NUMEL: usize = T::NUMEL * M;
}

struct Cpu;

struct ForEach<Op>(PhantomData<*const Op>);

trait BinaryKernelOp {
    const NAME: &'static str;
    const CU_SRC: &'static str;
    fn execute(&mut self, out: &mut f32, inp: &f32);
}

pub trait LaunchKernel<K, Args> {
    type Err;
    fn launch(&self, args: Args) -> Result<(), Self::Err>;
}

impl<T: Clone, Op> LaunchKernel<ForEach<Op>, (&mut Rc<T>, &Rc<T>, Op)> for Cpu
where
    Op: BinaryKernelOp,
    ForEach<Op>: ForEachCpuImpl<T, Op>,
{
    type Err = ();

    fn launch(&self, (out, inp, mut op): (&mut Rc<T>, &Rc<T>, Op)) -> Result<(), Self::Err> {
        ForEach::foreach(Rc::make_mut(out), inp.as_ref(), &mut op);
        Ok(())
    }
}

trait ForEachCpuImpl<T, Op> {
    fn foreach(out: &mut T, inp: &T, op: &mut Op);
}

impl<Op: BinaryKernelOp> ForEachCpuImpl<f32, Op> for ForEach<Op> {
    fn foreach(out: &mut f32, inp: &f32, op: &mut Op) {
        op.execute(out, inp);
    }
}

impl<T, const M: usize, Op: BinaryKernelOp> ForEachCpuImpl<[T; M], Op> for ForEach<Op>
where
    Self: ForEachCpuImpl<T, Op>,
{
    fn foreach(out: &mut [T; M], inp: &[T; M], op: &mut Op) {
        for i in 0..M {
            Self::foreach(&mut out[i], &inp[i], op);
        }
    }
}

impl<T, Op> LaunchKernel<ForEach<Op>, (&mut CudaRc<T>, &CudaRc<T>, Op)> for CudaDevice
where
    T: NumElements,
    Op: BinaryKernelOp,
{
    type Err = CudaError;
    fn launch(&self, (out, inp, _): (&mut CudaRc<T>, &CudaRc<T>, Op)) -> Result<(), Self::Err> {
        let module = self.get_module(Op::NAME).unwrap();
        let f = module.get_fn("kernel").unwrap();
        let cfg = LaunchConfig::for_num_elems(T::NUMEL as u32);
        unsafe { self.launch_cuda_function(f, cfg, (out, inp, &T::NUMEL)) }?;
        Ok(())
    }
}

pub struct SinOp;

impl BinaryKernelOp for SinOp {
    const NAME: &'static str = "SinOp";
    const CU_SRC: &'static str = "
__device__ void op(float *out, const float *in) {
    *out = sin(*in);
}
";
    fn execute(&mut self, out: &mut f32, inp: &f32) {
        *out = inp.sin();
    }
}

pub struct CosOp;

impl BinaryKernelOp for CosOp {
    const NAME: &'static str = "CosOp";
    const CU_SRC: &'static str = "
__device__ void op(float *out, const float *in) {
    *out = cos(*in);
}
";
    fn execute(&mut self, out: &mut f32, inp: &f32) {
        *out = inp.cos();
    }
}

trait CompileKernel {
    type Compiled;
    type Err;
    fn compile() -> Result<Self::Compiled, Self::Err>;
}

impl<Op: BinaryKernelOp> CompileKernel for ForEach<Op> {
    type Compiled = Ptx;
    type Err = CompileError;
    fn compile() -> Result<Self::Compiled, Self::Err> {
        let mut cu_src = format!(
            "
extern \"C\" __global__ void kernel(float *out, const float *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        op(out + i, inp + i);
    }}
}}
",
        );
        cu_src.insert_str(0, Op::CU_SRC);
        compile_ptx(cu_src)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = CudaDeviceBuilder::new(0)
        .with_nvrtc_module(SinOp::NAME, ForEach::<SinOp>::compile()?, &["kernel"])
        .with_nvrtc_module(CosOp::NAME, ForEach::<CosOp>::compile()?, &["kernel"])
        .build()
        .unwrap();

    let a_host: Rc<[f32; 3]> = Rc::new([1.0, 2.0, 3.0]);
    let mut b_host = a_host.clone();
    let mut c_host = a_host.clone();

    Cpu.launch((&mut b_host, &a_host, SinOp)).unwrap();
    Cpu.launch((&mut c_host, &a_host, CosOp)).unwrap();

    println!("cpu sin: a={a_host:?} b={b_host:?}");
    println!("cpu cos: a={a_host:?} c={c_host:?}");

    let a_dev = gpu.take(a_host.clone())?;
    let mut b_dev = a_dev.clone();
    let mut c_dev = a_dev.clone();

    gpu.launch((&mut b_dev, &a_dev, SinOp))?;
    gpu.launch((&mut c_dev, &a_dev, CosOp))?;

    let a_host_2 = a_dev.into_host()?;
    let b_host_2 = b_dev.into_host()?;
    let c_host_2 = c_dev.into_host()?;
    println!("gpu sin: a={a_host_2:?} b={b_host_2:?}");
    println!("gpu cos: a={a_host_2:?} b={c_host_2:?}");

    Ok(())
}
