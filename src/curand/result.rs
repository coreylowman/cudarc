use super::sys;
use std::mem::MaybeUninit;

#[derive(Debug)]
pub struct CurandError(pub sys::curandStatus_t);

impl sys::curandStatus_t {
    /// Transforms into a [Result] of [CurandError]
    pub fn result(self) -> Result<(), CurandError> {
        match self {
            sys::curandStatus_t::CURAND_STATUS_SUCCESS => Ok(()),
            _ => Err(CurandError(self)),
        }
    }
}

impl std::fmt::Display for CurandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

pub fn create_generator() -> Result<sys::curandGenerator_t, CurandError> {
    create_generator_kind(sys::curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT)
}

pub fn create_generator_kind(
    kind: sys::curandRngType_t,
) -> Result<sys::curandGenerator_t, CurandError> {
    let mut generator = MaybeUninit::uninit();
    unsafe {
        sys::curandCreateGenerator(generator.as_mut_ptr(), kind).result()?;
        Ok(generator.assume_init())
    }
}

pub unsafe fn set_seed(generator: sys::curandGenerator_t, seed: u64) -> Result<(), CurandError> {
    sys::curandSetPseudoRandomGeneratorSeed(generator, seed).result()
}

pub unsafe fn set_stream(
    generator: sys::curandGenerator_t,
    stream: sys::cudaStream_t,
) -> Result<(), CurandError> {
    sys::curandSetStream(generator, stream).result()
}

pub unsafe fn destroy_generator(generator: sys::curandGenerator_t) -> Result<(), CurandError> {
    sys::curandDestroyGenerator(generator).result()
}

pub mod generate {
    use super::{sys, CurandError};

    pub unsafe fn uniform_f32(
        gen: sys::curandGenerator_t,
        out: *mut f32,
        num: usize,
    ) -> Result<(), CurandError> {
        sys::curandGenerateUniform(gen, out, num).result()
    }

    pub unsafe fn uniform_f64(
        gen: sys::curandGenerator_t,
        out: *mut f64,
        num: usize,
    ) -> Result<(), CurandError> {
        sys::curandGenerateUniformDouble(gen, out, num).result()
    }

    pub unsafe fn uniform_u32(
        gen: sys::curandGenerator_t,
        out: *mut u32,
        num: usize,
    ) -> Result<(), CurandError> {
        sys::curandGenerate(gen, out, num).result()
    }

    pub unsafe fn normal_f32(
        gen: sys::curandGenerator_t,
        out: *mut f32,
        num: usize,
        mean: f32,
        std: f32,
    ) -> Result<(), CurandError> {
        sys::curandGenerateNormal(gen, out, num, mean, std).result()
    }

    pub unsafe fn normal_f64(
        gen: sys::curandGenerator_t,
        out: *mut f64,
        num: usize,
        mean: f64,
        std: f64,
    ) -> Result<(), CurandError> {
        sys::curandGenerateNormalDouble(gen, out, num, mean, std).result()
    }

    pub unsafe fn log_normal_f32(
        gen: sys::curandGenerator_t,
        out: *mut f32,
        num: usize,
        mean: f32,
        std: f32,
    ) -> Result<(), CurandError> {
        sys::curandGenerateLogNormal(gen, out, num, mean, std).result()
    }

    pub unsafe fn log_normal_f64(
        gen: sys::curandGenerator_t,
        out: *mut f64,
        num: usize,
        mean: f64,
        std: f64,
    ) -> Result<(), CurandError> {
        sys::curandGenerateLogNormalDouble(gen, out, num, mean, std).result()
    }

    pub unsafe fn poisson_u32(
        gen: sys::curandGenerator_t,
        out: *mut u32,
        num: usize,
        lambda: f64,
    ) -> Result<(), CurandError> {
        sys::curandGeneratePoisson(gen, out, num, lambda).result()
    }
}

pub trait UniformFill<T> {
    unsafe fn fill(self, out: *mut T, num: usize) -> Result<(), CurandError>;
}

impl UniformFill<f32> for sys::curandGenerator_t {
    unsafe fn fill(self, out: *mut f32, num: usize) -> Result<(), CurandError> {
        generate::uniform_f32(self, out, num)
    }
}

impl UniformFill<f64> for sys::curandGenerator_t {
    unsafe fn fill(self, out: *mut f64, num: usize) -> Result<(), CurandError> {
        generate::uniform_f64(self, out, num)
    }
}

impl UniformFill<u32> for sys::curandGenerator_t {
    unsafe fn fill(self, out: *mut u32, num: usize) -> Result<(), CurandError> {
        generate::uniform_u32(self, out, num)
    }
}

pub trait NormalFill<T> {
    unsafe fn fill(self, o: *mut T, n: usize, m: T, s: T) -> Result<(), CurandError>;
}

impl NormalFill<f32> for sys::curandGenerator_t {
    unsafe fn fill(self, o: *mut f32, n: usize, m: f32, s: f32) -> Result<(), CurandError> {
        generate::normal_f32(self, o, n, m, s)
    }
}

impl NormalFill<f64> for sys::curandGenerator_t {
    unsafe fn fill(self, o: *mut f64, n: usize, m: f64, s: f64) -> Result<(), CurandError> {
        generate::normal_f64(self, o, n, m, s)
    }
}

pub trait LogNormalFill<T> {
    unsafe fn fill(self, o: *mut T, n: usize, m: T, s: T) -> Result<(), CurandError>;
}

impl LogNormalFill<f32> for sys::curandGenerator_t {
    unsafe fn fill(self, o: *mut f32, n: usize, m: f32, s: f32) -> Result<(), CurandError> {
        generate::log_normal_f32(self, o, n, m, s)
    }
}

impl LogNormalFill<f64> for sys::curandGenerator_t {
    unsafe fn fill(self, o: *mut f64, n: usize, m: f64, s: f64) -> Result<(), CurandError> {
        generate::log_normal_f64(self, o, n, m, s)
    }
}
