#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub use self::curandDirectionVectorSet as curandDirectionVectorSet_t;
pub use self::curandMethod as curandMethod_t;
pub use self::curandOrdering as curandOrdering_t;
pub use self::curandRngType as curandRngType_t;
pub use self::curandStatus as curandStatus_t;
pub use self::libraryPropertyType_t as libraryPropertyType;
pub type cudaStream_t = *mut CUstream_st;
pub type curandDirectionVectors32_t = [::core::ffi::c_uint; 32usize];
pub type curandDirectionVectors64_t = [::core::ffi::c_ulonglong; 64usize];
pub type curandDiscreteDistribution_t = *mut curandDiscreteDistribution_st;
pub type curandDistributionM2Shift_t = *mut curandDistributionM2Shift_st;
pub type curandDistributionShift_t = *mut curandDistributionShift_st;
pub type curandDistribution_st = f64;
pub type curandDistribution_t = *mut curandDistribution_st;
pub type curandGenerator_t = *mut curandGenerator_st;
pub type curandHistogramM2K_st = ::core::ffi::c_uint;
pub type curandHistogramM2K_t = *mut curandHistogramM2K_st;
pub type curandHistogramM2V_st = curandDistribution_st;
pub type curandHistogramM2V_t = *mut curandHistogramM2V_st;
pub type curandHistogramM2_t = *mut curandHistogramM2_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandDirectionVectorSet {
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandMethod {
    CURAND_CHOOSE_BEST = 0,
    CURAND_ITR = 1,
    CURAND_KNUTH = 2,
    CURAND_HITR = 3,
    CURAND_M1 = 4,
    CURAND_M2 = 5,
    CURAND_BINARY_SEARCH = 6,
    CURAND_DISCRETE_GAUSS = 7,
    CURAND_REJECTION = 8,
    CURAND_DEVICE_API = 9,
    CURAND_FAST_REJECTION = 10,
    CURAND_3RD = 11,
    CURAND_DEFINITION = 12,
    CURAND_POISSON = 13,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100,
    CURAND_ORDERING_PSEUDO_DEFAULT = 101,
    CURAND_ORDERING_PSEUDO_SEEDED = 102,
    CURAND_ORDERING_PSEUDO_LEGACY = 103,
    CURAND_ORDERING_QUASI_DEFAULT = 201,
}
#[cfg(any(
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100,
    CURAND_ORDERING_PSEUDO_DEFAULT = 101,
    CURAND_ORDERING_PSEUDO_SEEDED = 102,
    CURAND_ORDERING_PSEUDO_LEGACY = 103,
    CURAND_ORDERING_PSEUDO_DYNAMIC = 104,
    CURAND_ORDERING_QUASI_DEFAULT = 201,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandRngType {
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 100,
    CURAND_RNG_PSEUDO_XORWOW = 101,
    CURAND_RNG_PSEUDO_MRG32K3A = 121,
    CURAND_RNG_PSEUDO_MTGP32 = 141,
    CURAND_RNG_PSEUDO_MT19937 = 142,
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
    CURAND_RNG_QUASI_DEFAULT = 200,
    CURAND_RNG_QUASI_SOBOL32 = 201,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
    CURAND_RNG_QUASI_SOBOL64 = 203,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandStatus {
    CURAND_STATUS_SUCCESS = 0,
    CURAND_STATUS_VERSION_MISMATCH = 100,
    CURAND_STATUS_NOT_INITIALIZED = 101,
    CURAND_STATUS_ALLOCATION_FAILED = 102,
    CURAND_STATUS_TYPE_ERROR = 103,
    CURAND_STATUS_OUT_OF_RANGE = 104,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE = 201,
    CURAND_STATUS_PREEXISTING_FAILURE = 202,
    CURAND_STATUS_INITIALIZATION_FAILED = 203,
    CURAND_STATUS_ARCH_MISMATCH = 204,
    CURAND_STATUS_INTERNAL_ERROR = 999,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum libraryPropertyType_t {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL = 2,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDiscreteDistribution_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDistributionM2Shift_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDistributionShift_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandGenerator_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandHistogramM2_st {
    _unused: [u8; 0],
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn curandCreateGenerator(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t;
    pub fn curandCreateGeneratorHost(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t;
    pub fn curandCreatePoissonDistribution(
        lambda: f64,
        discrete_distribution: *mut curandDiscreteDistribution_t,
    ) -> curandStatus_t;
    pub fn curandDestroyDistribution(
        discrete_distribution: curandDiscreteDistribution_t,
    ) -> curandStatus_t;
    pub fn curandDestroyGenerator(generator: curandGenerator_t) -> curandStatus_t;
    pub fn curandGenerate(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        num: usize,
    ) -> curandStatus_t;
    pub fn curandGenerateLogNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t;
    pub fn curandGenerateLogNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> curandStatus_t;
    pub fn curandGenerateLongLong(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_ulonglong,
        num: usize,
    ) -> curandStatus_t;
    pub fn curandGenerateNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t;
    pub fn curandGenerateNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> curandStatus_t;
    pub fn curandGeneratePoisson(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        n: usize,
        lambda: f64,
    ) -> curandStatus_t;
    pub fn curandGeneratePoissonMethod(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        n: usize,
        lambda: f64,
        method: curandMethod_t,
    ) -> curandStatus_t;
    pub fn curandGenerateSeeds(generator: curandGenerator_t) -> curandStatus_t;
    pub fn curandGenerateUniform(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        num: usize,
    ) -> curandStatus_t;
    pub fn curandGenerateUniformDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        num: usize,
    ) -> curandStatus_t;
    pub fn curandGetDirectionVectors32(
        vectors: *mut *mut curandDirectionVectors32_t,
        set: curandDirectionVectorSet_t,
    ) -> curandStatus_t;
    pub fn curandGetDirectionVectors64(
        vectors: *mut *mut curandDirectionVectors64_t,
        set: curandDirectionVectorSet_t,
    ) -> curandStatus_t;
    pub fn curandGetProperty(
        type_: libraryPropertyType,
        value: *mut ::core::ffi::c_int,
    ) -> curandStatus_t;
    pub fn curandGetScrambleConstants32(constants: *mut *mut ::core::ffi::c_uint)
        -> curandStatus_t;
    pub fn curandGetScrambleConstants64(
        constants: *mut *mut ::core::ffi::c_ulonglong,
    ) -> curandStatus_t;
    pub fn curandGetVersion(version: *mut ::core::ffi::c_int) -> curandStatus_t;
    pub fn curandSetGeneratorOffset(
        generator: curandGenerator_t,
        offset: ::core::ffi::c_ulonglong,
    ) -> curandStatus_t;
    pub fn curandSetGeneratorOrdering(
        generator: curandGenerator_t,
        order: curandOrdering_t,
    ) -> curandStatus_t;
    pub fn curandSetPseudoRandomGeneratorSeed(
        generator: curandGenerator_t,
        seed: ::core::ffi::c_ulonglong,
    ) -> curandStatus_t;
    pub fn curandSetQuasiRandomGeneratorDimensions(
        generator: curandGenerator_t,
        num_dimensions: ::core::ffi::c_uint,
    ) -> curandStatus_t;
    pub fn curandSetStream(generator: curandGenerator_t, stream: cudaStream_t) -> curandStatus_t;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn curandCreateGenerator(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t {
        (culib().curandCreateGenerator)(generator, rng_type)
    }
    pub unsafe fn curandCreateGeneratorHost(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t {
        (culib().curandCreateGeneratorHost)(generator, rng_type)
    }
    pub unsafe fn curandCreatePoissonDistribution(
        lambda: f64,
        discrete_distribution: *mut curandDiscreteDistribution_t,
    ) -> curandStatus_t {
        (culib().curandCreatePoissonDistribution)(lambda, discrete_distribution)
    }
    pub unsafe fn curandDestroyDistribution(
        discrete_distribution: curandDiscreteDistribution_t,
    ) -> curandStatus_t {
        (culib().curandDestroyDistribution)(discrete_distribution)
    }
    pub unsafe fn curandDestroyGenerator(generator: curandGenerator_t) -> curandStatus_t {
        (culib().curandDestroyGenerator)(generator)
    }
    pub unsafe fn curandGenerate(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        num: usize,
    ) -> curandStatus_t {
        (culib().curandGenerate)(generator, outputPtr, num)
    }
    pub unsafe fn curandGenerateLogNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t {
        (culib().curandGenerateLogNormal)(generator, outputPtr, n, mean, stddev)
    }
    pub unsafe fn curandGenerateLogNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> curandStatus_t {
        (culib().curandGenerateLogNormalDouble)(generator, outputPtr, n, mean, stddev)
    }
    pub unsafe fn curandGenerateLongLong(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_ulonglong,
        num: usize,
    ) -> curandStatus_t {
        (culib().curandGenerateLongLong)(generator, outputPtr, num)
    }
    pub unsafe fn curandGenerateNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t {
        (culib().curandGenerateNormal)(generator, outputPtr, n, mean, stddev)
    }
    pub unsafe fn curandGenerateNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> curandStatus_t {
        (culib().curandGenerateNormalDouble)(generator, outputPtr, n, mean, stddev)
    }
    pub unsafe fn curandGeneratePoisson(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        n: usize,
        lambda: f64,
    ) -> curandStatus_t {
        (culib().curandGeneratePoisson)(generator, outputPtr, n, lambda)
    }
    pub unsafe fn curandGeneratePoissonMethod(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        n: usize,
        lambda: f64,
        method: curandMethod_t,
    ) -> curandStatus_t {
        (culib().curandGeneratePoissonMethod)(generator, outputPtr, n, lambda, method)
    }
    pub unsafe fn curandGenerateSeeds(generator: curandGenerator_t) -> curandStatus_t {
        (culib().curandGenerateSeeds)(generator)
    }
    pub unsafe fn curandGenerateUniform(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        num: usize,
    ) -> curandStatus_t {
        (culib().curandGenerateUniform)(generator, outputPtr, num)
    }
    pub unsafe fn curandGenerateUniformDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        num: usize,
    ) -> curandStatus_t {
        (culib().curandGenerateUniformDouble)(generator, outputPtr, num)
    }
    pub unsafe fn curandGetDirectionVectors32(
        vectors: *mut *mut curandDirectionVectors32_t,
        set: curandDirectionVectorSet_t,
    ) -> curandStatus_t {
        (culib().curandGetDirectionVectors32)(vectors, set)
    }
    pub unsafe fn curandGetDirectionVectors64(
        vectors: *mut *mut curandDirectionVectors64_t,
        set: curandDirectionVectorSet_t,
    ) -> curandStatus_t {
        (culib().curandGetDirectionVectors64)(vectors, set)
    }
    pub unsafe fn curandGetProperty(
        type_: libraryPropertyType,
        value: *mut ::core::ffi::c_int,
    ) -> curandStatus_t {
        (culib().curandGetProperty)(type_, value)
    }
    pub unsafe fn curandGetScrambleConstants32(
        constants: *mut *mut ::core::ffi::c_uint,
    ) -> curandStatus_t {
        (culib().curandGetScrambleConstants32)(constants)
    }
    pub unsafe fn curandGetScrambleConstants64(
        constants: *mut *mut ::core::ffi::c_ulonglong,
    ) -> curandStatus_t {
        (culib().curandGetScrambleConstants64)(constants)
    }
    pub unsafe fn curandGetVersion(version: *mut ::core::ffi::c_int) -> curandStatus_t {
        (culib().curandGetVersion)(version)
    }
    pub unsafe fn curandSetGeneratorOffset(
        generator: curandGenerator_t,
        offset: ::core::ffi::c_ulonglong,
    ) -> curandStatus_t {
        (culib().curandSetGeneratorOffset)(generator, offset)
    }
    pub unsafe fn curandSetGeneratorOrdering(
        generator: curandGenerator_t,
        order: curandOrdering_t,
    ) -> curandStatus_t {
        (culib().curandSetGeneratorOrdering)(generator, order)
    }
    pub unsafe fn curandSetPseudoRandomGeneratorSeed(
        generator: curandGenerator_t,
        seed: ::core::ffi::c_ulonglong,
    ) -> curandStatus_t {
        (culib().curandSetPseudoRandomGeneratorSeed)(generator, seed)
    }
    pub unsafe fn curandSetQuasiRandomGeneratorDimensions(
        generator: curandGenerator_t,
        num_dimensions: ::core::ffi::c_uint,
    ) -> curandStatus_t {
        (culib().curandSetQuasiRandomGeneratorDimensions)(generator, num_dimensions)
    }
    pub unsafe fn curandSetStream(
        generator: curandGenerator_t,
        stream: cudaStream_t,
    ) -> curandStatus_t {
        (culib().curandSetStream)(generator, stream)
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub curandCreateGenerator: unsafe extern "C" fn(
            generator: *mut curandGenerator_t,
            rng_type: curandRngType_t,
        ) -> curandStatus_t,
        pub curandCreateGeneratorHost: unsafe extern "C" fn(
            generator: *mut curandGenerator_t,
            rng_type: curandRngType_t,
        ) -> curandStatus_t,
        pub curandCreatePoissonDistribution: unsafe extern "C" fn(
            lambda: f64,
            discrete_distribution: *mut curandDiscreteDistribution_t,
        ) -> curandStatus_t,
        pub curandDestroyDistribution: unsafe extern "C" fn(
            discrete_distribution: curandDiscreteDistribution_t,
        ) -> curandStatus_t,
        pub curandDestroyGenerator:
            unsafe extern "C" fn(generator: curandGenerator_t) -> curandStatus_t,
        pub curandGenerate: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut ::core::ffi::c_uint,
            num: usize,
        ) -> curandStatus_t,
        pub curandGenerateLogNormal: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut f32,
            n: usize,
            mean: f32,
            stddev: f32,
        ) -> curandStatus_t,
        pub curandGenerateLogNormalDouble: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut f64,
            n: usize,
            mean: f64,
            stddev: f64,
        ) -> curandStatus_t,
        pub curandGenerateLongLong: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut ::core::ffi::c_ulonglong,
            num: usize,
        ) -> curandStatus_t,
        pub curandGenerateNormal: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut f32,
            n: usize,
            mean: f32,
            stddev: f32,
        ) -> curandStatus_t,
        pub curandGenerateNormalDouble: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut f64,
            n: usize,
            mean: f64,
            stddev: f64,
        ) -> curandStatus_t,
        pub curandGeneratePoisson: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut ::core::ffi::c_uint,
            n: usize,
            lambda: f64,
        ) -> curandStatus_t,
        pub curandGeneratePoissonMethod: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut ::core::ffi::c_uint,
            n: usize,
            lambda: f64,
            method: curandMethod_t,
        ) -> curandStatus_t,
        pub curandGenerateSeeds:
            unsafe extern "C" fn(generator: curandGenerator_t) -> curandStatus_t,
        pub curandGenerateUniform: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut f32,
            num: usize,
        ) -> curandStatus_t,
        pub curandGenerateUniformDouble: unsafe extern "C" fn(
            generator: curandGenerator_t,
            outputPtr: *mut f64,
            num: usize,
        ) -> curandStatus_t,
        pub curandGetDirectionVectors32: unsafe extern "C" fn(
            vectors: *mut *mut curandDirectionVectors32_t,
            set: curandDirectionVectorSet_t,
        ) -> curandStatus_t,
        pub curandGetDirectionVectors64: unsafe extern "C" fn(
            vectors: *mut *mut curandDirectionVectors64_t,
            set: curandDirectionVectorSet_t,
        ) -> curandStatus_t,
        pub curandGetProperty: unsafe extern "C" fn(
            type_: libraryPropertyType,
            value: *mut ::core::ffi::c_int,
        ) -> curandStatus_t,
        pub curandGetScrambleConstants32:
            unsafe extern "C" fn(constants: *mut *mut ::core::ffi::c_uint) -> curandStatus_t,
        pub curandGetScrambleConstants64:
            unsafe extern "C" fn(constants: *mut *mut ::core::ffi::c_ulonglong) -> curandStatus_t,
        pub curandGetVersion:
            unsafe extern "C" fn(version: *mut ::core::ffi::c_int) -> curandStatus_t,
        pub curandSetGeneratorOffset: unsafe extern "C" fn(
            generator: curandGenerator_t,
            offset: ::core::ffi::c_ulonglong,
        ) -> curandStatus_t,
        pub curandSetGeneratorOrdering: unsafe extern "C" fn(
            generator: curandGenerator_t,
            order: curandOrdering_t,
        ) -> curandStatus_t,
        pub curandSetPseudoRandomGeneratorSeed: unsafe extern "C" fn(
            generator: curandGenerator_t,
            seed: ::core::ffi::c_ulonglong,
        ) -> curandStatus_t,
        pub curandSetQuasiRandomGeneratorDimensions: unsafe extern "C" fn(
            generator: curandGenerator_t,
            num_dimensions: ::core::ffi::c_uint,
        )
            -> curandStatus_t,
        pub curandSetStream: unsafe extern "C" fn(
            generator: curandGenerator_t,
            stream: cudaStream_t,
        ) -> curandStatus_t,
    }
    impl Lib {
        pub unsafe fn new<P>(path: P) -> Result<Self, ::libloading::Error>
        where
            P: AsRef<::std::ffi::OsStr>,
        {
            let library = ::libloading::Library::new(path)?;
            Self::from_library(library)
        }
        pub unsafe fn from_library<L>(library: L) -> Result<Self, ::libloading::Error>
        where
            L: Into<::libloading::Library>,
        {
            let __library = library.into();
            let curandCreateGenerator = __library
                .get(b"curandCreateGenerator\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandCreateGeneratorHost = __library
                .get(b"curandCreateGeneratorHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandCreatePoissonDistribution = __library
                .get(b"curandCreatePoissonDistribution\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandDestroyDistribution = __library
                .get(b"curandDestroyDistribution\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandDestroyGenerator = __library
                .get(b"curandDestroyGenerator\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGenerate = __library
                .get(b"curandGenerate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGenerateLogNormal = __library
                .get(b"curandGenerateLogNormal\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGenerateLogNormalDouble = __library
                .get(b"curandGenerateLogNormalDouble\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGenerateLongLong = __library
                .get(b"curandGenerateLongLong\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGenerateNormal = __library
                .get(b"curandGenerateNormal\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGenerateNormalDouble = __library
                .get(b"curandGenerateNormalDouble\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGeneratePoisson = __library
                .get(b"curandGeneratePoisson\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGeneratePoissonMethod = __library
                .get(b"curandGeneratePoissonMethod\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGenerateSeeds = __library
                .get(b"curandGenerateSeeds\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGenerateUniform = __library
                .get(b"curandGenerateUniform\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGenerateUniformDouble = __library
                .get(b"curandGenerateUniformDouble\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGetDirectionVectors32 = __library
                .get(b"curandGetDirectionVectors32\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGetDirectionVectors64 = __library
                .get(b"curandGetDirectionVectors64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGetProperty = __library
                .get(b"curandGetProperty\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGetScrambleConstants32 = __library
                .get(b"curandGetScrambleConstants32\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGetScrambleConstants64 = __library
                .get(b"curandGetScrambleConstants64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandGetVersion = __library
                .get(b"curandGetVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandSetGeneratorOffset = __library
                .get(b"curandSetGeneratorOffset\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandSetGeneratorOrdering = __library
                .get(b"curandSetGeneratorOrdering\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandSetPseudoRandomGeneratorSeed = __library
                .get(b"curandSetPseudoRandomGeneratorSeed\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandSetQuasiRandomGeneratorDimensions = __library
                .get(b"curandSetQuasiRandomGeneratorDimensions\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let curandSetStream = __library
                .get(b"curandSetStream\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                curandCreateGenerator,
                curandCreateGeneratorHost,
                curandCreatePoissonDistribution,
                curandDestroyDistribution,
                curandDestroyGenerator,
                curandGenerate,
                curandGenerateLogNormal,
                curandGenerateLogNormalDouble,
                curandGenerateLongLong,
                curandGenerateNormal,
                curandGenerateNormalDouble,
                curandGeneratePoisson,
                curandGeneratePoissonMethod,
                curandGenerateSeeds,
                curandGenerateUniform,
                curandGenerateUniformDouble,
                curandGetDirectionVectors32,
                curandGetDirectionVectors64,
                curandGetProperty,
                curandGetScrambleConstants32,
                curandGetScrambleConstants64,
                curandGetVersion,
                curandSetGeneratorOffset,
                curandSetGeneratorOrdering,
                curandSetPseudoRandomGeneratorSeed,
                curandSetQuasiRandomGeneratorDimensions,
                curandSetStream,
            })
        }
    }
    pub unsafe fn is_culib_present() -> bool {
        let lib_names = ["curand"];
        let choices = lib_names
            .iter()
            .map(|l| crate::get_lib_name_candidates(l))
            .flatten();
        for choice in choices {
            if Lib::new(choice).is_ok() {
                return true;
            }
        }
        false
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["curand"];
            let choices: std::vec::Vec<_> = lib_names
                .iter()
                .map(|l| crate::get_lib_name_candidates(l))
                .flatten()
                .collect();
            for choice in choices.iter() {
                if let Ok(lib) = Lib::new(choice) {
                    return lib;
                }
            }
            crate::panic_no_lib_found(lib_names[0], &choices);
        })
    }
}
#[cfg(feature = "dynamic-loading")]
pub use loaded::*;
