/* automatically generated by rust-bindgen 0.69.4 */

pub const CUDA_VERSION: u32 = 12000;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
pub type cudaStream_t = *mut CUstream_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum libraryPropertyType_t {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL = 2,
}
pub use self::libraryPropertyType_t as libraryPropertyType;
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
pub use self::curandStatus as curandStatus_t;
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
pub use self::curandRngType as curandRngType_t;
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
pub use self::curandOrdering as curandOrdering_t;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandDirectionVectorSet {
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104,
}
pub use self::curandDirectionVectorSet as curandDirectionVectorSet_t;
pub type curandDirectionVectors32_t = [::core::ffi::c_uint; 32usize];
pub type curandDirectionVectors64_t = [::core::ffi::c_ulonglong; 64usize];
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandGenerator_st {
    _unused: [u8; 0],
}
pub type curandGenerator_t = *mut curandGenerator_st;
pub type curandDistribution_st = f64;
pub type curandDistribution_t = *mut curandDistribution_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDistributionShift_st {
    _unused: [u8; 0],
}
pub type curandDistributionShift_t = *mut curandDistributionShift_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDistributionM2Shift_st {
    _unused: [u8; 0],
}
pub type curandDistributionM2Shift_t = *mut curandDistributionM2Shift_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandHistogramM2_st {
    _unused: [u8; 0],
}
pub type curandHistogramM2_t = *mut curandHistogramM2_st;
pub type curandHistogramM2K_st = ::core::ffi::c_uint;
pub type curandHistogramM2K_t = *mut curandHistogramM2K_st;
pub type curandHistogramM2V_st = curandDistribution_st;
pub type curandHistogramM2V_t = *mut curandHistogramM2V_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDiscreteDistribution_st {
    _unused: [u8; 0],
}
pub type curandDiscreteDistribution_t = *mut curandDiscreteDistribution_st;
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
pub use self::curandMethod as curandMethod_t;
extern "C" {
    pub fn curandCreateGenerator(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandCreateGeneratorHost(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandDestroyGenerator(generator: curandGenerator_t) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetVersion(version: *mut ::core::ffi::c_int) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetProperty(
        type_: libraryPropertyType,
        value: *mut ::core::ffi::c_int,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetStream(generator: curandGenerator_t, stream: cudaStream_t) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetPseudoRandomGeneratorSeed(
        generator: curandGenerator_t,
        seed: ::core::ffi::c_ulonglong,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetGeneratorOffset(
        generator: curandGenerator_t,
        offset: ::core::ffi::c_ulonglong,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetGeneratorOrdering(
        generator: curandGenerator_t,
        order: curandOrdering_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetQuasiRandomGeneratorDimensions(
        generator: curandGenerator_t,
        num_dimensions: ::core::ffi::c_uint,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerate(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        num: usize,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateLongLong(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_ulonglong,
        num: usize,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateUniform(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        num: usize,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateUniformDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        num: usize,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateLogNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateLogNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandCreatePoissonDistribution(
        lambda: f64,
        discrete_distribution: *mut curandDiscreteDistribution_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandDestroyDistribution(
        discrete_distribution: curandDiscreteDistribution_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGeneratePoisson(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        n: usize,
        lambda: f64,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGeneratePoissonMethod(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        n: usize,
        lambda: f64,
        method: curandMethod_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateBinomial(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        num: usize,
        n: ::core::ffi::c_uint,
        p: f64,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateBinomialMethod(
        generator: curandGenerator_t,
        outputPtr: *mut ::core::ffi::c_uint,
        num: usize,
        n: ::core::ffi::c_uint,
        p: f64,
        method: curandMethod_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateSeeds(generator: curandGenerator_t) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetDirectionVectors32(
        vectors: *mut *mut curandDirectionVectors32_t,
        set: curandDirectionVectorSet_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetScrambleConstants32(constants: *mut *mut ::core::ffi::c_uint)
        -> curandStatus_t;
}
extern "C" {
    pub fn curandGetDirectionVectors64(
        vectors: *mut *mut curandDirectionVectors64_t,
        set: curandDirectionVectorSet_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetScrambleConstants64(
        constants: *mut *mut ::core::ffi::c_ulonglong,
    ) -> curandStatus_t;
}
