use core::ffi::{c_int, c_longlong, c_void};
use core::mem::MaybeUninit;

use super::sys;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CufftError(pub sys::cufftResult);

impl sys::cufftResult {
    #[inline]
    pub fn result(self) -> Result<(), CufftError> {
        match self {
            sys::cufftResult::CUFFT_SUCCESS => Ok(()),
            _ => Err(CufftError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CufftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CufftError {}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftcreate)
pub fn create() -> Result<sys::cufftHandle, CufftError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cufftCreate(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftdestroy)
///
/// # Safety
/// The plan must not have been destroyed already.
pub unsafe fn destroy(plan: sys::cufftHandle) -> Result<(), CufftError> {
    sys::cufftDestroy(plan).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftestimate1d)
pub fn estimate_1d(nx: c_int, type_: sys::cufftType, batch: c_int) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    unsafe {
        sys::cufftEstimate1d(nx, type_, batch, work_size.as_mut_ptr()).result()?;
        Ok(work_size.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftestimate2d)
pub fn estimate_2d(nx: c_int, ny: c_int, type_: sys::cufftType) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    unsafe {
        sys::cufftEstimate2d(nx, ny, type_, work_size.as_mut_ptr()).result()?;
        Ok(work_size.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftestimate3d)
pub fn estimate_3d(
    nx: c_int,
    ny: c_int,
    nz: c_int,
    type_: sys::cufftType,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    unsafe {
        sys::cufftEstimate3d(nx, ny, nz, type_, work_size.as_mut_ptr()).result()?;
        Ok(work_size.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftestimatemany)
///
/// # Safety
/// Pointer parameters must be valid for reads of the expected lengths.
#[allow(clippy::too_many_arguments)]
pub unsafe fn estimate_many(
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    type_: sys::cufftType,
    batch: c_int,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftEstimateMany(
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type_,
        batch,
        work_size.as_mut_ptr(),
    )
    .result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#c.cufftExecC2C)
///
/// # Safety
/// Plan and device pointers must be valid.
pub unsafe fn exec_c2c(
    plan: sys::cufftHandle,
    idata: *mut sys::cufftComplex,
    odata: *mut sys::cufftComplex,
    direction: c_int,
) -> Result<(), CufftError> {
    sys::cufftExecC2C(plan, idata, odata, direction).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#c.cufftExecC2R)
///
/// # Safety
/// Plan and device pointers must be valid.
pub unsafe fn exec_c2r(
    plan: sys::cufftHandle,
    idata: *mut sys::cufftComplex,
    odata: *mut sys::cufftReal,
) -> Result<(), CufftError> {
    sys::cufftExecC2R(plan, idata, odata).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#c.cufftExecR2C)
///
/// # Safety
/// Plan and device pointers must be valid.
pub unsafe fn exec_r2c(
    plan: sys::cufftHandle,
    idata: *mut sys::cufftReal,
    odata: *mut sys::cufftComplex,
) -> Result<(), CufftError> {
    sys::cufftExecR2C(plan, idata, odata).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#c.cufftExecD2Z)
///
/// # Safety
/// Plan and device pointers must be valid.
pub unsafe fn exec_d2z(
    plan: sys::cufftHandle,
    idata: *mut sys::cufftDoubleReal,
    odata: *mut sys::cufftDoubleComplex,
) -> Result<(), CufftError> {
    sys::cufftExecD2Z(plan, idata, odata).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#c.cufftExecZ2D)
///
/// # Safety
/// Plan and device pointers must be valid.
pub unsafe fn exec_z2d(
    plan: sys::cufftHandle,
    idata: *mut sys::cufftDoubleComplex,
    odata: *mut sys::cufftDoubleReal,
) -> Result<(), CufftError> {
    sys::cufftExecZ2D(plan, idata, odata).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#c.cufftExecZ2Z)
///
/// # Safety
/// Plan and device pointers must be valid.
pub unsafe fn exec_z2z(
    plan: sys::cufftHandle,
    idata: *mut sys::cufftDoubleComplex,
    odata: *mut sys::cufftDoubleComplex,
    direction: c_int,
) -> Result<(), CufftError> {
    sys::cufftExecZ2Z(plan, idata, odata, direction).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftgetproperty)
pub fn get_property(property: sys::libraryPropertyType) -> Result<c_int, CufftError> {
    let mut value = MaybeUninit::uninit();
    unsafe {
        sys::cufftGetProperty(property, value.as_mut_ptr()).result()?;
        Ok(value.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftgetversion)
pub fn get_version() -> Result<c_int, CufftError> {
    let mut version = MaybeUninit::uninit();
    unsafe {
        sys::cufftGetVersion(version.as_mut_ptr()).result()?;
        Ok(version.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftgetsize)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn get_size(plan: sys::cufftHandle) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftGetSize(plan, work_size.as_mut_ptr()).result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftgetsize1d)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn get_size_1d(
    plan: sys::cufftHandle,
    nx: c_int,
    type_: sys::cufftType,
    batch: c_int,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftGetSize1d(plan, nx, type_, batch, work_size.as_mut_ptr()).result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftgetsize2d)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn get_size_2d(
    plan: sys::cufftHandle,
    nx: c_int,
    ny: c_int,
    type_: sys::cufftType,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftGetSize2d(plan, nx, ny, type_, work_size.as_mut_ptr()).result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftgetsize3d)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn get_size_3d(
    plan: sys::cufftHandle,
    nx: c_int,
    ny: c_int,
    nz: c_int,
    type_: sys::cufftType,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftGetSize3d(plan, nx, ny, nz, type_, work_size.as_mut_ptr()).result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftgetsizemany)
///
/// # Safety
/// Plan must be valid and pointer parameters must be readable.
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_size_many(
    plan: sys::cufftHandle,
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    type_: sys::cufftType,
    batch: c_int,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftGetSizeMany(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type_,
        batch,
        work_size.as_mut_ptr(),
    )
    .result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftgetsizemany64)
///
/// # Safety
/// Plan must be valid and pointer parameters must be readable.
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_size_many64(
    plan: sys::cufftHandle,
    rank: c_int,
    n: *mut c_longlong,
    inembed: *mut c_longlong,
    istride: c_longlong,
    idist: c_longlong,
    onembed: *mut c_longlong,
    ostride: c_longlong,
    odist: c_longlong,
    type_: sys::cufftType,
    batch: c_longlong,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftGetSizeMany64(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type_,
        batch,
        work_size.as_mut_ptr(),
    )
    .result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftmakeplan1d)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn make_plan_1d(
    plan: sys::cufftHandle,
    nx: c_int,
    type_: sys::cufftType,
    batch: c_int,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftMakePlan1d(plan, nx, type_, batch, work_size.as_mut_ptr()).result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftmakeplan2d)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn make_plan_2d(
    plan: sys::cufftHandle,
    nx: c_int,
    ny: c_int,
    type_: sys::cufftType,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftMakePlan2d(plan, nx, ny, type_, work_size.as_mut_ptr()).result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftmakeplan3d)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn make_plan_3d(
    plan: sys::cufftHandle,
    nx: c_int,
    ny: c_int,
    nz: c_int,
    type_: sys::cufftType,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftMakePlan3d(plan, nx, ny, nz, type_, work_size.as_mut_ptr()).result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftmakeplanmany)
///
/// # Safety
/// Plan must be valid and pointer parameters must be readable.
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_plan_many(
    plan: sys::cufftHandle,
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    type_: sys::cufftType,
    batch: c_int,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftMakePlanMany(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type_,
        batch,
        work_size.as_mut_ptr(),
    )
    .result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftmakeplanmany64)
///
/// # Safety
/// Plan must be valid and pointer parameters must be readable.
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_plan_many64(
    plan: sys::cufftHandle,
    rank: c_int,
    n: *mut c_longlong,
    inembed: *mut c_longlong,
    istride: c_longlong,
    idist: c_longlong,
    onembed: *mut c_longlong,
    ostride: c_longlong,
    odist: c_longlong,
    type_: sys::cufftType,
    batch: c_longlong,
) -> Result<usize, CufftError> {
    let mut work_size = MaybeUninit::uninit();
    sys::cufftMakePlanMany64(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type_,
        batch,
        work_size.as_mut_ptr(),
    )
    .result()?;
    Ok(work_size.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftplan1d)
pub fn plan_1d(
    nx: c_int,
    type_: sys::cufftType,
    batch: c_int,
) -> Result<sys::cufftHandle, CufftError> {
    let mut plan = MaybeUninit::uninit();
    unsafe {
        sys::cufftPlan1d(plan.as_mut_ptr(), nx, type_, batch).result()?;
        Ok(plan.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftplan2d)
pub fn plan_2d(
    nx: c_int,
    ny: c_int,
    type_: sys::cufftType,
) -> Result<sys::cufftHandle, CufftError> {
    let mut plan = MaybeUninit::uninit();
    unsafe {
        sys::cufftPlan2d(plan.as_mut_ptr(), nx, ny, type_).result()?;
        Ok(plan.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftplan3d)
pub fn plan_3d(
    nx: c_int,
    ny: c_int,
    nz: c_int,
    type_: sys::cufftType,
) -> Result<sys::cufftHandle, CufftError> {
    let mut plan = MaybeUninit::uninit();
    unsafe {
        sys::cufftPlan3d(plan.as_mut_ptr(), nx, ny, nz, type_).result()?;
        Ok(plan.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftplanmany)
///
/// # Safety
/// Pointer parameters must be valid for reads of the expected lengths.
#[allow(clippy::too_many_arguments)]
pub unsafe fn plan_many(
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    type_: sys::cufftType,
    batch: c_int,
) -> Result<sys::cufftHandle, CufftError> {
    let mut plan = MaybeUninit::uninit();
    sys::cufftPlanMany(
        plan.as_mut_ptr(),
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type_,
        batch,
    )
    .result()?;
    Ok(plan.assume_init())
}

#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftgetplanpropertyint64)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn get_plan_property_i64(
    plan: sys::cufftHandle,
    property: sys::cufftProperty,
) -> Result<c_longlong, CufftError> {
    let mut value = MaybeUninit::uninit();
    sys::cufftGetPlanPropertyInt64(plan, property, value.as_mut_ptr()).result()?;
    Ok(value.assume_init())
}

#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftresetplanproperty)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn reset_plan_property(
    plan: sys::cufftHandle,
    property: sys::cufftProperty,
) -> Result<(), CufftError> {
    sys::cufftResetPlanProperty(plan, property).result()
}

#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftsetplanpropertyint64)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn set_plan_property_i64(
    plan: sys::cufftHandle,
    property: sys::cufftProperty,
    value: c_longlong,
) -> Result<(), CufftError> {
    sys::cufftSetPlanPropertyInt64(plan, property, value).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftsetautoallocation)
///
/// # Safety
/// Plan must be valid.
pub unsafe fn set_auto_allocation(
    plan: sys::cufftHandle,
    auto_allocate: bool,
) -> Result<(), CufftError> {
    sys::cufftSetAutoAllocation(plan, auto_allocate as c_int).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftsetstream)
///
/// # Safety
/// Plan and stream must be valid.
pub unsafe fn set_stream(
    plan: sys::cufftHandle,
    stream: sys::cudaStream_t,
) -> Result<(), CufftError> {
    sys::cufftSetStream(plan, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cufft/#cufftsetworkarea)
///
/// # Safety
/// Plan must be valid and work_area must be a usable device pointer for the plan.
pub unsafe fn set_work_area(
    plan: sys::cufftHandle,
    work_area: *mut c_void,
) -> Result<(), CufftError> {
    sys::cufftSetWorkArea(plan, work_area).result()
}
