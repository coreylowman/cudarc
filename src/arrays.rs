use core::{
    mem::{size_of, MaybeUninit},
    ops::{Deref, DerefMut},
};

use crate::cudarc::ValidAsZeroBits;

pub trait NumElements {
    const NUMEL: usize;
    type Dtype;
}

// TODO change names?
/// An [Array] with `A` as the "real" length but always an even "actual" length
pub type FixedSizeArray<N, const A: usize> = Array<N, A, {(A + 1) / 2 * 2}>;

// basically derive everything?
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
// using 2 const generics `A` and `B` to prevent
// `where [(); (A + 1) / 2 * 2]:` everywhere
/// A fixed sized array of type `N`, "real" length `A`, but "actual" length `B`.
/// `B` must be even.
/// [Array] implements [NumElements] with [NumElements::NUMEL] = `B`,
/// so cuda rng won't fail on odd `A`.
pub struct Array<N, const A: usize, const B: usize> {
    array: [N; B],
}
unsafe impl<N: ValidAsZeroBits, const A: usize, const B: usize> ValidAsZeroBits for Array<N, A, B> {}
impl<N, const A: usize, const B: usize> Deref for Array<N, A, B> {
    type Target = [N; A];

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.array.as_ptr() as *const Self::Target) }
    }
}
impl<N, const A: usize, const B: usize> DerefMut for Array<N, A, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self.array.as_ptr() as *mut Self::Target) }
    }
}

struct Assert<const A: usize, const B: usize>;
trait ConstEq {}

impl<const A: usize> ConstEq for Assert<A, A> {}
// can't use `FixedSizeArray<N, A>` instead of `Array<N, A, A>` (impl'ed 2 times)
impl<N, const A: usize> From<[N; A]> for Array<N, A, A>
where
    Assert<{ A % 2 }, 0>: ConstEq,
{
    /// Use `#![feature(generic_const_exprs)]` to prevent weird errors.
    fn from(array: [N; A]) -> Self {
        Self { array }
    }
}
impl<N: ValidAsZeroBits, const A: usize> From<[N; A]> for Array<N, A, {A + 1}>
where
    Assert<{ A % 2 }, 1>: ConstEq,
{
    /// Use `#![feature(generic_const_exprs)]` to prevent weird errors.
    fn from(array: [N; A]) -> Self {
        Self {
            array: unsafe {
                let mut res = MaybeUninit::zeroed();
                std::ptr::copy_nonoverlapping(
                    array.as_ptr(),
                    res.as_mut_ptr() as *mut _,
                    size_of::<[N; A]>(),
                );
                res.assume_init()
            },
        }
    }
}

macro_rules! impl_numel_for_builtin {
    ($T:ty) => {
        impl NumElements for $T {
            const NUMEL: usize = 1;
            type Dtype = Self;
        }
    };
}

impl_numel_for_builtin!(u8);
impl_numel_for_builtin!(u16);
impl_numel_for_builtin!(u32);
impl_numel_for_builtin!(u64);
impl_numel_for_builtin!(usize);
impl_numel_for_builtin!(i8);
impl_numel_for_builtin!(i16);
impl_numel_for_builtin!(i32);
impl_numel_for_builtin!(i64);
impl_numel_for_builtin!(isize);
impl_numel_for_builtin!(f32);
impl_numel_for_builtin!(f64);

impl<T: NumElements, const M: usize, const N: usize> NumElements for Array<T, M, N> {
    const NUMEL: usize = T::NUMEL * N;
    type Dtype = T::Dtype;
}
