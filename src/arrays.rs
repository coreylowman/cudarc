pub trait NumElements {
    const NUMEL: usize;
    type Dtype;
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

impl<T: NumElements, const M: usize> NumElements for [T; M] {
    const NUMEL: usize = T::NUMEL * M;
    type Dtype = T::Dtype;
}
