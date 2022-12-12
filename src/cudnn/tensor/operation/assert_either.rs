use const_panic::concat_assert;

/// If an error points here, check for errors in other files of this dir (more info there)
pub struct AssertTrue<const A: bool> {}
pub trait ConstTrue {}
impl ConstTrue for AssertTrue<true> {}

#[track_caller]
pub const fn is_either(a: usize, b: usize, c: usize) -> bool {
    concat_assert! {
        a == b || a == c,
        "Dimension size `", a, "` must either be `", b, "` or `", c, "`"
    };
    a == b || a == c
}
