pub struct AssertBool<const B: bool>;
pub trait ConstTrue {}
impl ConstTrue for AssertBool<true> {}
pub struct AssertEither<const A: usize, const B: usize, const C: usize>;
pub trait IsEither {}
const fn either<const A: usize, const B: usize, const C: usize>() -> bool {
    A == B || A == C
}
impl<const A: usize, const B: usize, const C: usize> IsEither for AssertEither<A, B, C> where
    AssertBool<{ either::<A, B, C>() }>: ConstTrue
{
}
