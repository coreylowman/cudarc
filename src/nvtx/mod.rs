//! Safe wrapper for [nvtx](https://nvidia.github.io/NVTX/) apis. Use [`scoped_range()`], [`mark()`], or [`Event`].
//! 
/// Example [`mark()`]/[`Event::mark()`] usage:
/// ```no_run
/// Event::message("Hello world").mark();
/// ```
///
/// Example [`scoped_range()`]/[`Event::range()`] usage:
/// ```no_run
/// let range = Event::message("Hello_world").argb(0xffff0000).range();
/// // ... stuff you want to mark
/// drop(range);
/// ```

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
