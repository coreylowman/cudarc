#[cfg(feature = "dynamic-loading")]
mod loaded;
#[cfg(feature = "dynamic-loading")]
pub use loaded::*;

#[cfg(not(feature = "dynamic-loading"))]
mod linked;
#[cfg(not(feature = "dynamic-loading"))]
pub use linked::*;