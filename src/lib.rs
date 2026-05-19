#![cfg_attr(not(feature = "std"), no_std)]
#![deny(clippy::all)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

pub mod error;
pub use error::DecodeError;

#[cfg(feature = "alloc")]
pub(crate) mod coder;

#[cfg(feature = "alloc")]
pub mod delta;
#[cfg(feature = "alloc")]
pub mod zigzag;

#[cfg(feature = "alloc")]
pub mod u16;
#[cfg(feature = "alloc")]
pub mod u32;
#[cfg(feature = "alloc")]
pub mod u64;
