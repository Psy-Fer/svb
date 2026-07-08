//! Quantize-trailing-shift (qts) as a composable layer over `i16` samples.
//!
//! ADC samples are often multiples of a power of two (the low bits carry no
//! information). qts finds the largest right-shift `q` that loses none of
//! that information — every sample's low `q` bits are zero — and applies it
//! before further compression. The shift is lossless: [`unshift_inplace`]
//! restores the original values exactly.
//!
//! This is a fixed `i16`-only transform (unlike [`crate::delta`] /
//! [`crate::zigzag`], which are generic over several integer types) — it
//! exists specifically to feed the ex-zd pipeline, and there is only one
//! type it is ever applied to.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Find the largest `q` in `0..=max` such that every sample's low `q` bits are zero.
///
/// Returns `0` if no such shift exists (e.g. `samples` is empty, or any
/// sample has its lowest bit set).
///
/// # Examples
///
/// ```
/// # use svb::quantize::find_qts;
/// assert_eq!(find_qts(&[8i16, 16, -24], 5), 3);
/// assert_eq!(find_qts(&[8i16, 17, -24], 5), 0);
/// ```
pub fn find_qts(samples: &[i16], max: u8) -> u8 {
    let mut q = max;
    for &s in samples {
        while q > 0 && (s & ((1i16 << q) - 1)) != 0 {
            q -= 1;
        }
        if q == 0 {
            break;
        }
    }
    q
}

/// Right-shift every sample by `q` (arithmetic shift), returning a new `Vec`.
///
/// # Examples
///
/// ```
/// # use svb::quantize::apply_shift;
/// assert_eq!(apply_shift(&[8i16, -24], 3), [1, -3]);
/// ```
pub fn apply_shift(samples: &[i16], q: u8) -> Vec<i16> {
    samples.iter().map(|&s| s >> q).collect()
}

/// Left-shift every sample by `q` in place, undoing [`apply_shift`].
///
/// `q` is masked to `0..=15` so the shift can never panic, even on a `q`
/// value read from untrusted/corrupted input.
///
/// # Examples
///
/// ```
/// # use svb::quantize::unshift_inplace;
/// let mut samples = [1i16, -3];
/// unshift_inplace(&mut samples, 3);
/// assert_eq!(samples, [8, -24]);
/// ```
pub fn unshift_inplace(samples: &mut [i16], q: u8) {
    let q = q & 15;
    for s in samples.iter_mut() {
        *s <<= q;
    }
}
