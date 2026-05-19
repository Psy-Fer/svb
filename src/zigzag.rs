#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

pub fn encode(samples: &[i16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(samples.len());
    encode_into(samples, &mut out);
    out
}

pub fn encode_into(samples: &[i16], out: &mut Vec<u16>) {
    out.extend(samples.iter().copied().map(encode_one));
}

pub fn decode(codes: &[u16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(codes.len());
    decode_into(codes, &mut out);
    out
}

pub fn decode_into(codes: &[u16], out: &mut Vec<i16>) {
    out.extend(codes.iter().copied().map(decode_one));
}

#[inline]
fn encode_one(x: i16) -> u16 {
    // Cast to u16 before left-shift to avoid i16 overflow on i16::MIN.
    // Right shift on i16 is arithmetic (sign-extending), yielding 0x0000 or 0xFFFF.
    ((x as u16) << 1) ^ ((x >> 15) as u16)
}

#[inline]
fn decode_one(n: u16) -> i16 {
    ((n >> 1) as i16) ^ (-((n & 1) as i16))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn known_values() {
        let cases: &[(i16, u16)] = &[
            (0, 0),
            (-1, 1),
            (1, 2),
            (-2, 3),
            (i16::MAX, 65534),
            (i16::MIN, 65535),
        ];
        for &(input, expected) in cases {
            assert_eq!(encode_one(input), expected, "encode {input}");
            assert_eq!(decode_one(expected), input, "decode {expected}");
        }
    }

    #[test]
    fn exhaustive_roundtrip() {
        // All 65536 i16 values must round-trip through zigzag.
        for raw in u16::MIN..=u16::MAX {
            let x = raw as i16;
            assert_eq!(decode_one(encode_one(x)), x);
        }
    }

    #[test]
    fn slice_roundtrip() {
        let samples: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        assert_eq!(decode(&encode(&samples)), samples);
    }

    #[test]
    fn encode_into_appends() {
        let mut out = vec![99u16];
        encode_into(&[0i16, -1, 1], &mut out);
        assert_eq!(out, [99, 0, 1, 2]);
    }

    #[test]
    fn small_values_encode_small() {
        // The point of zigzag: small absolute values → small unsigned codes.
        for x in -127i16..=127 {
            assert!(encode_one(x) <= 254, "x={x} encoded to {} (>254)", encode_one(x));
        }
    }
}
