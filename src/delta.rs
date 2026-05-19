#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

pub fn encode(samples: &[i16]) -> Vec<i16> {
    encode_with_initial(0, samples)
}

pub fn encode_with_initial(initial: i16, samples: &[i16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(samples.len());
    encode_with_initial_into(initial, samples, &mut out);
    out
}

pub fn encode_into(samples: &[i16], out: &mut Vec<i16>) {
    encode_with_initial_into(0, samples, out);
}

pub fn decode(deltas: &[i16]) -> Vec<i16> {
    decode_with_initial(0, deltas)
}

pub fn decode_with_initial(initial: i16, deltas: &[i16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(deltas.len());
    decode_with_initial_into(initial, deltas, &mut out);
    out
}

pub fn decode_into(deltas: &[i16], out: &mut Vec<i16>) {
    decode_with_initial_into(0, deltas, out);
}

fn encode_with_initial_into(initial: i16, samples: &[i16], out: &mut Vec<i16>) {
    let mut prev = initial;
    for &s in samples {
        out.push(s.wrapping_sub(prev));
        prev = s;
    }
}

fn decode_with_initial_into(initial: i16, deltas: &[i16], out: &mut Vec<i16>) {
    let mut acc = initial;
    for &d in deltas {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode(&encode(&[])), &[] as &[i16]);
    }

    #[test]
    fn roundtrip_single() {
        for v in [0i16, 1, -1, i16::MIN, i16::MAX] {
            assert_eq!(decode(&encode(&[v])), &[v]);
        }
    }

    #[test]
    fn roundtrip_sequence() {
        let samples: Vec<i16> = (-128..=127).collect();
        assert_eq!(decode(&encode(&samples)), samples);
    }

    #[test]
    fn encode_produces_differences() {
        let samples = [10i16, 20, 15, 30];
        let deltas = encode(&samples);
        assert_eq!(deltas, [10, 10, -5, 15]);
    }

    #[test]
    fn encode_wraps_on_overflow() {
        // i16::MIN - i16::MAX wraps
        let samples = [i16::MAX, i16::MIN];
        let deltas = encode(&samples);
        assert_eq!(deltas[0], i16::MAX);
        assert_eq!(deltas[1], i16::MIN.wrapping_sub(i16::MAX));
        assert_eq!(decode(&deltas), samples);
    }

    #[test]
    fn encode_with_initial_nonzero() {
        let samples = [10i16, 20, 30];
        let deltas = encode_with_initial(5, &samples);
        assert_eq!(deltas, [5, 10, 10]);
        assert_eq!(decode_with_initial(5, &deltas), samples);
    }

    #[test]
    fn encode_into_appends() {
        let mut out = vec![99i16];
        encode_into(&[3i16, 6, 9], &mut out);
        assert_eq!(out, [99, 3, 3, 3]);
    }
}
