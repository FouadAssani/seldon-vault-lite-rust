//! Strict base64 decoding for media carried inside JSON requests.
//!
//! The HTTP API cannot depend on a crate for base64. The decoder accepts the
//! RFC 4648 standard alphabet with mandatory padding, rejects whitespace and
//! non-canonical trailing bits, and enforces an explicit decoded-size cap. A
//! hostile body therefore cannot make the server allocate an unbounded buffer
//! before the media decoder applies its own limits.

use crate::error::{Error, Result};

/// Maximum decoded media size accepted by one request.
///
/// This is deliberately smaller than the HTTP body cap: a base64 expansion is
/// four thirds of the decoded size, and the media decoders impose their own
/// pixel, sample, and memory caps below this value.
pub(crate) const MAX_MEDIA_BYTES: usize = 16 * 1024 * 1024;

/// Decodes one base64 string, rejecting anything that is not canonical.
///
/// # Errors
///
/// Returns a corruption error for an invalid alphabet, a wrong length, missing
/// or misplaced padding, non-zero trailing bits, or invalid UTF-8. Returns a
/// limit error when the decoded size would exceed `max_bytes`.
pub(crate) fn decode(input: &str, max_bytes: usize) -> Result<Vec<u8>> {
    let bytes = input.as_bytes();
    if bytes.is_empty() {
        return Err(Error::corrupt("base64 data is empty"));
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::corrupt(
            "base64 data length must be a multiple of four",
        ));
    }

    let padding = if bytes.ends_with(b"==") {
        2
    } else if bytes.ends_with(b"=") {
        1
    } else {
        0
    };
    let output_len = bytes
        .len()
        .checked_div(4)
        .and_then(|groups| groups.checked_mul(3))
        .and_then(|bytes| bytes.checked_sub(padding))
        .ok_or_else(|| Error::limit_exceeded("base64 decoded length overflow"))?;
    if output_len > max_bytes {
        return Err(Error::limit_exceeded(format!(
            "decoded media is {output_len} bytes, above the {max_bytes}-byte cap"
        )));
    }

    let mut output = Vec::with_capacity(output_len);
    let mut index = 0;
    while index < bytes.len() {
        let remaining = bytes.len() - index;
        let chunk_len = remaining.min(4);
        let mut values = [0u8; 4];
        for offset in 0..chunk_len {
            values[offset] = decode_char(bytes[index + offset])?;
        }
        if chunk_len < 4 {
            return Err(Error::corrupt("base64 data is truncated"));
        }
        if values[0] == 0xFF || values[1] == 0xFF {
            return Err(Error::corrupt("base64 padding must appear only at the end"));
        }
        let is_last = index + 4 == bytes.len();
        if !is_last && (values[2] == 0xFF || values[3] == 0xFF) {
            return Err(Error::corrupt("base64 padding appears before the end"));
        }
        if values[2] == 0xFF && values[3] != 0xFF {
            return Err(Error::corrupt("base64 has misplaced padding"));
        }
        if is_last {
            if values[2] == 0xFF && values[3] == 0xFF {
                if values[1] & 0x0F != 0 {
                    return Err(Error::corrupt("base64 has non-zero trailing bits"));
                }
            } else if values[3] == 0xFF && values[2] & 0x03 != 0 {
                return Err(Error::corrupt("base64 has non-zero trailing bits"));
            }
        }
        let first = (values[0] << 2) | (values[1] >> 4);
        output.push(first);
        if values[2] != 0xFF {
            let second = (values[1] << 4) | (values[2] >> 2);
            output.push(second);
        }
        if values[3] != 0xFF {
            let third = (values[2] << 6) | values[3];
            output.push(third);
        }
        index += 4;
    }

    if output.len() != output_len {
        return Err(Error::internal(
            "base64 decoder produced an unexpected length",
        ));
    }
    Ok(output)
}

fn decode_char(byte: u8) -> Result<u8> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        b'=' => Ok(0xFF),
        _ => Err(Error::corrupt(format!(
            "base64 contains an invalid byte 0x{byte:02X}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_known_vectors() {
        assert!(
            decode("", 16)
                .expect_err("empty")
                .to_string()
                .contains("empty")
        );
        assert_eq!(decode("Zg==", 16).expect("f"), b"f");
        assert_eq!(decode("Zm8=", 16).expect("fo"), b"fo");
        assert_eq!(decode("Zm9v", 16).expect("foo"), b"foo");
        assert_eq!(decode("Zm9vYg==", 16).expect("foob"), b"foob");
        assert_eq!(decode("Zm9vYmE=", 16).expect("fooba"), b"fooba");
        assert_eq!(decode("Zm9vYmFy", 16).expect("foobar"), b"foobar");
    }

    #[test]
    fn rejects_invalid_input() {
        for value in [
            "Zg", "Zg=", "Z===", "Zg===", "Zg==Zg==", "Zg$=", "Zg= ", " Zg==", "=AAA",
        ] {
            assert!(decode(value, 16).is_err(), "accepted {value:?}");
        }
    }

    #[test]
    fn enforces_the_decoded_cap() {
        let error = decode("Zm9vYmFy", 2).expect_err("cap");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }
}
