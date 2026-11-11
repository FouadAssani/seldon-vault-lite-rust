//! Read-only Safetensors parser with strict validation.
//!
//! The format is: an 8-byte little-endian header length, a UTF-8 JSON header,
//! then a data region. The header maps tensor names to dtype, shape, and
//! byte offsets. Metadata is an optional object of strings.
//!
//! This parser treats the file as hostile. It caps the header size, tensor
//! count, and metadata count; validates every shape and offset with checked
//! arithmetic; rejects overlapping tensors; and never executes metadata.

use std::collections::BTreeMap;

use crate::binary::bf16;
use crate::binary::reader::ByteReader;
use crate::binary::sha256;
use crate::error::{Error, Result};
use crate::json::{self, Limits, Number, Value};
use crate::tensor::{DType, MAX_TENSOR_ELEMENTS, Shape, Tensor};

/// Maximum size of the JSON header in bytes.
pub const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;

/// Maximum number of tensors in one file.
pub const MAX_TENSORS: usize = 100_000;

/// Maximum number of metadata entries in one file.
pub const MAX_METADATA_ENTRIES: usize = 10_000;

/// One tensor entry of a Safetensors file.
#[derive(Debug, Clone)]
pub struct TensorEntry<'a> {
    name: String,
    dtype: DType,
    shape: Shape,
    offsets: (usize, usize),
    bytes: &'a [u8],
}

impl<'a> TensorEntry<'a> {
    /// Tensor name as written in the header.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Storage dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Tensor shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Byte offsets (begin, end) relative to the data region.
    pub fn offsets(&self) -> (usize, usize) {
        self.offsets
    }

    /// Borrowed tensor bytes.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Decodes the tensor into FP32.
    ///
    /// BF16 values are widened exactly. F32 values are read little-endian.
    /// The resulting tensor obeys the crate element cap.
    pub fn to_f32(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        if numel > MAX_TENSOR_ELEMENTS {
            return Err(Error::limit_exceeded(format!(
                "tensor {} has {numel} elements, above the {MAX_TENSOR_ELEMENTS}-element cap",
                self.name
            )));
        }

        match self.dtype {
            DType::F32 => {
                let expected = numel
                    .checked_mul(4)
                    .ok_or_else(|| Error::corrupt("F32 byte length overflow"))?;
                if self.bytes.len() != expected {
                    return Err(Error::corrupt(format!(
                        "tensor {} has {} bytes, expected {expected} for {numel} F32 values",
                        self.name,
                        self.bytes.len()
                    )));
                }
                let mut data = Vec::with_capacity(numel);
                let (chunks, remainder) = self.bytes.as_chunks::<4>();
                if !remainder.is_empty() {
                    return Err(Error::internal(
                        "F32 byte length check did not consume all bytes",
                    ));
                }
                for chunk in chunks {
                    data.push(f32::from_le_bytes(*chunk));
                }
                Tensor::new(self.shape.clone(), data)
            }
            DType::Bf16 => {
                let mut data = vec![0.0f32; numel];
                bf16::decode_bf16(self.bytes, &mut data)?;
                Tensor::new(self.shape.clone(), data)
            }
        }
    }
}

/// Parsed Safetensors file borrowing the original bytes.
#[derive(Debug, Clone)]
pub struct SafetensorsFile<'a> {
    bytes: &'a [u8],
    data: &'a [u8],
    header_len: usize,
    entries: Vec<TensorEntry<'a>>,
    metadata: BTreeMap<String, String>,
}

impl<'a> SafetensorsFile<'a> {
    /// Parses a file with the default JSON limits.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        Self::parse_with_limits(bytes, &Limits::default())
    }

    /// Parses a file with explicit JSON limits.
    ///
    /// # Errors
    ///
    /// Returns Error::Corrupt for malformed headers, invalid offsets, size
    /// mismatches, and overlaps; Error::LimitExceeded for an oversized header,
    /// too many tensors, or too many metadata entries; Error::Unsupported for
    /// a dtype the engine does not implement.
    pub fn parse_with_limits(bytes: &'a [u8], limits: &Limits) -> Result<Self> {
        let mut reader = ByteReader::new(bytes);
        let header_length_u64 = reader.read_u64()?;
        let header_len = usize::try_from(header_length_u64)
            .map_err(|_| Error::limit_exceeded("Safetensors header length does not fit usize"))?;

        if header_len > MAX_HEADER_BYTES {
            return Err(Error::limit_exceeded(format!(
                "Safetensors header is {header_len} bytes, above the {MAX_HEADER_BYTES}-byte cap"
            )));
        }
        if header_len > limits.max_input_bytes {
            return Err(Error::limit_exceeded(format!(
                "Safetensors header is {header_len} bytes, above the {}-byte JSON cap",
                limits.max_input_bytes
            )));
        }

        let header_bytes = reader.read_exact(header_len)?;
        let header_text = std::str::from_utf8(header_bytes)
            .map_err(|_| Error::corrupt("Safetensors header is not valid UTF-8"))?;
        let header = json::parse_with_limits(header_text, limits)?;
        let object = header
            .as_object()
            .ok_or_else(|| Error::corrupt("Safetensors header must be a JSON object"))?;
        let data = reader.remaining_slice();

        let mut entries: Vec<TensorEntry<'a>> = Vec::new();
        let mut metadata: BTreeMap<String, String> = BTreeMap::new();
        let mut ranges: Vec<(usize, usize, String)> = Vec::new();

        for (name, value) in object {
            if name == "__metadata__" {
                let metadata_object = value.as_object().ok_or_else(|| {
                    Error::corrupt("Safetensors __metadata__ must be a JSON object")
                })?;
                for (key, metadata_value) in metadata_object {
                    if metadata.len() >= MAX_METADATA_ENTRIES {
                        return Err(Error::limit_exceeded(format!(
                            "Safetensors metadata exceeds the {MAX_METADATA_ENTRIES}-entry cap"
                        )));
                    }
                    let text = metadata_value.as_str().ok_or_else(|| {
                        Error::corrupt(format!(
                            "Safetensors metadata value for {key} must be a string"
                        ))
                    })?;
                    metadata.insert(key.clone(), text.to_owned());
                }
                continue;
            }

            if entries.len() >= MAX_TENSORS {
                return Err(Error::limit_exceeded(format!(
                    "Safetensors file exceeds the {MAX_TENSORS}-tensor cap"
                )));
            }

            let entry = parse_tensor(name, value, data)?;
            ranges.push((entry.offsets.0, entry.offsets.1, entry.name.clone()));
            entries.push(entry);
        }

        ranges.sort_by_key(|(begin, _, _)| *begin);
        let mut previous_end = 0usize;
        for (begin, end, name) in &ranges {
            // A zero-sized tensor occupies no byte and cannot overlap.
            if begin == end {
                continue;
            }
            if *begin < previous_end {
                return Err(Error::corrupt(format!(
                    "Safetensors tensor {name} overlaps a previous tensor"
                )));
            }
            previous_end = *end;
        }

        Ok(Self {
            bytes,
            data,
            header_len,
            entries,
            metadata,
        })
    }

    /// Original file bytes.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Data region after the header.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Header length in bytes.
    pub fn header_len(&self) -> usize {
        self.header_len
    }

    /// Tensor entries, sorted by name because the header object is ordered.
    pub fn entries(&self) -> &[TensorEntry<'a>] {
        &self.entries
    }

    /// Tensor names in header order.
    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.entries.iter().map(|entry| entry.name.as_str())
    }

    /// Looks up a tensor by name using the sorted entry order.
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry<'a>> {
        self.entries
            .binary_search_by(|entry| entry.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.entries[index])
    }

    /// Metadata key-value pairs.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// SHA-256 fingerprint of the complete file.
    pub fn fingerprint(&self) -> [u8; 32] {
        sha256::sha256(self.bytes)
    }

    /// Lowercase hexadecimal SHA-256 fingerprint of the complete file.
    pub fn fingerprint_hex(&self) -> String {
        sha256::sha256_hex(self.bytes)
    }

    /// Decodes every tensor into FP32, preserving header order.
    pub fn decode_all(&self) -> Result<Vec<(String, Tensor)>> {
        self.entries
            .iter()
            .map(|entry| Ok((entry.name.clone(), entry.to_f32()?)))
            .collect()
    }
}

/// Parses one tensor descriptor and validates its byte range.
fn parse_tensor<'a>(name: &str, value: &Value, data: &'a [u8]) -> Result<TensorEntry<'a>> {
    if name.is_empty() {
        return Err(Error::corrupt("Safetensors tensor name must not be empty"));
    }

    let descriptor = value
        .as_object()
        .ok_or_else(|| Error::corrupt(format!("Safetensors tensor {name} must be an object")))?;

    for key in descriptor.keys() {
        if !matches!(key.as_str(), "dtype" | "shape" | "data_offsets") {
            return Err(Error::corrupt(format!(
                "Safetensors tensor {name} has an unknown field {key}"
            )));
        }
    }

    let dtype_value = descriptor
        .get("dtype")
        .ok_or_else(|| Error::corrupt(format!("Safetensors tensor {name} is missing dtype")))?;
    let dtype_name = dtype_value.as_str().ok_or_else(|| {
        Error::corrupt(format!("Safetensors tensor {name} dtype must be a string"))
    })?;
    let dtype = DType::parse(dtype_name)?;

    let shape_value = descriptor
        .get("shape")
        .ok_or_else(|| Error::corrupt(format!("Safetensors tensor {name} is missing shape")))?;
    let shape_array = shape_value.as_array().ok_or_else(|| {
        Error::corrupt(format!("Safetensors tensor {name} shape must be an array"))
    })?;
    let mut dims = Vec::with_capacity(shape_array.len());
    for dim in shape_array {
        let value = dim.as_number().and_then(Number::as_u64).ok_or_else(|| {
            Error::corrupt(format!(
                "Safetensors tensor {name} shape entries must be non-negative integers"
            ))
        })?;
        let dim = usize::try_from(value)
            .map_err(|_| Error::limit_exceeded("Safetensors shape entry does not fit usize"))?;
        dims.push(dim);
    }
    let shape = Shape::new(dims)?;

    let offsets_value = descriptor.get("data_offsets").ok_or_else(|| {
        Error::corrupt(format!("Safetensors tensor {name} is missing data_offsets"))
    })?;
    let offsets_array = offsets_value.as_array().ok_or_else(|| {
        Error::corrupt(format!(
            "Safetensors tensor {name} data_offsets must be an array"
        ))
    })?;
    if offsets_array.len() != 2 {
        return Err(Error::corrupt(format!(
            "Safetensors tensor {name} data_offsets must have two entries"
        )));
    }

    let begin = offset_value(name, &offsets_array[0], "begin")?;
    let end = offset_value(name, &offsets_array[1], "end")?;

    if begin > end {
        return Err(Error::corrupt(format!(
            "Safetensors tensor {name} has begin {begin} greater than end {end}"
        )));
    }
    if end > data.len() {
        return Err(Error::corrupt(format!(
            "Safetensors tensor {name} ends at {end}, beyond the {}-byte data region",
            data.len()
        )));
    }

    let element_size = dtype.size();
    if begin % element_size != 0 || end % element_size != 0 {
        return Err(Error::corrupt(format!(
            "Safetensors tensor {name} offsets are not aligned to {element_size}-byte elements"
        )));
    }

    let expected_bytes = shape
        .numel()
        .checked_mul(element_size)
        .ok_or_else(|| Error::corrupt("Safetensors tensor byte length overflow"))?;
    if end - begin != expected_bytes {
        return Err(Error::corrupt(format!(
            "Safetensors tensor {name} spans {} bytes, expected {expected_bytes} for shape {shape}",
            end - begin
        )));
    }

    Ok(TensorEntry {
        name: name.to_owned(),
        dtype,
        shape,
        offsets: (begin, end),
        bytes: &data[begin..end],
    })
}

/// Parses one data offset entry.
fn offset_value(name: &str, value: &Value, label: &str) -> Result<usize> {
    let offset = value.as_number().and_then(Number::as_u64).ok_or_else(|| {
        Error::corrupt(format!(
            "Safetensors tensor {name} {label} offset must be a non-negative integer"
        ))
    })?;
    usize::try_from(offset)
        .map_err(|_| Error::limit_exceeded("Safetensors offset does not fit usize"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(header: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn parses_f32_and_bf16_tensors() {
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]},"__metadata__":{"k":"v"}}"#;
        let mut data = Vec::new();
        data.extend_from_slice(&1.5f32.to_le_bytes());
        data.extend_from_slice(&(-2.0f32).to_le_bytes());
        data.extend_from_slice(&0x3f00u16.to_le_bytes());
        data.extend_from_slice(&0xbf80u16.to_le_bytes());

        let bytes = build(header, &data);
        let file = SafetensorsFile::parse(&bytes).expect("file must parse");
        assert_eq!(file.header_len(), header.len());
        assert_eq!(file.data(), data.as_slice());
        assert_eq!(file.entries().len(), 2);
        assert_eq!(file.names().collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!(file.metadata().get("k").map(String::as_str), Some("v"));

        let a = file.tensor("a").expect("tensor a");
        assert_eq!(a.dtype(), DType::F32);
        assert_eq!(a.shape().dims(), &[2]);
        assert_eq!(a.offsets(), (0, 8));
        let decoded = a.to_f32().expect("decode a");
        assert_eq!(decoded.data(), &[1.5, -2.0]);

        let b = file.tensor("b").expect("tensor b");
        assert_eq!(b.dtype(), DType::Bf16);
        let decoded = b.to_f32().expect("decode b");
        assert_eq!(decoded.data(), &[0.5, -1.0]);
    }

    #[test]
    fn accepts_zero_sized_tensors() {
        let header = r#"{"empty":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        let bytes = build(header, &[]);
        let file = SafetensorsFile::parse(&bytes).expect("file must parse");
        let tensor = file.tensor("empty").expect("empty tensor");
        assert_eq!(tensor.shape().numel(), 0);
        assert_eq!(tensor.to_f32().expect("decode").data(), &[] as &[f32]);
    }

    #[test]
    fn fingerprint_is_the_file_sha256() {
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let bytes = build(header, &1.0f32.to_le_bytes());
        let file = SafetensorsFile::parse(&bytes).expect("parse");
        assert_eq!(file.fingerprint(), sha256::sha256(&bytes));
        assert_eq!(file.fingerprint_hex(), sha256::sha256_hex(&bytes));
    }

    #[test]
    fn rejects_short_files() {
        assert!(matches!(
            SafetensorsFile::parse(&[]).expect_err("empty file must fail"),
            Error::Corrupt(_)
        ));
        assert!(matches!(
            SafetensorsFile::parse(&[0, 0, 0]).expect_err("short file must fail"),
            Error::Corrupt(_)
        ));
    }

    #[test]
    fn rejects_header_length_above_the_cap() {
        let bytes = u64::MAX.to_le_bytes();
        let error = SafetensorsFile::parse(&bytes).expect_err("huge header must fail");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn rejects_truncated_header() {
        let mut bytes = 100u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let error = SafetensorsFile::parse(&bytes).expect_err("truncated header must fail");
        assert!(matches!(error, Error::Corrupt(_)));
        assert!(error.to_string().contains("truncated"));
    }

    #[test]
    fn rejects_non_utf8_and_non_object_headers() {
        let bytes = build_bytes(&[0xff], &[]);
        assert!(matches!(
            SafetensorsFile::parse(&bytes).expect_err("invalid UTF-8 must fail"),
            Error::Corrupt(_)
        ));

        let bytes = build("[]", &[]);
        assert!(matches!(
            SafetensorsFile::parse(&bytes).expect_err("array header must fail"),
            Error::Corrupt(_)
        ));
    }

    #[test]
    fn rejects_missing_and_unknown_tensor_fields() {
        for header in [
            r#"{"a":{"shape":[1],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1}}"#,
        ] {
            let error = SafetensorsFile::parse(&build(header, &[0u8; 4]))
                .expect_err("invalid descriptor must fail");
            assert!(matches!(error, Error::Corrupt(_)), "header {header}");
        }
    }

    #[test]
    fn rejects_unsupported_dtypes() {
        let header = r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        let error =
            SafetensorsFile::parse(&build(header, &[0u8; 2])).expect_err("F16 must be rejected");
        assert!(matches!(error, Error::Unsupported(_)));
    }

    #[test]
    fn rejects_invalid_shapes_and_offsets() {
        let cases = [
            r#"{"a":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[-1,3]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}}"#,
        ];
        for header in cases {
            let error = SafetensorsFile::parse(&build(header, &[0u8; 8]))
                .expect_err("invalid descriptor must fail");
            assert!(matches!(error, Error::Corrupt(_)), "header {header}");
        }
    }

    #[test]
    fn rejects_overlapping_tensors() {
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        let error =
            SafetensorsFile::parse(&build(header, &[0u8; 12])).expect_err("overlap must fail");
        assert!(matches!(error, Error::Corrupt(_)));
        assert!(error.to_string().contains("overlaps"));
    }

    #[test]
    fn rejects_non_string_metadata() {
        let header =
            r#"{"__metadata__":{"k":1},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let error = SafetensorsFile::parse(&build(header, &[0u8; 4]))
            .expect_err("metadata must be strings");
        assert!(matches!(error, Error::Corrupt(_)));
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let error = SafetensorsFile::parse(&build(header, &[0u8; 8]))
            .expect_err("duplicate names must fail");
        assert!(matches!(error, Error::Corrupt(_)));
    }

    #[test]
    fn rejects_empty_tensor_names() {
        let header = r#"{"":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let error =
            SafetensorsFile::parse(&build(header, &[0u8; 4])).expect_err("empty name must fail");
        assert!(matches!(error, Error::Corrupt(_)));
    }

    #[test]
    fn parses_the_versioned_fixture() {
        const FIXTURE: &[u8] = include_bytes!("../../fixtures/formats/mini.safetensors");
        let file = SafetensorsFile::parse(FIXTURE).expect("fixture must parse");
        assert_eq!(file.entries().len(), 3);
        assert_eq!(
            file.names().collect::<Vec<_>>(),
            vec!["alpha", "beta", "empty"]
        );
        assert_eq!(
            file.metadata().get("generator").map(String::as_str),
            Some("seldon-vault-lite M1 fixture")
        );
        assert_eq!(
            file.tensor("alpha")
                .expect("alpha")
                .to_f32()
                .expect("decode alpha")
                .data(),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            file.tensor("beta")
                .expect("beta")
                .to_f32()
                .expect("decode beta")
                .data(),
            &[0.5, -1.0, 2.0]
        );
        assert_eq!(
            file.tensor("empty")
                .expect("empty")
                .to_f32()
                .expect("decode empty")
                .numel(),
            0
        );
        assert_eq!(
            file.fingerprint_hex(),
            "caca3b030faeb4ded52275d54c2fca0f2ee0d6b1aeeb8ab1819e0785b1f89ca7"
        );
    }

    fn build_bytes(header: &[u8], data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(data);
        bytes
    }
}
