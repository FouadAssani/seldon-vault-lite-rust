//! Strict JSON parser with explicit limits.
//!
//! The crate cannot depend on serde_json. This parser implements the subset of
//! RFC 8259 needed by configuration files, Safetensors headers, and, later,
//! the HTTP API: objects, arrays, numbers, strings, escapes, and Unicode
//! scalar values. It is strict: duplicate keys, trailing commas, leading
//! zeros, unescaped control characters, lone surrogates, NaN, infinity, and
//! trailing content are rejected.
//!
//! Every allocation is bounded by Limits: input size, nesting depth, string
//! length, container entries, and total values. The parser never panics on
//! hostile input.

use std::collections::BTreeMap;

use crate::error::{Error, Result};

/// JSON object with deterministic key ordering.
///
/// A BTreeMap is used instead of a hash map so that iteration order is
/// reproducible and does not depend on a random seed.
pub type Object = BTreeMap<String, Value>;

/// A JSON number.
///
/// Integers that fit in u64 or i64 are preserved exactly. Integers outside
/// those ranges, and numbers written with a fraction or an exponent, are
/// represented as f64. Non-finite results are rejected at parse time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    /// Non-negative integer that fits in u64.
    Unsigned(u64),
    /// Negative integer that fits in i64.
    Signed(i64),
    /// Floating-point number.
    Float(f64),
}

impl Number {
    /// Returns the number as f64.
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Unsigned(value) => value as f64,
            Self::Signed(value) => value as f64,
            Self::Float(value) => value,
        }
    }

    /// Returns the number as u64 when it is an exact non-negative integer.
    pub fn as_u64(self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(value),
            Self::Signed(value) => u64::try_from(value).ok(),
            Self::Float(_) => None,
        }
    }

    /// Returns the number as i64 when it is an exact integer in range.
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Self::Unsigned(value) => i64::try_from(value).ok(),
            Self::Signed(value) => Some(value),
            Self::Float(_) => None,
        }
    }

    /// Returns true when the number was parsed as an integer token.
    pub fn is_integer(self) -> bool {
        !matches!(self, Self::Float(_))
    }
}

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// JSON null.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// JSON number.
    Number(Number),
    /// JSON string.
    String(String),
    /// JSON array.
    Array(Vec<Value>),
    /// JSON object.
    Object(Object),
}

impl Value {
    /// Returns the boolean value when this is a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the number when this is a number.
    pub fn as_number(&self) -> Option<Number> {
        match self {
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the string when this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the array when this is an array.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the object when this is an object.
    pub fn as_object(&self) -> Option<&Object> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    /// Looks up a key when this is an object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?.get(key)
    }

    /// Human-readable type name used in error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "boolean",
            Self::Number(_) => "number",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }
}

/// Parser limits.
///
/// The defaults are chosen for configuration and Safetensors headers. Media
/// and HTTP bodies will use their own, documented limits.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Maximum size of the input text in bytes.
    pub max_input_bytes: usize,
    /// Maximum nesting depth of arrays and objects.
    pub max_depth: usize,
    /// Maximum size of a single string in bytes after unescaping.
    pub max_string_bytes: usize,
    /// Maximum number of entries in a single array or object.
    pub max_container_entries: usize,
    /// Maximum total number of values in the document.
    pub max_total_values: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_input_bytes: 16 * 1024 * 1024,
            max_depth: 128,
            max_string_bytes: 1024 * 1024,
            max_container_entries: 100_000,
            max_total_values: 1_000_000,
        }
    }
}

/// Parses a JSON document with the default limits.
pub fn parse(input: &str) -> Result<Value> {
    parse_with_limits(input, &Limits::default())
}

/// Parses a JSON document from bytes, validating UTF-8 first.
pub fn parse_bytes(input: &[u8]) -> Result<Value> {
    let text =
        std::str::from_utf8(input).map_err(|_| Error::corrupt("JSON input is not valid UTF-8"))?;
    parse(text)
}

/// Parses a JSON document with explicit limits.
pub fn parse_with_limits(input: &str, limits: &Limits) -> Result<Value> {
    if input.len() > limits.max_input_bytes {
        return Err(Error::limit_exceeded(format!(
            "JSON input is {} bytes, above the {}-byte cap",
            input.len(),
            limits.max_input_bytes
        )));
    }

    let mut parser = Parser {
        input: input.as_bytes(),
        position: 0,
        limits,
        values: 0,
    };

    parser.skip_whitespace();
    let value = parser.parse_value(0)?;
    parser.skip_whitespace();
    if parser.position != parser.input.len() {
        return Err(parser.error("trailing content after the top-level value"));
    }
    Ok(value)
}

struct Parser<'a> {
    input: &'a [u8],
    position: usize,
    limits: &'a Limits,
    values: usize,
}

impl<'a> Parser<'a> {
    fn error(&self, message: impl Into<String>) -> Error {
        Error::corrupt(format!(
            "JSON at byte {}: {}",
            self.position,
            message.into()
        ))
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }

    fn next(&mut self) -> Result<u8> {
        let byte = self
            .peek()
            .ok_or_else(|| self.error("unexpected end of input"))?;
        self.position += 1;
        Ok(byte)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.position += 1;
        }
    }

    fn expect_literal(&mut self, literal: &[u8]) -> Result<()> {
        let end = self
            .position
            .checked_add(literal.len())
            .ok_or_else(|| self.error("literal range overflow"))?;
        if self.input.get(self.position..end) != Some(literal) {
            return Err(self.error("invalid literal"));
        }
        self.position = end;
        Ok(())
    }

    fn count_value(&mut self) -> Result<()> {
        self.values = self.values.saturating_add(1);
        if self.values > self.limits.max_total_values {
            return Err(Error::limit_exceeded(format!(
                "JSON document exceeds the {}-value cap",
                self.limits.max_total_values
            )));
        }
        Ok(())
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value> {
        if depth > self.limits.max_depth {
            return Err(Error::limit_exceeded(format!(
                "JSON nesting exceeds the depth cap of {}",
                self.limits.max_depth
            )));
        }
        self.count_value()?;
        self.skip_whitespace();
        let byte = self
            .peek()
            .ok_or_else(|| self.error("unexpected end of input"))?;

        match byte {
            b'n' => {
                self.expect_literal(b"null")?;
                Ok(Value::Null)
            }
            b't' => {
                self.expect_literal(b"true")?;
                Ok(Value::Bool(true))
            }
            b'f' => {
                self.expect_literal(b"false")?;
                Ok(Value::Bool(false))
            }
            b'"' => Ok(Value::String(self.parse_string()?)),
            b'[' => self.parse_array(depth),
            b'{' => self.parse_object(depth),
            b'-' | b'0'..=b'9' => Ok(Value::Number(self.parse_number()?)),
            other => Err(self.error(format!("unexpected byte {other:#04x}"))),
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<Value> {
        self.position += 1;
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.position += 1;
            return Ok(Value::Array(Vec::new()));
        }

        let mut values = Vec::new();
        loop {
            if values.len() >= self.limits.max_container_entries {
                return Err(Error::limit_exceeded(format!(
                    "JSON array exceeds the {}-entry cap",
                    self.limits.max_container_entries
                )));
            }
            values.push(self.parse_value(depth + 1)?);
            self.skip_whitespace();
            match self.next()? {
                b',' => {
                    self.skip_whitespace();
                    if self.peek() == Some(b']') {
                        return Err(self.error("trailing comma in array"));
                    }
                }
                b']' => break,
                other => {
                    return Err(self.error(format!(
                        "expected ',' or ']' after an array element, got {other:#04x}"
                    )));
                }
            }
        }
        Ok(Value::Array(values))
    }

    fn parse_object(&mut self, depth: usize) -> Result<Value> {
        self.position += 1;
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.position += 1;
            return Ok(Value::Object(Object::new()));
        }

        let mut object = Object::new();
        loop {
            if object.len() >= self.limits.max_container_entries {
                return Err(Error::limit_exceeded(format!(
                    "JSON object exceeds the {}-entry cap",
                    self.limits.max_container_entries
                )));
            }
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.error("object key must be a string"));
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.next()? != b':' {
                return Err(self.error("expected ':' after an object key"));
            }
            let value = self.parse_value(depth + 1)?;
            if object.insert(key, value).is_some() {
                return Err(self.error("duplicate object key"));
            }
            self.skip_whitespace();
            match self.next()? {
                b',' => {
                    self.skip_whitespace();
                    if self.peek() == Some(b'}') {
                        return Err(self.error("trailing comma in object"));
                    }
                }
                b'}' => break,
                other => {
                    return Err(self.error(format!(
                        "expected ',' or '}}' after an object member, got {other:#04x}"
                    )));
                }
            }
        }
        Ok(Value::Object(object))
    }

    fn parse_string(&mut self) -> Result<String> {
        self.position += 1;
        let mut output = Vec::new();
        loop {
            let byte = self.next()?;
            match byte {
                b'"' => break,
                b'\\' => self.parse_escape(&mut output)?,
                0x00..=0x1f => {
                    return Err(self.error("unescaped control character in string"));
                }
                _ => output.push(byte),
            }
            if output.len() > self.limits.max_string_bytes {
                return Err(Error::limit_exceeded(format!(
                    "JSON string exceeds the {}-byte cap",
                    self.limits.max_string_bytes
                )));
            }
        }
        String::from_utf8(output).map_err(|_| self.error("string is not valid UTF-8"))
    }

    fn parse_escape(&mut self, output: &mut Vec<u8>) -> Result<()> {
        let byte = self.next()?;
        match byte {
            b'"' => output.push(b'"'),
            b'\\' => output.push(b'\\'),
            b'/' => output.push(b'/'),
            b'b' => output.push(0x08),
            b'f' => output.push(0x0c),
            b'n' => output.push(b'\n'),
            b'r' => output.push(b'\r'),
            b't' => output.push(b'\t'),
            b'u' => {
                let first = self.parse_hex4()?;
                let scalar = if (0xd800..=0xdbff).contains(&first) {
                    if self.next()? != b'\\' {
                        return Err(self.error("high surrogate is not followed by an escape"));
                    }
                    if self.next()? != b'u' {
                        return Err(self.error("high surrogate is not followed by a \\u escape"));
                    }
                    let second = self.parse_hex4()?;
                    if !(0xdc00..=0xdfff).contains(&second) {
                        return Err(self.error("high surrogate is not followed by a low surrogate"));
                    }
                    0x1_0000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00)
                } else if (0xdc00..=0xdfff).contains(&first) {
                    return Err(self.error("lone low surrogate"));
                } else {
                    u32::from(first)
                };
                let character = char::from_u32(scalar)
                    .ok_or_else(|| self.error("invalid Unicode scalar value"))?;
                let mut buffer = [0u8; 4];
                output.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
            }
            other => {
                return Err(self.error(format!("invalid escape sequence \\{other}")));
            }
        }
        Ok(())
    }

    fn parse_hex4(&mut self) -> Result<u16> {
        let mut value = 0u16;
        for _ in 0..4 {
            let byte = self.next()?;
            let digit = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => return Err(self.error("invalid hexadecimal digit in \\u escape")),
            };
            value = (value << 4) | u16::from(digit);
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<Number> {
        let start = self.position;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.position += 1;
        }

        match self.peek() {
            Some(b'0') => {
                self.position += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error("leading zero in number"));
                }
            }
            Some(b'1'..=b'9') => {
                self.position += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.position += 1;
                }
            }
            _ => return Err(self.error("invalid number: expected a digit")),
        }

        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.position += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid number: expected a digit after the decimal point"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }

        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid number: expected a digit in the exponent"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }

        let token = std::str::from_utf8(&self.input[start..self.position])
            .map_err(|_| self.error("number is not valid UTF-8"))?;

        if is_float {
            let value: f64 = token
                .parse()
                .map_err(|_| self.error("invalid floating-point number"))?;
            if !value.is_finite() {
                return Err(self.error("number overflows to a non-finite value"));
            }
            return Ok(Number::Float(value));
        }

        if negative {
            match token.parse::<i64>() {
                Ok(value) => Ok(Number::Signed(value)),
                Err(_) => {
                    let value: f64 = token.parse().map_err(|_| self.error("invalid integer"))?;
                    if !value.is_finite() {
                        return Err(self.error("number overflows to a non-finite value"));
                    }
                    Ok(Number::Float(value))
                }
            }
        } else {
            match token.parse::<u64>() {
                Ok(value) => Ok(Number::Unsigned(value)),
                Err(_) => {
                    let value: f64 = token.parse().map_err(|_| self.error("invalid integer"))?;
                    if !value.is_finite() {
                        return Err(self.error("number overflows to a non-finite value"));
                    }
                    Ok(Number::Float(value))
                }
            }
        }
    }
}

/// Serializes a JSON value into compact, deterministic JSON.
///
/// Object keys are emitted in BTreeMap order. Strings are escaped so the output
/// is valid RFC 8259 text. A non-finite number is rejected because JSON cannot
/// represent NaN or infinity; this is an internal invariant error, not hostile
/// input.
pub fn to_string(value: &Value) -> Result<String> {
    let mut output = String::new();
    write_value(value, &mut output)?;
    Ok(output)
}

fn write_value(value: &Value, output: &mut String) -> Result<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(true) => output.push_str("true"),
        Value::Bool(false) => output.push_str("false"),
        Value::Number(Number::Unsigned(value)) => output.push_str(&value.to_string()),
        Value::Number(Number::Signed(value)) => output.push_str(&value.to_string()),
        Value::Number(Number::Float(value)) => {
            if !value.is_finite() {
                return Err(Error::internal("cannot serialize a non-finite JSON number"));
            }
            output.push_str(&value.to_string());
        }
        Value::String(value) => write_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_value(value, output)?;
            }
            output.push(']');
        }
        Value::Object(object) => {
            output.push('{');
            for (index, (key, value)) in object.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_string(key, output);
                output.push(':');
                write_value(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn write_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character < '\u{20}' => {
                output.push_str("\\u");
                let scalar = character as u32;
                for shift in [12u32, 8, 4, 0] {
                    let digit = ((scalar >> shift) & 0xF) as u8;
                    output.push(if digit < 10 {
                        (b'0' + digit) as char
                    } else {
                        (b'a' + digit - 10) as char
                    });
                }
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(input: &str) -> Value {
        parse(input).expect("input must parse")
    }

    fn parse_err(input: &str) -> Error {
        parse(input).expect_err("input must be rejected")
    }

    #[test]
    fn parses_primitives() {
        assert_eq!(parse_ok("null"), Value::Null);
        assert_eq!(parse_ok("true"), Value::Bool(true));
        assert_eq!(parse_ok("false"), Value::Bool(false));
        assert_eq!(parse_ok("0"), Value::Number(Number::Unsigned(0)));
        assert_eq!(parse_ok("-1"), Value::Number(Number::Signed(-1)));
        assert_eq!(parse_ok("1.5"), Value::Number(Number::Float(1.5)));
        assert_eq!(parse_ok("1e3"), Value::Number(Number::Float(1000.0)));
        assert_eq!(parse_ok("\"hello\""), Value::String("hello".to_owned()));
    }

    #[test]
    fn parses_nested_containers() {
        let value = parse_ok("{\"a\":[1,2,{\"b\":null}],\"c\":true}");
        let object = value.as_object().expect("object");
        assert_eq!(object.len(), 2);
        let array = object.get("a").and_then(Value::as_array).expect("array");
        assert_eq!(array.len(), 3);
        assert_eq!(array[0].as_number().and_then(Number::as_u64), Some(1));
        assert_eq!(array[2].get("b"), Some(&Value::Null));
        assert_eq!(object.get("c").and_then(Value::as_bool), Some(true));
    }

    #[test]
    fn object_field_order_is_irrelevant() {
        let first = parse_ok("{\"b\":2,\"a\":1}");
        let second = parse_ok("{\"a\":1,\"b\":2}");
        assert_eq!(first, second);
    }

    #[test]
    fn decodes_string_escapes() {
        let value =
            parse_ok("\"quote:\\\" slash:\\/ backslash:\\\\ b:\\b f:\\f n:\\n r:\\r t:\\t\"");
        assert_eq!(
            value.as_str().expect("string"),
            "quote:\" slash:/ backslash:\\ b:\u{0008} f:\u{000c} n:\n r:\r t:\t"
        );
    }

    #[test]
    fn decodes_unicode_escapes() {
        assert_eq!(parse_ok("\"\\u00e9\""), Value::String("é".to_owned()));
        assert_eq!(parse_ok("\"\\u20ac\""), Value::String("€".to_owned()));
        assert_eq!(
            parse_ok("\"\\ud83d\\ude00\""),
            Value::String("😀".to_owned())
        );
        assert_eq!(parse_ok("\"\\u0000\""), Value::String("\0".to_owned()));
    }

    #[test]
    fn accepts_raw_utf8() {
        assert_eq!(
            parse_ok("\"héllo 😀\""),
            Value::String("héllo 😀".to_owned())
        );
    }

    #[test]
    fn rejects_lone_surrogates() {
        for input in ["\"\\ud83d\"", "\"\\ude00\"", "\"\\ud83d\\u0041\""] {
            let error = parse_err(input);
            assert!(matches!(error, Error::Corrupt(_)), "input {input}");
            assert!(error.to_string().contains("surrogate"), "input {input}");
        }
    }

    #[test]
    fn rejects_invalid_escapes() {
        for input in ["\"\\x\"", "\"\\u12\"", "\"\\uZZZZ\""] {
            assert!(
                matches!(parse_err(input), Error::Corrupt(_)),
                "input {input}"
            );
        }
    }

    #[test]
    fn rejects_unescaped_control_characters() {
        let input = "\"a\nb\"";
        let error = parse_err(input);
        assert!(matches!(error, Error::Corrupt(_)));
        assert!(error.to_string().contains("control character"));
    }

    #[test]
    fn rejects_duplicate_keys() {
        let error = parse_err("{\"a\":1,\"a\":2}");
        assert!(matches!(error, Error::Corrupt(_)));
        assert!(error.to_string().contains("duplicate"));
    }

    #[test]
    fn rejects_trailing_commas() {
        for input in ["[1,]", "{\"a\":1,}"] {
            let error = parse_err(input);
            assert!(matches!(error, Error::Corrupt(_)), "input {input}");
            assert!(
                error.to_string().contains("trailing comma"),
                "input {input}"
            );
        }
    }

    #[test]
    fn rejects_malformed_numbers() {
        for input in [
            "01", "+1", ".5", "1.", "1e", "1e+", "1e400", "NaN", "Infinity",
        ] {
            assert!(
                matches!(parse_err(input), Error::Corrupt(_)),
                "input {input}"
            );
        }
    }

    #[test]
    fn preserves_integer_precision() {
        let max = parse_ok("18446744073709551615");
        assert_eq!(max.as_number().and_then(Number::as_u64), Some(u64::MAX));
        let min = parse_ok("-9223372036854775808");
        assert_eq!(min.as_number().and_then(Number::as_i64), Some(i64::MIN));
        let overflow = parse_ok("18446744073709551616");
        assert!(matches!(overflow.as_number(), Some(Number::Float(_))));
    }

    #[test]
    fn rejects_trailing_content() {
        for input in ["null null", "{} []", "1 2"] {
            let error = parse_err(input);
            assert!(matches!(error, Error::Corrupt(_)), "input {input}");
            assert!(error.to_string().contains("trailing"), "input {input}");
        }
    }

    #[test]
    fn rejects_empty_input() {
        assert!(matches!(parse_err(""), Error::Corrupt(_)));
        assert!(matches!(parse_err("   "), Error::Corrupt(_)));
    }

    #[test]
    fn rejects_non_string_object_keys() {
        assert!(matches!(parse_err("{1:2}"), Error::Corrupt(_)));
    }

    #[test]
    fn rejects_invalid_utf8_bytes() {
        let error = parse_bytes(b"\"\xff\"").expect_err("invalid UTF-8 must fail");
        assert!(matches!(error, Error::Corrupt(_)));
        assert!(error.to_string().contains("UTF-8"));
    }

    #[test]
    fn enforces_input_size_cap() {
        let limits = Limits {
            max_input_bytes: 4,
            ..Limits::default()
        };
        let error =
            parse_with_limits("[1,2,3]", &limits).expect_err("input above the cap must fail");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn enforces_depth_cap() {
        let limits = Limits {
            max_depth: 2,
            ..Limits::default()
        };
        assert!(parse_with_limits("[[1]]", &limits).is_ok());
        let error =
            parse_with_limits("[[[1]]]", &limits).expect_err("depth above the cap must fail");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn enforces_string_size_cap() {
        let limits = Limits {
            max_string_bytes: 2,
            ..Limits::default()
        };
        assert!(parse_with_limits("\"ab\"", &limits).is_ok());
        let error =
            parse_with_limits("\"abc\"", &limits).expect_err("string above the cap must fail");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn enforces_container_entry_cap() {
        let limits = Limits {
            max_container_entries: 2,
            ..Limits::default()
        };
        assert!(parse_with_limits("[1,2]", &limits).is_ok());
        let error =
            parse_with_limits("[1,2,3]", &limits).expect_err("array above the cap must fail");
        assert!(matches!(error, Error::LimitExceeded(_)));

        let error = parse_with_limits("{\"a\":1,\"b\":2,\"c\":3}", &limits)
            .expect_err("object above the cap must fail");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn enforces_total_value_cap() {
        let limits = Limits {
            max_total_values: 3,
            ..Limits::default()
        };
        let error = parse_with_limits("[1,2,3]", &limits).expect_err("total value cap must fail");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn number_accessors_are_conservative() {
        let float = parse_ok("1.0").as_number().expect("number");
        assert_eq!(float.as_u64(), None);
        assert_eq!(float.as_i64(), None);
        assert_eq!(float.as_f64(), 1.0);
        assert!(!float.is_integer());

        let negative = parse_ok("-1").as_number().expect("number");
        assert_eq!(negative.as_u64(), None);
        assert_eq!(negative.as_i64(), Some(-1));
        assert!(negative.is_integer());
    }

    #[test]
    fn serializer_round_trips_nested_values() {
        let value = parse_ok("{\"a\":[1,2.5,{\"b\":null}],\"c\":true,\"d\":\"héllo\"}");
        let encoded = to_string(&value).expect("serialize");
        assert_eq!(parse_ok(&encoded), value);
    }

    #[test]
    fn serializer_escapes_strings_and_controls() {
        let value = Value::String(
            "quote:\" slash:/ backslash:\\ b:\u{0008} n:\n r:\r t:\t nul:\u{0000}".to_owned(),
        );
        let encoded = to_string(&value).expect("serialize");
        assert_eq!(parse_ok(&encoded), value);
        assert!(encoded.contains("\\u0000"));
        assert!(encoded.contains("\\n"));
    }

    #[test]
    fn serializer_emits_deterministic_object_order() {
        let first = parse_ok("{\"b\":2,\"a\":1}");
        let second = parse_ok("{\"a\":1,\"b\":2}");
        assert_eq!(
            to_string(&first).expect("first"),
            to_string(&second).expect("second")
        );
        assert_eq!(to_string(&first).expect("serialize"), "{\"a\":1,\"b\":2}");
    }

    #[test]
    fn serializer_rejects_non_finite_numbers() {
        let value = Value::Number(Number::Float(f64::NAN));
        let error = to_string(&value).expect_err("NaN is not JSON");
        assert!(matches!(error, Error::Internal(_)));
    }
}
