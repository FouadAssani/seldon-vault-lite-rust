//! Strict HTTP/1.1 request parsing and response writing.
//!
//! The server accepts one request per connection and closes the connection
//! after the response. This keeps the state machine small and bounds the work
//! one slow client can retain. Parsing is deliberately strict: only HTTP/1.1,
//! a Host header, decimal Content-Length, and no transfer encoding are
//! accepted. Every line and header is bounded before it is buffered, and the
//! body length is checked against the configured cap before allocation.

use std::io::{Read, Write};

use crate::error::Result as CrateResult;
use crate::json::{self, Value};

/// Maximum request-line length in bytes.
pub(crate) const MAX_REQUEST_LINE_BYTES: usize = 8 * 1024;
/// Maximum single header-line length in bytes.
pub(crate) const MAX_HEADER_LINE_BYTES: usize = 8 * 1024;
/// Maximum total header bytes, including line terminators.
pub(crate) const MAX_HEADER_BYTES: usize = 32 * 1024;
/// Maximum number of header fields.
pub(crate) const MAX_HEADERS: usize = 64;
/// Maximum request-target length in bytes.
pub(crate) const MAX_TARGET_BYTES: usize = 2 * 1024;
/// Maximum query-string length in bytes.
pub(crate) const MAX_QUERY_BYTES: usize = 2 * 1024;

/// HTTP method supported by the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Method {
    /// Read a resource.
    Get,
    /// Create or execute an operation.
    Post,
    /// Delete a resource or cancel a job.
    Delete,
}

impl Method {
    /// Parses a method token.
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        match bytes {
            b"GET" => Some(Self::Get),
            b"POST" => Some(Self::Post),
            b"DELETE" => Some(Self::Delete),
            _ => None,
        }
    }

    /// Method name used in responses and error messages.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Delete => "DELETE",
        }
    }
}

/// One parsed request header. Names are lowercased for case-insensitive lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Header {
    /// Lowercased field name.
    pub(crate) name: String,
    /// Trimmed field value.
    pub(crate) value: String,
}

/// A complete HTTP request.
#[derive(Debug)]
pub(crate) struct Request {
    /// Request method.
    pub(crate) method: Method,
    /// Raw path without the query string.
    pub(crate) path: String,
    /// Optional query string without the leading question mark.
    pub(crate) query: Option<String>,
    /// Headers in wire order.
    pub(crate) headers: Vec<Header>,
    /// Request body, already bounded.
    pub(crate) body: Vec<u8>,
}

impl Request {
    /// Returns the first header with the given case-insensitive name.
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|header| header.name == name)
            .map(|header| header.value.as_str())
    }
}

/// A response ready to be written to the socket.
#[derive(Debug)]
pub(crate) struct Response {
    /// HTTP status code.
    pub(crate) status: u16,
    /// Optional Content-Type.
    pub(crate) content_type: Option<&'static str>,
    /// Response body.
    pub(crate) body: Vec<u8>,
    /// Additional response headers.
    pub(crate) headers: Vec<(String, String)>,
}

impl Response {
    /// Builds a JSON response from a value.
    pub(crate) fn json(status: u16, value: &Value) -> CrateResult<Self> {
        let body = json::to_string(value)?.into_bytes();
        Ok(Self {
            status,
            content_type: Some("application/json; charset=utf-8"),
            body,
            headers: Vec::new(),
        })
    }

    /// Builds an empty response.
    pub(crate) fn empty(status: u16) -> Self {
        Self {
            status,
            content_type: None,
            body: Vec::new(),
            headers: Vec::new(),
        }
    }
}

/// A request failure that maps directly to an HTTP response.
#[derive(Debug)]
pub(crate) struct HttpError {
    /// HTTP status code.
    pub(crate) status: u16,
    /// Stable machine-readable code.
    pub(crate) code: &'static str,
    /// Human-readable message.
    pub(crate) message: String,
}

impl HttpError {
    /// Builds a typed request error.
    pub(crate) fn new(status: u16, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    /// A malformed request.
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self::new(400, "invalid_request", message)
    }

    /// An unsupported feature or format.
    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self::new(501, "unsupported", message)
    }

    /// A body above the configured cap.
    pub(crate) fn too_large(message: impl Into<String>) -> Self {
        Self::new(413, "payload_too_large", message)
    }

    /// A request that timed out.
    pub(crate) fn timeout(message: impl Into<String>) -> Self {
        Self::new(408, "request_timeout", message)
    }

    /// An internal server failure.
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(500, "internal_error", message)
    }
}

/// Reads and validates one HTTP request.
///
/// # Errors
///
/// Returns a typed error for an incomplete or malformed request, an unsupported
/// transfer encoding, a missing Host header, an oversized body, or a read
/// timeout. The caller writes the error response and closes the connection.
pub(crate) fn read_request<R: Read>(
    reader: &mut R,
    max_body_bytes: usize,
) -> Result<Request, HttpError> {
    let request_line = read_crlf_line(reader, MAX_REQUEST_LINE_BYTES)
        .map_err(|error| error.context("request line"))?;
    if request_line.is_empty() {
        return Err(HttpError::bad_request("empty request line"));
    }

    let (method, target) = parse_request_line(&request_line)?;
    let (path, query) = split_target(target)?;

    let mut headers = Vec::new();
    let mut header_bytes = 0usize;
    loop {
        let line = read_crlf_line(reader, MAX_HEADER_LINE_BYTES)
            .map_err(|error| error.context("header"))?;
        header_bytes = header_bytes
            .checked_add(line.len() + 2)
            .ok_or_else(|| HttpError::bad_request("header size overflow"))?;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(HttpError::bad_request("headers exceed the byte cap"));
        }
        if line.is_empty() {
            break;
        }
        if headers.len() >= MAX_HEADERS {
            return Err(HttpError::bad_request("too many headers"));
        }
        headers.push(parse_header(&line)?);
    }

    validate_headers(&headers)?;
    let host = headers
        .iter()
        .find(|header| header.name == "host")
        .ok_or_else(|| HttpError::bad_request("HTTP/1.1 requires a Host header"))?;
    if host.value.is_empty() {
        return Err(HttpError::bad_request("Host header must not be empty"));
    }

    if headers
        .iter()
        .any(|header| header.name == "transfer-encoding")
    {
        return Err(HttpError::unsupported(
            "Transfer-Encoding is not supported; use Content-Length",
        ));
    }
    if headers.iter().any(|header| header.name == "expect") {
        return Err(HttpError::new(
            417,
            "expectation_failed",
            "Expect headers are not supported",
        ));
    }

    let content_length = match headers
        .iter()
        .find(|header| header.name == "content-length")
    {
        Some(header) => parse_content_length(&header.value)?,
        None => 0,
    };

    match method {
        Method::Get | Method::Delete if content_length > 0 => {
            return Err(HttpError::bad_request(
                "GET and DELETE requests must not carry a body",
            ));
        }
        Method::Post if !headers.iter().any(|header| header.name == "content-length") => {
            return Err(HttpError::bad_request(
                "POST requires a Content-Length header",
            ));
        }
        _ => {}
    }

    let max_body_bytes_u64 = u64::try_from(max_body_bytes).unwrap_or(u64::MAX);
    if content_length > max_body_bytes_u64 {
        return Err(HttpError::too_large(format!(
            "request body is {content_length} bytes, above the {max_body_bytes}-byte cap"
        )));
    }
    let content_length = usize::try_from(content_length)
        .map_err(|_| HttpError::too_large("request body does not fit usize"))?;

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).map_err(map_io_error)?;
    }

    Ok(Request {
        method,
        path,
        query,
        headers,
        body,
    })
}

fn parse_request_line(line: &[u8]) -> Result<(Method, &[u8]), HttpError> {
    let mut parts = line.split(|byte| *byte == b' ');
    let method = parts
        .next()
        .ok_or_else(|| HttpError::bad_request("missing method"))?;
    let target = parts
        .next()
        .ok_or_else(|| HttpError::bad_request("missing request target"))?;
    let version = parts
        .next()
        .ok_or_else(|| HttpError::bad_request("missing HTTP version"))?;
    if parts.next().is_some() {
        return Err(HttpError::bad_request("request line has extra whitespace"));
    }
    if method.is_empty() || target.is_empty() || version.is_empty() {
        return Err(HttpError::bad_request(
            "request line contains an empty field",
        ));
    }
    if version != b"HTTP/1.1" {
        return Err(HttpError::bad_request("only HTTP/1.1 is supported"));
    }
    let method = Method::from_bytes(method)
        .ok_or_else(|| HttpError::bad_request("unsupported HTTP method"))?;
    Ok((method, target))
}

fn split_target(target: &[u8]) -> Result<(String, Option<String>), HttpError> {
    if target.len() > MAX_TARGET_BYTES {
        return Err(HttpError::bad_request("request target is too long"));
    }
    if target.first() != Some(&b'/') {
        return Err(HttpError::bad_request(
            "request target must start with a slash",
        ));
    }
    if target.contains(&b'#') {
        return Err(HttpError::bad_request(
            "fragments are not allowed in a request target",
        ));
    }
    for byte in target {
        if *byte < 0x20 || *byte == 0x7F {
            return Err(HttpError::bad_request(
                "request target contains a control character",
            ));
        }
    }
    let (path, query) = match target.iter().position(|byte| *byte == b'?') {
        Some(index) => (&target[..index], Some(&target[index + 1..])),
        None => (target, None),
    };
    let path = std::str::from_utf8(path)
        .map_err(|_| HttpError::bad_request("request path is not valid UTF-8"))?
        .to_owned();
    let query = match query {
        Some(query) => {
            if query.len() > MAX_QUERY_BYTES {
                return Err(HttpError::bad_request("query string is too long"));
            }
            Some(
                std::str::from_utf8(query)
                    .map_err(|_| HttpError::bad_request("query string is not valid UTF-8"))?
                    .to_owned(),
            )
        }
        None => None,
    };
    Ok((path, query))
}

fn parse_header(line: &[u8]) -> Result<Header, HttpError> {
    if line.first() == Some(&b' ') || line.first() == Some(&b'\t') {
        return Err(HttpError::bad_request(
            "obsolete header line folding is not allowed",
        ));
    }
    let colon = line
        .iter()
        .position(|byte| *byte == b':')
        .ok_or_else(|| HttpError::bad_request("header is missing a colon"))?;
    let name = &line[..colon];
    let value = &line[colon + 1..];
    if name.is_empty() {
        return Err(HttpError::bad_request("header name must not be empty"));
    }
    if !name.iter().all(|byte| is_token_char(*byte)) {
        return Err(HttpError::bad_request("header name contains invalid bytes"));
    }
    let mut value = value;
    while value
        .first()
        .is_some_and(|byte| *byte == b' ' || *byte == b'\t')
    {
        value = &value[1..];
    }
    while value
        .last()
        .is_some_and(|byte| *byte == b' ' || *byte == b'\t')
    {
        value = &value[..value.len() - 1];
    }
    if value.len() > MAX_HEADER_LINE_BYTES {
        return Err(HttpError::bad_request("header value is too long"));
    }
    for byte in value {
        if (*byte < 0x20 && *byte != b'\t') || *byte == 0x7F {
            return Err(HttpError::bad_request(
                "header value contains a control character",
            ));
        }
    }
    let name = std::str::from_utf8(name)
        .map_err(|_| HttpError::bad_request("header name is not valid UTF-8"))?
        .to_ascii_lowercase();
    let value = std::str::from_utf8(value)
        .map_err(|_| HttpError::bad_request("header value is not valid UTF-8"))?
        .to_owned();
    Ok(Header { name, value })
}

fn validate_headers(headers: &[Header]) -> Result<(), HttpError> {
    for (index, header) in headers.iter().enumerate() {
        if matches!(
            header.name.as_str(),
            "host"
                | "content-length"
                | "transfer-encoding"
                | "authorization"
                | "expect"
                | "content-type"
                | "x-api-key"
        ) && headers[..index]
            .iter()
            .any(|previous| previous.name == header.name)
        {
            return Err(HttpError::bad_request(format!(
                "duplicate {} header",
                header.name
            )));
        }
    }
    Ok(())
}

fn parse_content_length(value: &str) -> Result<u64, HttpError> {
    if value.is_empty() {
        return Err(HttpError::bad_request("Content-Length must not be empty"));
    }
    if !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(HttpError::bad_request(
            "Content-Length must be a decimal integer",
        ));
    }
    value
        .parse::<u64>()
        .map_err(|_| HttpError::bad_request("Content-Length overflows u64"))
}

fn is_token_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

fn read_crlf_line<R: Read>(reader: &mut R, max_bytes: usize) -> Result<Vec<u8>, HttpError> {
    let mut line = Vec::new();
    let mut pending_cr = false;
    loop {
        let mut byte = [0u8; 1];
        match reader.read(&mut byte) {
            Ok(0) => {
                return Err(HttpError::bad_request(
                    "connection closed before the request was complete",
                ));
            }
            Ok(_) => {}
            Err(error) => return Err(map_io_error(error)),
        }
        if pending_cr {
            if byte[0] != b'\n' {
                return Err(HttpError::bad_request("line terminator must be CRLF"));
            }
            return Ok(line);
        }
        match byte[0] {
            b'\r' => pending_cr = true,
            b'\n' => {
                return Err(HttpError::bad_request("line terminator must be CRLF"));
            }
            byte => {
                if line.len() >= max_bytes {
                    return Err(HttpError::bad_request("line exceeds the byte cap"));
                }
                line.push(byte);
            }
        }
    }
}

fn map_io_error(error: std::io::Error) -> HttpError {
    if matches!(
        error.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    ) {
        HttpError::timeout("request read timed out")
    } else {
        HttpError::bad_request(format!("request read failed: {error}"))
    }
}

impl HttpError {
    fn context(mut self, context: &str) -> Self {
        self.message = format!("{context}: {}", self.message);
        self
    }
}

/// Decodes one percent-encoded path segment as UTF-8.
///
/// # Errors
///
/// Returns a bad-request error for a malformed escape, an invalid UTF-8
/// sequence, or a control character in the decoded segment.
pub(crate) fn decode_path_segment(segment: &str) -> Result<String, HttpError> {
    let bytes = segment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' => {
                let high = bytes
                    .get(index + 1)
                    .and_then(|byte| hex_value(*byte))
                    .ok_or_else(|| HttpError::bad_request("invalid percent escape"))?;
                let low = bytes
                    .get(index + 2)
                    .and_then(|byte| hex_value(*byte))
                    .ok_or_else(|| HttpError::bad_request("invalid percent escape"))?;
                decoded.push((high << 4) | low);
                index += 3;
            }
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }
    let value = String::from_utf8(decoded)
        .map_err(|_| HttpError::bad_request("path segment is not valid UTF-8"))?;
    if value.chars().any(char::is_control) {
        return Err(HttpError::bad_request(
            "path segment contains a control character",
        ));
    }
    Ok(value)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Writes one response and closes the connection.
///
/// # Errors
///
/// Returns the underlying I/O error. The caller treats a failed write as a
/// closed connection, never as a panic.
pub(crate) fn write_response<W: Write>(writer: &mut W, response: &Response) -> std::io::Result<()> {
    let reason = reason_phrase(response.status);
    write!(
        writer,
        "HTTP/1.1 {} {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n",
        response.status, reason
    )?;
    if let Some(content_type) = response.content_type {
        write!(writer, "Content-Type: {content_type}\r\n")?;
    }
    for (name, value) in &response.headers {
        write!(writer, "{name}: {value}\r\n")?;
    }
    write!(writer, "Content-Length: {}\r\n\r\n", response.body.len())?;
    writer.write_all(&response.body)?;
    writer.flush()
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        417 => "Expectation Failed",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse(raw: &[u8], cap: usize) -> Result<Request, HttpError> {
        let mut cursor = Cursor::new(raw);
        read_request(&mut cursor, cap)
    }

    #[test]
    fn parses_a_minimal_get() {
        let request =
            parse(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", 1024).expect("request");
        assert_eq!(request.method, Method::Get);
        assert_eq!(request.path, "/health");
        assert_eq!(request.query, None);
        assert_eq!(request.header("host"), Some("localhost"));
        assert!(request.body.is_empty());
    }

    #[test]
    fn parses_a_post_with_query_and_body() {
        let raw = b"POST /v1/search?page=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 4\r\n\r\nnull";
        let request = parse(raw, 1024).expect("request");
        assert_eq!(request.method, Method::Post);
        assert_eq!(request.path, "/v1/search");
        assert_eq!(request.query.as_deref(), Some("page=1"));
        assert_eq!(request.body, b"null");
    }

    #[test]
    fn rejects_malformed_request_lines() {
        for raw in [
            &b"GET /health\n\r\nHost: x\r\n\r\n"[..],
            &b"GET /health HTTP/1.0\r\nHost: x\r\n\r\n"[..],
            &b"GET health HTTP/1.1\r\nHost: x\r\n\r\n"[..],
            &b"GET  /health HTTP/1.1\r\nHost: x\r\n\r\n"[..],
            &b"PUT /health HTTP/1.1\r\nHost: x\r\n\r\n"[..],
        ] {
            assert!(parse(raw, 1024).is_err(), "accepted {raw:?}");
        }
    }

    #[test]
    fn requires_host_and_valid_content_length() {
        for raw in [
            &b"GET /health HTTP/1.1\r\n\r\n"[..],
            &b"POST /v1/search HTTP/1.1\r\nHost: x\r\n\r\n"[..],
            &b"POST /v1/search HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n"[..],
            &b"POST /v1/search HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\n"[..],
        ] {
            assert!(parse(raw, 1024).is_err(), "accepted {raw:?}");
        }
    }

    #[test]
    fn rejects_transfer_encoding_and_expect() {
        let raw = b"POST /v1/search HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n";
        assert_eq!(parse(raw, 1024).expect_err("TE").status, 501);
        let raw = b"POST /v1/search HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\nExpect: 100-continue\r\n\r\nx";
        assert_eq!(parse(raw, 1024).expect_err("Expect").status, 417);
    }

    #[test]
    fn enforces_the_body_cap_before_reading() {
        let raw = b"POST /v1/search HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n";
        assert_eq!(parse(raw, 10).expect_err("cap").status, 413);
    }

    #[test]
    fn rejects_a_short_body() {
        let raw = b"POST /v1/search HTTP/1.1\r\nHost: x\r\nContent-Length: 4\r\n\r\nab";
        assert!(parse(raw, 1024).is_err());
    }

    #[test]
    fn decodes_percent_encoded_path_segments() {
        assert_eq!(decode_path_segment("doc%2F1").expect("slash"), "doc/1");
        assert_eq!(decode_path_segment("caf%C3%A9").expect("utf8"), "café");
        assert!(decode_path_segment("%ZZ").is_err());
        assert!(decode_path_segment("%00").is_err());
    }
}
