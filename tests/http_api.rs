//! Integration tests for the milestone M6 HTTP API.
//!
//! The tests start the real server on an ephemeral loopback port and speak raw
//! HTTP/1.1 over std::net::TcpStream. They cover health, readiness, model
//! metadata, embeddings, ingestion jobs, search, deletion, authentication,
//! malformed requests, connection limits, slow connections, Unicode JSON, and
//! graceful shutdown. No third-party HTTP client is used.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use seldon_vault_lite::api::{RunningServer, Server, ServerConfig};
use seldon_vault_lite::json::{self, Value};

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TestServer {
    running: RunningServer,
    data_dir: PathBuf,
}

impl TestServer {
    fn start(model_dir: &str, token: Option<&str>) -> Self {
        Self::start_with(model_dir, token, 32, 2_000, 8)
    }

    fn start_with(
        model_dir: &str,
        token: Option<&str>,
        max_connections: usize,
        request_timeout_ms: u64,
        dimension: usize,
    ) -> Self {
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("seldon-http-{}-{unique}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let data_dir = dir.join("data");
        let config = ServerConfig {
            bind: "127.0.0.1:0".parse().expect("address"),
            data_dir: data_dir.clone(),
            model_dir: PathBuf::from(model_dir),
            dimension,
            vision_budget: 70,
            access_token: token.map(str::to_owned),
            max_connections,
            request_timeout: Duration::from_millis(request_timeout_ms),
            max_body_bytes: 4 * 1024 * 1024,
            queue_capacity: 8,
            job_workers: 1,
        };
        let server = Server::bind(config).expect("bind server");
        let running = server.spawn().expect("spawn server");
        Self { running, data_dir }
    }

    fn addr(&self) -> SocketAddr {
        self.running.local_addr()
    }
}

#[derive(Debug)]
struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, value)| value.as_str())
    }

    fn json(&self) -> Value {
        json::parse(std::str::from_utf8(&self.body).expect("response body is UTF-8"))
            .expect("response body is JSON")
    }
}

fn request(addr: SocketAddr, raw: &[u8]) -> HttpResponse {
    let mut stream = TcpStream::connect(addr).expect("connect");
    // The scalar image tower can take over a minute in the ignored full-model
    // test; keep a generous client-side read timeout for that case.
    stream
        .set_read_timeout(Some(Duration::from_secs(180)))
        .expect("read timeout");
    stream.write_all(raw).expect("write request");
    let mut bytes = Vec::new();
    match stream.read_to_end(&mut bytes) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::ConnectionReset && !bytes.is_empty() => {}
        Err(error) => panic!("read response: {error:?}"),
    }
    parse_response(&bytes)
}

fn http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: &str,
    token: Option<&str>,
) -> HttpResponse {
    let mut raw = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    if let Some(token) = token {
        raw.push_str(&format!("Authorization: Bearer {token}\r\n"));
    }
    raw.push_str("\r\n");
    raw.push_str(body);
    request(addr, raw.as_bytes())
}

fn parse_response(bytes: &[u8]) -> HttpResponse {
    let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
        panic!(
            "response has no header terminator: {:?}",
            String::from_utf8_lossy(bytes)
        );
    };
    let head = std::str::from_utf8(&bytes[..position]).expect("response head is UTF-8");
    let mut lines = head.split("\r\n");
    let status_line = lines.next().expect("status line");
    let status = status_line
        .split(' ')
        .nth(1)
        .expect("status code")
        .parse::<u16>()
        .expect("numeric status");
    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
        }
    }
    let body = bytes[position + 4..].to_vec();
    let response = HttpResponse {
        status,
        headers,
        body,
    };
    if let Some(length) = response.header("content-length") {
        assert_eq!(
            length.parse::<usize>().expect("numeric length"),
            response.body.len(),
            "Content-Length must match the body"
        );
    }
    response
}

fn wait_job(addr: SocketAddr, id: u64, token: Option<&str>) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let response = http_request(addr, "GET", &format!("/v1/jobs/{id}"), "", token);
        assert_eq!(response.status, 200, "job status response");
        let value = response.json();
        let job = value.get("job").expect("job object");
        let status = job
            .get("status")
            .and_then(Value::as_str)
            .expect("job status");
        if matches!(status, "succeeded" | "failed" | "cancelled") {
            return value;
        }
        assert!(
            Instant::now() < deadline,
            "job {id} did not finish; last status {status}"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied().unwrap_or(0);
        let third = chunk.get(2).copied().unwrap_or(0);
        output.push(TABLE[(first >> 2) as usize] as char);
        output.push(TABLE[(((first & 0x03) << 4) | (second >> 4)) as usize] as char);
        if chunk.len() > 1 {
            output.push(TABLE[(((second & 0x0F) << 2) | (third >> 6)) as usize] as char);
        } else {
            output.push('=');
        }
        if chunk.len() > 2 {
            output.push(TABLE[(third & 0x3F) as usize] as char);
        } else {
            output.push('=');
        }
    }
    output
}

fn vector_norm(values: &[Value]) -> f64 {
    values
        .iter()
        .map(|value| {
            let value = value.as_number().expect("number").as_f64();
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

#[test]
fn health_ready_models_and_stats_are_available() {
    let server = TestServer::start("fixtures/model/tiny_text", None);
    let addr = server.addr();
    assert!(server.data_dir.is_dir());

    let health = http_request(addr, "GET", "/health", "", None);
    assert_eq!(health.status, 200);
    let health = health.json();
    assert_eq!(health.get("status").and_then(Value::as_str), Some("ok"));

    let ready = http_request(addr, "GET", "/ready", "", None);
    assert_eq!(ready.status, 200);
    let ready = ready.json();
    assert_eq!(ready.get("ready").and_then(Value::as_bool), Some(true));
    assert_eq!(
        ready.get("text_input").and_then(Value::as_bool),
        Some(false)
    );
    let models = ready.get("models").expect("models object");
    assert_eq!(models.get("text").and_then(Value::as_bool), Some(true));
    assert_eq!(models.get("image").and_then(Value::as_bool), Some(false));
    assert_eq!(models.get("audio").and_then(Value::as_bool), Some(false));

    let metadata = http_request(addr, "GET", "/v1/models", "", None);
    assert_eq!(metadata.status, 200);
    let metadata = metadata.json();
    assert_eq!(
        metadata
            .get("embedding_dimension")
            .and_then(Value::as_number)
            .and_then(|number| number.as_u64()),
        Some(8)
    );
    assert_eq!(
        metadata
            .get("modalities")
            .and_then(|value| value.get("text"))
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        metadata
            .get("matryoshka_dimensions")
            .and_then(Value::as_array)
            .map(<[Value]>::len),
        Some(0)
    );

    let stats = http_request(addr, "GET", "/v1/stats", "", None);
    assert_eq!(stats.status, 200);
    let stats = stats.json();
    assert!(stats.get("jobs").is_some());
    assert!(stats.get("requests_total").is_some());
}

#[test]
fn malformed_requests_are_rejected_without_panicking() {
    let server = TestServer::start("fixtures/model/tiny_text", None);
    let addr = server.addr();

    let unknown = http_request(addr, "GET", "/nope", "", None);
    assert_eq!(unknown.status, 404);

    let wrong_method = http_request(addr, "POST", "/health", "", None);
    assert_eq!(wrong_method.status, 405);
    assert_eq!(wrong_method.header("allow"), Some("GET"));

    let missing_host = request(addr, b"GET /health HTTP/1.1\r\n\r\n");
    assert_eq!(missing_host.status, 400);

    let oversized = request(
        addr,
        b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: 99999999\r\n\r\n",
    );
    assert_eq!(oversized.status, 413);

    let transfer_encoding = request(
        addr,
        b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n",
    );
    assert_eq!(transfer_encoding.status, 501);

    let expect = request(
        addr,
        b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1\r\nExpect: 100-continue\r\n\r\n",
    );
    assert_eq!(expect.status, 417);

    let bad_method = request(addr, b"PUT /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert_eq!(bad_method.status, 400);

    let no_content_type = request(
        addr,
        b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}",
    );
    assert_eq!(no_content_type.status, 415);

    let invalid_json = http_request(addr, "POST", "/v1/embeddings", "{", None);
    assert_eq!(invalid_json.status, 400);

    let query = http_request(addr, "GET", "/health?page=1", "", None);
    assert_eq!(query.status, 400);
}

#[test]
fn token_embeddings_are_real_vectors_and_invalid_inputs_are_typed() {
    let server = TestServer::start("fixtures/model/tiny_text", None);
    let addr = server.addr();

    let response = http_request(
        addr,
        "POST",
        "/v1/embeddings",
        r#"{"input":{"type":"tokens","ids":[1,5,7,9,2]},"dimension":8}"#,
        None,
    );
    assert_eq!(response.status, 200, "body: {:?}", response.json());
    let value = response.json();
    assert_eq!(value.get("modality").and_then(Value::as_str), Some("text"));
    let embedding = value
        .get("embedding")
        .and_then(Value::as_array)
        .expect("embedding array");
    assert_eq!(embedding.len(), 8);
    assert!((vector_norm(embedding) - 1.0).abs() <= 1e-4);

    let out_of_vocabulary = http_request(
        addr,
        "POST",
        "/v1/embeddings",
        r#"{"input":{"type":"tokens","ids":[1,64]},"dimension":8}"#,
        None,
    );
    assert_eq!(out_of_vocabulary.status, 400);

    let invalid_dimension = http_request(
        addr,
        "POST",
        "/v1/embeddings",
        r#"{"input":{"type":"tokens","ids":[1,5]},"dimension":9}"#,
        None,
    );
    assert_eq!(invalid_dimension.status, 400);

    let unknown_field = http_request(
        addr,
        "POST",
        "/v1/embeddings",
        r#"{"input":{"type":"tokens","ids":[1,5],"extra":true}}"#,
        None,
    );
    assert_eq!(unknown_field.status, 400);

    let text_without_tokenizer = http_request(
        addr,
        "POST",
        "/v1/embeddings",
        r#"{"input":{"type":"text","text":"hello"}}"#,
        None,
    );
    assert_eq!(text_without_tokenizer.status, 501);

    let png = std::fs::read("fixtures/media/gray_2x2.png").expect("PNG fixture");
    let image_body = format!(
        r#"{{"input":{{"type":"image","data":"{}"}}}}"#,
        base64_encode(&png)
    );
    let image_without_vision = http_request(addr, "POST", "/v1/embeddings", &image_body, None);
    assert_eq!(image_without_vision.status, 501);
}

#[test]
fn documents_jobs_search_and_delete_work_together() {
    let server = TestServer::start("fixtures/model/tiny_text", None);
    let addr = server.addr();

    let create = http_request(
        addr,
        "POST",
        "/v1/documents",
        r#"{"id":"doc-1","input":{"type":"tokens","ids":[1,5,7,9,2],"text":"ownership and borrowing in rust"},"title":"Rust ownership","metadata":{"lang":"en"}}"#,
        None,
    );
    assert_eq!(create.status, 202);
    let create = create.json();
    let job_id = create
        .get("job")
        .and_then(|job| job.get("id"))
        .and_then(Value::as_number)
        .and_then(|number| number.as_u64())
        .expect("job id");
    let job = wait_job(addr, job_id, None);
    assert_eq!(
        job.get("job")
            .and_then(|job| job.get("status"))
            .and_then(Value::as_str),
        Some("succeeded")
    );

    let cancel_finished = http_request(addr, "DELETE", &format!("/v1/jobs/{job_id}"), "", None);
    assert_eq!(cancel_finished.status, 200);
    assert_eq!(
        cancel_finished.json().get("status").and_then(Value::as_str),
        Some("already_finished")
    );

    let lexical = http_request(
        addr,
        "POST",
        "/v1/search",
        r#"{"query":{"type":"text","text":"ownership"},"mode":"lexical"}"#,
        None,
    );
    assert_eq!(lexical.status, 200);
    let lexical = lexical.json();
    let hits = lexical.get("hits").and_then(Value::as_array).expect("hits");
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].get("document_id").and_then(Value::as_str),
        Some("doc-1")
    );
    assert_eq!(
        hits[0].get("title").and_then(Value::as_str),
        Some("Rust ownership")
    );
    assert_eq!(
        hits[0]
            .get("metadata")
            .and_then(|value| value.get("lang"))
            .and_then(Value::as_str),
        Some("en")
    );

    let vector = http_request(
        addr,
        "POST",
        "/v1/search",
        r#"{"query":{"type":"tokens","ids":[1,5,7,9,2]},"mode":"vector"}"#,
        None,
    );
    assert_eq!(vector.status, 200);
    let vector = vector.json();
    let hits = vector.get("hits").and_then(Value::as_array).expect("hits");
    assert_eq!(hits.len(), 1);
    assert!(hits[0].get("score").and_then(Value::as_number).is_some());

    let hybrid = http_request(
        addr,
        "POST",
        "/v1/search",
        r#"{"query":{"type":"tokens","ids":[1,5,7,9,2],"text":"ownership"},"mode":"hybrid"}"#,
        None,
    );
    assert_eq!(hybrid.status, 200);
    let hybrid = hybrid.json();
    let hits = hybrid.get("hits").and_then(Value::as_array).expect("hits");
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0]
            .get("lexical_rank")
            .and_then(Value::as_number)
            .is_some()
    );
    assert!(
        hits[0]
            .get("vector_rank")
            .and_then(Value::as_number)
            .is_some()
    );

    let filtered = http_request(
        addr,
        "POST",
        "/v1/search",
        r#"{"query":{"type":"text","text":"ownership"},"mode":"lexical","filters":{"metadata":{"lang":"en"}}}"#,
        None,
    );
    assert_eq!(filtered.status, 200);
    assert_eq!(
        filtered
            .json()
            .get("hits")
            .and_then(Value::as_array)
            .map(<[Value]>::len),
        Some(1)
    );

    let filtered_out = http_request(
        addr,
        "POST",
        "/v1/search",
        r#"{"query":{"type":"text","text":"ownership"},"mode":"lexical","filters":{"metadata":{"lang":"fr"}}}"#,
        None,
    );
    assert_eq!(filtered_out.status, 200);
    assert_eq!(
        filtered_out
            .json()
            .get("hits")
            .and_then(Value::as_array)
            .map(<[Value]>::len),
        Some(0)
    );

    let deleted = http_request(addr, "DELETE", "/v1/documents/doc-1", "", None);
    assert_eq!(deleted.status, 200);
    assert_eq!(
        deleted.json().get("deleted").and_then(Value::as_bool),
        Some(true)
    );

    let deleted_again = http_request(addr, "DELETE", "/v1/documents/doc-1", "", None);
    assert_eq!(
        deleted_again.json().get("deleted").and_then(Value::as_bool),
        Some(false)
    );

    let after_delete = http_request(
        addr,
        "POST",
        "/v1/search",
        r#"{"query":{"type":"text","text":"ownership"},"mode":"lexical"}"#,
        None,
    );
    assert_eq!(
        after_delete
            .json()
            .get("hits")
            .and_then(Value::as_array)
            .map(<[Value]>::len),
        Some(0)
    );

    let missing_job = http_request(addr, "GET", "/v1/jobs/99999", "", None);
    assert_eq!(missing_job.status, 404);

    let duplicate = http_request(
        addr,
        "POST",
        "/v1/documents",
        r#"{"id":"doc-2","input":{"type":"tokens","ids":[1,5],"text":"second"}}"#,
        None,
    );
    assert_eq!(duplicate.status, 202);
    let duplicate_job = duplicate
        .json()
        .get("job")
        .and_then(|job| job.get("id"))
        .and_then(Value::as_number)
        .and_then(|number| number.as_u64())
        .expect("job id");
    let duplicate_job = wait_job(addr, duplicate_job, None);
    assert_eq!(
        duplicate_job
            .get("job")
            .and_then(|job| job.get("status"))
            .and_then(Value::as_str),
        Some("succeeded")
    );

    let conflict = http_request(
        addr,
        "POST",
        "/v1/documents",
        r#"{"id":"doc-2","input":{"type":"tokens","ids":[1,5],"text":"duplicate"}}"#,
        None,
    );
    assert_eq!(conflict.status, 202);
    let conflict_job = conflict
        .json()
        .get("job")
        .and_then(|job| job.get("id"))
        .and_then(Value::as_number)
        .and_then(|number| number.as_u64())
        .expect("job id");
    let conflict_job = wait_job(addr, conflict_job, None);
    assert_eq!(
        conflict_job
            .get("job")
            .and_then(|job| job.get("status"))
            .and_then(Value::as_str),
        Some("failed")
    );
    assert!(
        conflict_job
            .get("job")
            .and_then(|job| job.get("message"))
            .and_then(Value::as_str)
            .is_some_and(|message| message.contains("already exists"))
    );
}

#[test]
fn unicode_documents_and_percent_encoded_ids_round_trip() {
    let server = TestServer::start("fixtures/model/tiny_text", None);
    let addr = server.addr();

    let create = http_request(
        addr,
        "POST",
        "/v1/documents",
        r#"{"id":"café-日本","input":{"type":"tokens","ids":[1,5,7],"text":"café 日本語 ownership"},"title":"Café 😀","metadata":{"lang":"fr"}}"#,
        None,
    );
    assert_eq!(create.status, 202);
    let job_id = create
        .json()
        .get("job")
        .and_then(|job| job.get("id"))
        .and_then(Value::as_number)
        .and_then(|number| number.as_u64())
        .expect("job id");
    let job = wait_job(addr, job_id, None);
    assert_eq!(
        job.get("job")
            .and_then(|job| job.get("status"))
            .and_then(Value::as_str),
        Some("succeeded")
    );

    let search = http_request(
        addr,
        "POST",
        "/v1/search",
        r#"{"query":{"type":"text","text":"café"},"mode":"lexical"}"#,
        None,
    );
    assert_eq!(search.status, 200);
    let hits = search.json();
    let hits = hits.get("hits").and_then(Value::as_array).expect("hits");
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].get("title").and_then(Value::as_str),
        Some("Café 😀")
    );

    let delete = http_request(
        addr,
        "DELETE",
        "/v1/documents/caf%C3%A9-%E6%97%A5%E6%9C%AC",
        "",
        None,
    );
    assert_eq!(delete.status, 200);
    assert_eq!(
        delete.json().get("deleted").and_then(Value::as_bool),
        Some(true)
    );

    let invalid_metadata = http_request(
        addr,
        "POST",
        "/v1/documents",
        r#"{"id":"bad","input":{"type":"tokens","ids":[1]},"metadata":{"lang":7}}"#,
        None,
    );
    assert_eq!(invalid_metadata.status, 400);
}

#[test]
fn bearer_token_is_enforced_when_configured() {
    let server = TestServer::start("fixtures/model/tiny_text", Some("0123456789abcdef"));
    let addr = server.addr();

    let missing = http_request(addr, "GET", "/health", "", None);
    assert_eq!(missing.status, 401);
    assert_eq!(missing.header("www-authenticate"), Some("Bearer"));

    let wrong = http_request(addr, "GET", "/health", "", Some("wrong-token"));
    assert_eq!(wrong.status, 401);

    let correct = http_request(addr, "GET", "/health", "", Some("0123456789abcdef"));
    assert_eq!(correct.status, 200);
}

#[test]
fn connection_limit_and_slow_request_timeout_are_bounded() {
    let server = TestServer::start_with("fixtures/model/tiny_text", None, 1, 1_000, 8);
    let addr = server.addr();

    let mut first = TcpStream::connect(addr).expect("first connection");
    first
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n")
        .expect("partial request");
    thread::sleep(Duration::from_millis(200));

    // The accept order of simultaneous connections is not guaranteed by the
    // platform. Retry until the server has accepted the slow first connection
    // and therefore rejects a new one.
    let mut saw_limit = false;
    for _ in 0..50 {
        let response = http_request(addr, "GET", "/health", "", None);
        if response.status == 503 {
            saw_limit = true;
            break;
        }
        assert_eq!(response.status, 200);
        thread::sleep(Duration::from_millis(10));
    }
    assert!(saw_limit, "the connection limit was never reached");

    first
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("read timeout");
    let mut response = Vec::new();
    first.read_to_end(&mut response).expect("timeout response");
    let response = parse_response(&response);
    assert_eq!(response.status, 408);

    let after = http_request(addr, "GET", "/health", "", None);
    assert_eq!(after.status, 200);
}

#[test]
fn concurrent_health_requests_all_succeed() {
    let server = TestServer::start_with("fixtures/model/tiny_text", None, 16, 2_000, 8);
    let addr = server.addr();
    thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(move || {
                let response = http_request(addr, "GET", "/health", "", None);
                assert_eq!(response.status, 200);
            });
        }
    });
}

#[test]
fn audio_embeddings_work_with_the_tiny_audio_fixture() {
    let server = TestServer::start("fixtures/model/tiny_audio", None);
    let addr = server.addr();

    let models = http_request(addr, "GET", "/v1/models", "", None).json();
    assert_eq!(
        models
            .get("modalities")
            .and_then(|value| value.get("audio"))
            .and_then(Value::as_bool),
        Some(true)
    );

    let wav = std::fs::read("fixtures/media/audio/short_tone_100ms.wav").expect("WAV fixture");
    let body = format!(
        r#"{{"input":{{"type":"audio","data":"{}"}},"dimension":8}}"#,
        base64_encode(&wav)
    );
    let response = http_request(addr, "POST", "/v1/embeddings", &body, None);
    assert_eq!(response.status, 200, "body: {:?}", response.json());
    let value = response.json();
    assert_eq!(value.get("modality").and_then(Value::as_str), Some("audio"));
    let embedding = value
        .get("embedding")
        .and_then(Value::as_array)
        .expect("embedding array");
    assert_eq!(embedding.len(), 8);
    assert!((vector_norm(embedding) - 1.0).abs() <= 1e-4);
}

#[test]
fn shutdown_endpoint_stops_the_server() {
    let server = TestServer::start("fixtures/model/tiny_text", None);
    let addr = server.addr();
    let response = http_request(addr, "POST", "/v1/shutdown", "", None);
    assert_eq!(response.status, 202);
    server.running.join().expect("server joins");
    assert!(TcpStream::connect(addr).is_err());
}

#[test]
#[ignore = "loads the frozen model and tokenizer"]
fn text_embedding_over_http_returns_a_normalized_model_vector() {
    let server = TestServer::start_with(
        ".model-cache/f9972ac22e696a105c482202646979e33e4b5a1e",
        None,
        8,
        10_000,
        768,
    );
    let addr = server.addr();
    let response = http_request(
        addr,
        "POST",
        "/v1/embeddings",
        r#"{"input":{"type":"text","text":"hello","prompt":"SearchQuery"},"dimension":768}"#,
        None,
    );
    assert_eq!(response.status, 200, "body: {:?}", response.json());
    let value = response.json();
    let embedding = value
        .get("embedding")
        .and_then(Value::as_array)
        .expect("embedding array");
    assert_eq!(embedding.len(), 768);
    assert!((vector_norm(embedding) - 1.0).abs() <= 1e-4);
}

#[test]
#[ignore = "loads the frozen model and runs the scalar vision tower"]
fn image_embedding_over_http_returns_a_model_vector() {
    let server = TestServer::start_with(
        ".model-cache/f9972ac22e696a105c482202646979e33e4b5a1e",
        None,
        4,
        30_000,
        768,
    );
    let addr = server.addr();
    let png = std::fs::read("fixtures/media/quadrants_16x16.png").expect("PNG fixture");
    let body = format!(
        r#"{{"input":{{"type":"image","data":"{}"}},"dimension":768}}"#,
        base64_encode(&png)
    );
    let response = http_request(addr, "POST", "/v1/embeddings", &body, None);
    assert_eq!(response.status, 200, "body: {:?}", response.json());
    let value = response.json();
    let embedding = value
        .get("embedding")
        .and_then(Value::as_array)
        .expect("embedding array");
    assert_eq!(embedding.len(), 768);
    assert!((vector_norm(embedding) - 1.0).abs() <= 1e-4);
}

#[test]
fn audio_document_ingestion_job_and_audio_query_work() {
    let server = TestServer::start("fixtures/model/tiny_audio", None);
    let addr = server.addr();
    let wav = std::fs::read("fixtures/media/audio/short_tone_100ms.wav").expect("WAV fixture");
    let data = base64_encode(&wav);

    let body =
        format!(r#"{{"id":"audio-1","input":{{"type":"audio","data":"{data}"}},"title":"Tone"}}"#);
    let create = http_request(addr, "POST", "/v1/documents", &body, None);
    assert_eq!(create.status, 202);
    let job_id = create
        .json()
        .get("job")
        .and_then(|job| job.get("id"))
        .and_then(Value::as_number)
        .and_then(|number| number.as_u64())
        .expect("job id");
    let job = wait_job(addr, job_id, None);
    assert_eq!(
        job.get("job")
            .and_then(|job| job.get("status"))
            .and_then(Value::as_str),
        Some("succeeded")
    );

    let query = format!(r#"{{"query":{{"type":"audio","data":"{data}"}},"mode":"vector"}}"#);
    let response = http_request(addr, "POST", "/v1/search", &query, None);
    assert_eq!(response.status, 200, "body: {:?}", response.json());
    let value = response.json();
    let hits = value.get("hits").and_then(Value::as_array).expect("hits");
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].get("document_id").and_then(Value::as_str),
        Some("audio-1")
    );
    assert_eq!(
        hits[0].get("modality").and_then(Value::as_str),
        Some("audio")
    );
}
