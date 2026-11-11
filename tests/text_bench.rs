//! Manual text-embedding benchmark for milestone M2.
//!
//! The test is ignored by default because it loads the 1.5 GB frozen model.
//! Run it with:
//!
//! ```bash
//! cargo test --release --test text_bench -- --ignored --nocapture
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use seldon_vault_lite::model::prompt::Prompt;
use seldon_vault_lite::model::text::TextEmbedder;

const REVISION: &str = "f9972ac22e696a105c482202646979e33e4b5a1e";

#[test]
#[ignore = "loads the frozen model; run explicitly with --release --ignored"]
fn text_embedding_latency_and_throughput() {
    let model_dir = Path::new(".model-cache").join(REVISION);
    if !model_dir.join("model.safetensors").is_file() {
        eprintln!(
            "skipping benchmark: {} is not present",
            model_dir.join("model.safetensors").display()
        );
        return;
    }

    let embedder = Arc::new(TextEmbedder::load(&model_dir).expect("load the frozen embedder"));
    let text = "What causes the northern lights?";
    let prompt = Prompt::SearchQuery;

    let warmup = embedder.embed(text, prompt, 768).expect("warmup embedding");
    assert_eq!(warmup.len(), 768);
    let norm: f32 = warmup.iter().map(|value| value * value).sum();
    assert!((norm - 1.0).abs() <= 1e-4);

    let mut durations = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        let embedding = embedder.embed(text, prompt, 768).expect("embedding");
        durations.push(start.elapsed());
        assert_eq!(embedding.len(), 768);
    }
    durations.sort();
    let p50 = durations[durations.len() / 2];
    let p95 = durations[durations.len() * 95 / 100];
    println!(
        "sequential: p50 {:?}, p95 {:?}, min {:?}, max {:?}",
        p50,
        p95,
        durations[0],
        durations[durations.len() - 1]
    );

    let workers = 4;
    let per_worker = 5;
    let total = workers * per_worker;
    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let embedder = Arc::clone(&embedder);
            scope.spawn(move || {
                for _ in 0..per_worker {
                    let embedding = embedder.embed(text, prompt, 768).expect("embedding");
                    assert_eq!(embedding.len(), 768);
                }
            });
        }
    });
    let elapsed = start.elapsed();
    let throughput = total as f64 / elapsed.as_secs_f64();
    println!(
        "concurrent: {total} requests on {workers} threads in {elapsed:?} ({throughput:.2} requests/s)"
    );
}
