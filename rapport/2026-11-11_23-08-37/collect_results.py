#!/usr/bin/env python3
"""Assemble results.json for this E2E benchmark run."""
import json
import pathlib
import re
import statistics

D = pathlib.Path(__file__).resolve().parent


def parse_time_text(text: str) -> dict:
    out = {}
    m = re.search(r"([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+([0-9.]+)\s+sys", text)
    if m:
        out["wall_s"] = float(m.group(1))
        out["user_s"] = float(m.group(2))
        out["sys_s"] = float(m.group(3))
    m = re.search(r"(\d+)\s+maximum resident set size", text)
    if m:
        out["max_rss_bytes"] = int(m.group(1))
    m = re.search(r"(\d+)\s+peak memory footprint", text)
    if m:
        out["peak_footprint_bytes"] = int(m.group(1))
    return out


def parse_time(name: str) -> dict:
    return parse_time_text((D / "raw" / name).read_text(errors="replace"))


def parse_test_result(name: str) -> dict:
    text = (D / "logs" / name).read_text(errors="replace")
    m = re.search(
        r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
        r"(\d+) measured; (\d+) filtered out; finished in ([0-9.]+)s",
        text,
    )
    if not m:
        raise SystemExit(f"could not parse test result from {name}")
    return {
        "status": m.group(1),
        "passed": int(m.group(2)),
        "failed": int(m.group(3)),
        "ignored": int(m.group(4)),
        "measured": int(m.group(5)),
        "filtered": int(m.group(6)),
        "test_seconds": float(m.group(7)),
    }


parsed = json.loads((D / "raw" / "parsed.json").read_text())
harness_report = json.loads((D / "release-e2e-report.json").read_text())

# Text benchmark numbers are printed by tests/text_bench.rs.
def parse_bench_log(name: str) -> dict:
    text = (D / "logs" / name).read_text(errors="replace")
    seq = re.search(
        r"sequential: p50 ([0-9.]+)ms, p95 ([0-9.]+)ms, min ([0-9.]+)ms, max ([0-9.]+)ms",
        text,
    )
    con = re.search(
        r"concurrent: (\d+) requests on (\d+) threads in ([0-9.]+)(ms|s) \(([0-9.]+) requests/s\)",
        text,
    )
    result = re.search(r"test result: ok\. 1 passed;.*finished in ([0-9.]+)s", text)
    if not seq or not con or not result:
        raise SystemExit(f"could not parse the text benchmark output from {name}")
    suffix = name.removeprefix("text-bench").removesuffix(".log").strip("-")
    timing = parse_time_text(text)
    return {
        "sample": int(suffix) if suffix else 1,
        "p50_ms": float(seq.group(1)),
        "p95_ms": float(seq.group(2)),
        "min_ms": float(seq.group(3)),
        "max_ms": float(seq.group(4)),
        "concurrent_requests": int(con.group(1)),
        "threads": int(con.group(2)),
        "concurrent_wall_s": float(con.group(3)) / (1000.0 if con.group(4) == "ms" else 1.0),
        "throughput_rps": float(con.group(5)),
        "test_seconds": float(result.group(1)),
        **timing,
    }


bench_samples = [
    parse_bench_log("text-bench.log"),
    parse_bench_log("text-bench-2.log"),
    parse_bench_log("text-bench-3.log"),
]
median = statistics.median
p50s = [sample["p50_ms"] for sample in bench_samples]
p95s = [sample["p95_ms"] for sample in bench_samples]
throughputs = [sample["throughput_rps"] for sample in bench_samples]
concurrent_walls = [sample["concurrent_wall_s"] for sample in bench_samples]
walls = [sample["wall_s"] for sample in bench_samples]
rss = [sample["max_rss_bytes"] for sample in bench_samples]
test_seconds = [sample["test_seconds"] for sample in bench_samples]
stable = bench_samples[1:]
text_bench = {
    "command": "cargo test --release --test text_bench -- --ignored --nocapture",
    "sample_count": len(bench_samples),
    "samples": bench_samples,
    "p50_ms": median(p50s),
    "p95_ms": median(p95s),
    "min_ms": min(sample["min_ms"] for sample in bench_samples),
    "max_ms": max(sample["max_ms"] for sample in bench_samples),
    "concurrent_requests": bench_samples[0]["concurrent_requests"],
    "threads": bench_samples[0]["threads"],
    "concurrent_wall_s": median(concurrent_walls),
    "throughput_rps": median(throughputs),
    "test_seconds": median(test_seconds),
    "wall_s": median(walls),
    "user_s": median([sample["user_s"] for sample in bench_samples]),
    "sys_s": median([sample["sys_s"] for sample in bench_samples]),
    "max_rss_bytes": median(rss),
    "peak_footprint_bytes": median([sample["peak_footprint_bytes"] for sample in bench_samples]),
    "note": (
        "Median of three release-mode samples. Sample 1 ran immediately after the "
        "fast suite and was an outlier "
        f"({bench_samples[0]['p50_ms']:.1f} ms p50, {bench_samples[0]['throughput_rps']:.2f} req/s); "
        "samples 2-3 are stable at "
        f"{min(s['p50_ms'] for s in stable):.1f}-{max(s['p50_ms'] for s in stable):.1f} ms p50 and "
        f"{min(s['throughput_rps'] for s in stable):.2f}-{max(s['throughput_rps'] for s in stable):.2f} req/s."
    ),
}

# Standalone embedded-binary embeddings.
embeds = {
    "text": {
        "label": "Text",
        "input": "What causes the northern lights?",
        "prompt": "SearchQuery",
        "dimension": 768,
        **parse_time("embed/text.time"),
        "cold_wall_s": parse_time("embed/text-warmup.time")["wall_s"],
    },
    "image70": {
        "label": "Image (70 visual tokens)",
        "input": "fixtures/media/quadrants_16x16.png",
        "vision_tokens": 70,
        "dimension": 768,
        **parse_time("embed/image70.time"),
        "cold_wall_s": parse_time("embed/image70-warmup.time")["wall_s"],
    },
    "audio": {
        "label": "Audio (100 ms tone)",
        "input": "fixtures/media/audio/short_tone_100ms.wav",
        "dimension": 768,
        **parse_time("embed/audio.time"),
        "cold_wall_s": parse_time("embed/audio-warmup.time")["wall_s"],
    },
}

# Frozen-model parity replay.
parity = {
    "text": {
        "label": "Text parity",
        "command": "cargo test --release --test text_embeddings real_model_parity -- --nocapture",
        **parse_test_result("parity-text.log"),
        **parse_time("parity-text.time"),
    },
    "image": {
        "label": "Image parity (70 visual tokens)",
        "command": "cargo test --release --test image_embeddings image_embedding_matches_the_official_reference -- --nocapture",
        **parse_test_result("parity-image.log"),
        **parse_time("parity-image.time"),
    },
    "audio": {
        "label": "Audio parity (short tone)",
        "command": "cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture",
        **parse_test_result("parity-audio.log"),
        **parse_time("parity-audio.time"),
    },
}

# The harness phases are fixed; assert that each marker appears in the log.
harness_log = (D / "logs" / "release-e2e.log").read_text(errors="replace")
phase_markers = [
    ("Artifact checks", "release-e2e: artifact checks"),
    ("Embedded model probe", "release-e2e: embedded model probe passed"),
    ("Text embedding", "release-e2e: text embedding"),
    ("Image embedding", "release-e2e: image embedding"),
    ("Audio embedding", "release-e2e: audio embedding"),
    ("Index text, image, and audio", "release-e2e: indexing text, image, and audio documents"),
    ("Lexical, vector, and hybrid search", "release-e2e: lexical, vector, and hybrid search"),
    ("Verify and compact", "release-e2e: verification and compaction"),
    ("Restart after compaction", "release-e2e: restart after compaction preserved all documents"),
    ("HTTP API lifecycle", "release-e2e: HTTP API lifecycle"),
    ("HTTP restart persistence", "release-e2e: HTTP restart preserved all documents"),
    ("Report and PASSED", "release-e2e: PASSED"),
]
phases = []
for label, marker in phase_markers:
    if marker not in harness_log:
        raise SystemExit(f"missing harness phase marker: {marker}")
    phases.append({"phase": label, "status": "passed"})

tests = {
    "command": "SELDON_SKIP_MODEL_PARITY=1 cargo test",
    "note": "The environment variable makes the real-model parity tests return early in the fast suite. Full parity was replayed separately in release mode.",
    "wall_s": parsed["cargo_test_fast"]["real_s"],
    "user_s": parsed["cargo_test_fast"]["user_s"],
    "sys_s": parsed["cargo_test_fast"]["sys_s"],
    "max_rss_bytes": parsed["cargo_test_fast"]["max_rss_bytes"],
    "peak_footprint_bytes": parsed["cargo_test_fast"]["peak_footprint_bytes"],
    "passed": parsed["totals"]["passed"],
    "failed": parsed["totals"]["failed"],
    "ignored": parsed["totals"]["ignored"],
    "total": parsed["totals"]["passed"] + parsed["totals"]["failed"] + parsed["totals"]["ignored"],
    "sum_suite_seconds": parsed["totals"]["seconds"],
    "suites": parsed["suites"],
}

release_e2e = {
    "command": "scripts/release_e2e.sh --binary target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite --report <absolute path>",
    **parse_time("release-e2e.time"),
    **harness_report,
    "phases": phases,
    "attempts": 2,
    "attempt_note": "Attempt 1 completed all lifecycle checks but exited 1 at the final report copy because the report path was relative and the harness had changed into its staging directory. Attempt 2 used an absolute report path and passed.",
}

historical = {
    "m2_text_p50_ms": 1485,
    "m2_text_p95_ms": 1504,
    "m2_throughput_rps": 2.43,
    "m7_text_p50_ms": 159,
    "m7_text_p95_ms": 159,
    "m7_throughput_rps": 24.47,
    "m7_text_parity_s": 12.98,
    "m7_image_parity_s": 22.02,
    "m7_audio_parity_s": 15.03,
    "m8_binary_sha256": "b2e697b78602d8eaf82fc170b8d92056ff0088a38bcb77bbfd6f648cf52902e8",
    "m8_binary_bytes": 1539215600,
    "m8_report_timestamp": "2026-11-11T21:49:09Z",
}

derived = {
    "tests_pass_rate_pct": 100.0 * tests["passed"] / (tests["passed"] + tests["failed"]),
    "text_speedup_vs_m2": historical["m2_text_p50_ms"] / text_bench["p50_ms"],
    "text_p50_delta_vs_m7_pct": 100.0 * (text_bench["p50_ms"] - historical["m7_text_p50_ms"]) / historical["m7_text_p50_ms"],
    "text_p95_delta_vs_m7_pct": 100.0 * (text_bench["p95_ms"] - historical["m7_text_p95_ms"]) / historical["m7_text_p95_ms"],
    "throughput_ratio_vs_m2": text_bench["throughput_rps"] / historical["m2_throughput_rps"],
    "throughput_ratio_vs_m7": text_bench["throughput_rps"] / historical["m7_throughput_rps"],
    "image_delta_vs_m7_pct": 100.0 * (embeds["image70"]["wall_s"] - historical["m7_image_parity_s"]) / historical["m7_image_parity_s"],
    "audio_delta_vs_m7_pct": 100.0 * (embeds["audio"]["wall_s"] - historical["m7_audio_parity_s"]) / historical["m7_audio_parity_s"],
    "binary_size_gib": 1539215600 / (1024 ** 3),
    "binary_size_gb": 1539215600 / 1_000_000_000,
}

results = {
    "schema_version": 1,
    "run": {
        "id": D.name,
        "started_local": "2026-11-11T23:08:37+01:00",
        "started_utc": "2026-11-11T22:08:37Z",
        "finished_local": "2026-11-11T23:35:16+01:00",
        "timezone": "Europe/Paris",
        "git_commit": "45da9c794b805bec2c32bb988dacdfda25817bb1",
        "git_short": "45da9c7",
        "git_branch": "main",
        "git_status": "clean except the new rapport/ directory",
    },
    "environment": {
        "cpu": "Apple M2 Pro",
        "cores": 10,
        "performance_cores": 6,
        "efficiency_cores": 4,
        "memory_bytes": 17179869184,
        "os": "macOS 26.5.1 (25F80)",
        "kernel": "Darwin 25.5.0",
        "arch": "arm64",
        "target": "aarch64-apple-darwin",
        "rustc": "1.99.1 (58cf8f9c3 2026-10-13)",
        "cargo": "1.99.1 (5f94df478 2026-08-27)",
        "release_profile": "opt-level=3, thin LTO, codegen-units=1",
        "test_profile": "opt-level=2",
    },
    "model": {
        "id": "google/embeddinggemma-2",
        "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        "safetensors_bytes": 1488915288,
        "bundle_bytes": 1525784026,
    },
    "artifact": {
        "path": "target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite",
        "sha256": "b2e697b78602d8eaf82fc170b8d92056ff0088a38bcb77bbfd6f648cf52902e8",
        "bytes": 1539215600,
        "linked_libraries": ["/usr/lib/libSystem.B.dylib"],
        "minos": "11.0",
        "signature": "adhoc",
        "gatekeeper": "rejected",
        "cargo_graph": "root crate only",
    },
    "tests": tests,
    "text_bench": text_bench,
    "embeds": embeds,
    "parity": parity,
    "release_e2e": release_e2e,
    "historical": historical,
    "derived": derived,
}

(D / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print("wrote", D / "results.json")
print(json.dumps({"tests": {"passed": tests["passed"], "failed": tests["failed"], "ignored": tests["ignored"]}, "text_p50_ms": text_bench["p50_ms"], "throughput_rps": text_bench["throughput_rps"], "release_e2e_wall_s": release_e2e["wall_s"], "parity": {k: v["test_seconds"] for k, v in parity.items()}}, indent=2))
