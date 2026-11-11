#!/usr/bin/env python3
"""Generate benchmark.md and rapport.html from results.json."""
from __future__ import annotations

import html
import json
import pathlib

D = pathlib.Path(__file__).resolve().parent
R = json.loads((D / "results.json").read_text())

BT = chr(96)
FENCE = BT * 3


def code(text: str) -> str:
    return BT + text + BT


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_ms(value: float) -> str:
    return f"{value:,.1f} ms"


def fmt_s(value: float) -> str:
    return f"{value:,.2f} s"


def fmt_bytes(value: int) -> str:
    gib = value / (1024 ** 3)
    mib = value / (1024 ** 2)
    if gib >= 1:
        return f"{gib:.2f} GiB ({fmt_int(value)} B)"
    return f"{mib:.1f} MiB ({fmt_int(value)} B)"


def fmt_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def fmt_x(value: float) -> str:
    return f"{value:.2f}x"


def md_badge(status: str) -> str:
    return {"ok": "PASS", "passed": "PASS", "failed": "FAIL"}.get(status, status)


def html_badge(status: str, label: str | None = None) -> str:
    text = label or ("PASS" if status in ("ok", "passed") else "FAIL")
    cls = "ok" if status in ("ok", "passed") else "bad"
    return f'<span class="badge {cls}">{esc(text)}</span>'


def html_card(label: str, value: str, sub: str, delta: str | None = None, tone: str = "") -> str:
    delta_html = ""
    if delta:
        delta_html = f'<div class="card-delta {tone}">{esc(delta)}</div>'
    return f"""
      <div class="card">
        <div class="card-label">{esc(label)}</div>
        <div class="card-value">{esc(value)}</div>
        <div class="card-sub">{esc(sub)}</div>
        {delta_html}
      </div>"""


def html_bar_chart(title: str, rows: list[tuple[str, float, str, str]], unit: str) -> str:
    max_value = max(row[1] for row in rows) or 1.0
    bars = []
    for label, value, display, cls in rows:
        width = max(1.5, 100.0 * value / max_value)
        bars.append(
            f"""
        <div class="bar-row">
          <div class="bar-label">{esc(label)}</div>
          <div class="bar-track"><div class="bar-fill {esc(cls)}" style="width:{width:.1f}%"></div></div>
          <div class="bar-value">{esc(display)}</div>
        </div>"""
        )
    return f"""
      <div class="chart">
        <div class="chart-title">{esc(title)} <span class="chart-unit">{esc(unit)}</span></div>
        <div class="bars">{''.join(bars)}</div>
      </div>"""


SUITE_NAMES = {
    "src/lib.rs": "Unit tests (lib)",
    "src/main.rs": "Unit tests (bin)",
    "tests/audio.rs": "Audio decoder",
    "tests/audio_embeddings.rs": "Audio embeddings",
    "tests/audio_tiny.rs": "Tiny audio fixture",
    "tests/bundle_loading.rs": "Bundle loading",
    "tests/cli.rs": "CLI integration",
    "tests/formats.rs": "Formats / JSON / tensors",
    "tests/http_api.rs": "HTTP API",
    "tests/image_embeddings.rs": "Image embeddings",
    "tests/media.rs": "Media decoders",
    "tests/preprocess.rs": "Image preprocessing",
    "tests/release_e2e.rs": "Release E2E lifecycle",
    "tests/text_bench.rs": "Text benchmark",
    "tests/text_embeddings.rs": "Text embeddings",
    "tests/tokenizer.rs": "Tokenizer",
}

suites = [
    suite
    for suite in R["tests"]["suites"]
    if suite["passed"] + suite["failed"] + suite["ignored"] > 0
]
for suite in suites:
    suite["friendly"] = SUITE_NAMES.get(suite["label"], suite["label"])

bench = R["text_bench"]
tests = R["tests"]
e2e = R["release_e2e"]
derived = R["derived"]
hist = R["historical"]
run = R["run"]
env = R["environment"]
artifact = R["artifact"]
model = R["model"]

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

md_suites = []
for suite in suites:
    md_suites.append(
        f"| {suite['friendly']} | {suite['passed']} | {suite['failed']} | "
        f"{suite['ignored']} | {suite['seconds']:.2f} s |"
    )

md_phases = []
for index, phase in enumerate(e2e["phases"], start=1):
    md_phases.append(f"| {index} | {phase['phase']} | {md_badge(phase['status'])} |")

md_parity = []
for key in ("text", "image", "audio"):
    item = R["parity"][key]
    md_parity.append(
        f"| {item['label']} | {md_badge(item['status'])} | {item['test_seconds']:.2f} s | "
        f"{item['wall_s']:.2f} s | {fmt_gib(item['max_rss_bytes'])} |"
    )

md_embeds = []
for key in ("text", "image70", "audio"):
    item = R["embeds"][key]
    md_embeds.append(
        f"| {item['label']} | {code(item['input'])} | {item['wall_s']:.2f} s | "
        f"{item['cold_wall_s']:.2f} s | {fmt_gib(item['max_rss_bytes'])} |"
    )

md_samples = []
for sample in bench["samples"]:
    md_samples.append(
        f"| {sample['sample']} | {fmt_ms(sample['p50_ms'])} | {fmt_ms(sample['p95_ms'])} | "
        f"{fmt_ms(sample['min_ms'])} | {fmt_ms(sample['max_ms'])} | "
        f"{sample['throughput_rps']:.2f} req/s | {sample['wall_s']:.2f} s | "
        f"{fmt_gib(sample['max_rss_bytes'])} |"
    )
md_samples.append(
    f"| **Median** | **{fmt_ms(bench['p50_ms'])}** | **{fmt_ms(bench['p95_ms'])}** | "
    f"{fmt_ms(bench['min_ms'])} | {fmt_ms(bench['max_ms'])} | "
    f"**{bench['throughput_rps']:.2f} req/s** | {bench['wall_s']:.2f} s | "
    f"{fmt_gib(bench['max_rss_bytes'])} |"
)

markdown = f"""# Seldon Vault Lite — E2E benchmark

**Run ID:** {code(run['id'])}  
**Started:** 2026-11-11 23:08:37 +01:00 (Europe/Paris) · {run['started_utc']}  
**Finished:** 2026-11-11 23:35:16 +01:00  
**Status:** PASSED — {tests['passed']} tests passed, {tests['failed']} failed, full release E2E passed.

> Reproducible snapshot for comparing future improvements. The machine-readable
> source is {code('results.json')}; raw logs are in {code('logs/')} and {code('raw/')}.
> This report was generated by {code('generate_report.py')} from the recorded data.

## At a glance

| Area | Result |
|---|---|
| Fast test suite | **{tests['passed']} passed / {tests['failed']} failed / {tests['ignored']} ignored** ({tests['total']} total) |
| Fast suite wall time | **{fmt_s(tests['wall_s'])}** |
| Release E2E harness | **PASSED** in **{fmt_s(e2e['wall_s'])}** (12/12 phases) |
| Text p50 / p95 | **{fmt_ms(bench['p50_ms'])} / {fmt_ms(bench['p95_ms'])}** (median of 3 samples) |
| Concurrent throughput | **{bench['throughput_rps']:.2f} req/s** ({bench['threads']} threads, {bench['concurrent_requests']} requests; median of 3) |
| Standalone embed (warm) | text **{fmt_s(R['embeds']['text']['wall_s'])}** · image 70 tok **{fmt_s(R['embeds']['image70']['wall_s'])}** · audio **{fmt_s(R['embeds']['audio']['wall_s'])}** |
| Peak RSS | text **{fmt_gib(R['embeds']['text']['max_rss_bytes'])}** · image **{fmt_gib(R['embeds']['image70']['max_rss_bytes'])}** · audio **{fmt_gib(R['embeds']['audio']['max_rss_bytes'])}** · harness **{fmt_gib(e2e['max_rss_bytes'])}** |
| Artifact | **{derived['binary_size_gib']:.2f} GiB** · SHA-256 {code(artifact['sha256'][:12] + '…')} |
| Frozen-model parity | text **{R['parity']['text']['test_seconds']:.2f} s** · image **{R['parity']['image']['test_seconds']:.2f} s** · audio **{R['parity']['audio']['test_seconds']:.2f} s** — all pass |

## Run fingerprint

| Field | Value |
|---|---|
| Run ID | {code(run['id'])} |
| Started | 2026-11-11 23:08:37 +01:00 (Europe/Paris) |
| Finished | 2026-11-11 23:35:16 +01:00 |
| Git commit | {code(run['git_commit'])} ({code(run['git_branch'])}) |
| Git status | {run['git_status']} |
| Model | {code(model['id'])} @ {code(model['revision'])} |
| Model weights | Safetensors {fmt_int(model['safetensors_bytes'])} B; {code('.svlm')} bundle {fmt_int(model['bundle_bytes'])} B |
| Target | {code(env['target'])} |
| Release profile | {env['release_profile']} |
| Test profile | {env['test_profile']} |

## Environment

| Item | Value |
|---|---|
| CPU | {env['cpu']}, {env['cores']} cores ({env['performance_cores']} performance + {env['efficiency_cores']} efficiency) |
| Memory | {env['memory_bytes'] // (1024 ** 3)} GiB |
| OS | {env['os']}, {env['kernel']} |
| Architecture | {env['arch']} |
| Rust | rustc {env['rustc']} |
| Cargo | {env['cargo']} |
| Cargo graph | root crate only (zero third-party crates) |

## Artifact

| Property | Value |
|---|---|
| Path | {code(artifact['path'])} |
| Size | {fmt_int(artifact['bytes'])} bytes ({derived['binary_size_gib']:.2f} GiB / {derived['binary_size_gb']:.2f} GB) |
| SHA-256 | {code(artifact['sha256'])} |
| Linked libraries | {code(artifact['linked_libraries'][0])} only |
| Deployment target | {code('minos ' + artifact['minos'])} |
| Code signature | {artifact['signature']} (linker-signed) |
| Gatekeeper | {artifact['gatekeeper']} — expected for an ad-hoc local-use build |
| Model source | embedded {code('.svlm')} bundle (no model cache at runtime) |
| Version | {code('seldon-vault-lite 0.1.0 (' + env['target'] + ')')} |

## Test suite

Command:

{FENCE}bash
{tests['command']}
{FENCE}

Result: **{tests['passed']} passed, {tests['failed']} failed, {tests['ignored']} ignored** ({tests['total']} total) in **{fmt_s(tests['wall_s'])}** wall time
({tests['sum_suite_seconds']:.2f} s summed test-binary time; cargo/build overhead accounts for the rest).
Max RSS {fmt_bytes(tests['max_rss_bytes'])}.

> {code('SELDON_SKIP_MODEL_PARITY=1')} makes the real-model parity tests return early.
> Full parity was replayed separately in release mode (see below).

### Suite breakdown

| Suite | Passed | Failed | Ignored | Duration |
|---|---:|---:|---:|---:|
{chr(10).join(md_suites)}
| **Total** | **{tests['passed']}** | **{tests['failed']}** | **{tests['ignored']}** | **{tests['sum_suite_seconds']:.2f} s** |

## Release E2E harness

Command:

{FENCE}bash
scripts/release_e2e.sh \\
  --binary target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite \\
  --report "$PWD/rapport/{run['id']}/release-e2e-report.json"
{FENCE}

Result: **PASSED** in **{fmt_s(e2e['wall_s'])}** (max RSS {fmt_bytes(e2e['max_rss_bytes'])}).
The harness ran with {code('env -i')}, a temporary HOME/TMPDIR, a system-only PATH,
and a sanitized staging directory. It exercised the embedded model, text/image/
audio embeddings, ingestion, lexical/vector/hybrid search, verify, compact,
restart, and the loopback HTTP API.

| Report field | Value |
|---|---|
| Status | **{e2e['status']}** |
| Timestamp | {e2e['timestamp']} |
| Platform | {e2e['platform']} |
| OS | {e2e['os']} |
| Version | {code(e2e['version'])} |
| Dimension | {e2e['dimension']} |
| Documents | {e2e['documents']} |
| Signing | {e2e['signing']} |
| Gatekeeper | {e2e['gatekeeper']} |
| Network isolation | {e2e['network_isolation']} ({code('sandbox-exec')} unavailable) |
| Binary SHA-256 | {code(e2e['binary_sha256'])} |

### Lifecycle phases

| # | Phase | Status |
|---:|---|---|
{chr(10).join(md_phases)}

## Performance

All measurements use the release profile on the reference Apple M2 Pro.
The text benchmark uses {code('std::time::Instant')}; the standalone embed timings use
{code('/usr/bin/time -l')} around the embedded binary. Each standalone run includes
loading the embedded model and producing a 768-dimensional vector.

### Text latency and throughput

| Metric | Value |
|---|---:|
| p50 | {fmt_ms(bench['p50_ms'])} |
| p95 | {fmt_ms(bench['p95_ms'])} |
| min | {fmt_ms(bench['min_ms'])} |
| max | {fmt_ms(bench['max_ms'])} |
| Concurrent requests | {bench['concurrent_requests']} on {bench['threads']} threads |
| Concurrent wall time | {bench['concurrent_wall_s']:.4f} s |
| Throughput | {bench['throughput_rps']:.2f} req/s |
| Test time | {bench['test_seconds']:.2f} s |
| Process wall time | {bench['wall_s']:.2f} s (median; sample 1 includes 2.93 s compile) |
| Max RSS | {fmt_bytes(bench['max_rss_bytes'])} |

The text benchmark embeds {code('What causes the northern lights?')} with the
{code('SearchQuery')} prompt at 768 dimensions. Ten sequential calls after one warm-up
produce the latency figures; four threads × five requests produce throughput.
The table above uses the median of three release-mode samples.

### Text benchmark samples

| Sample | p50 | p95 | min | max | Throughput | Process wall | Max RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(md_samples)}

{bench['note']} Sample 1 process wall includes 2.93 s of compilation; samples 2-3 are warm builds.

### Standalone embedded embeddings

| Modality | Input | Warm wall | Cold wall | Max RSS |
|---|---|---:|---:|---:|
{chr(10).join(md_embeds)}

Warm = second run in the same session; cold = first run. The image and audio
towers remain scalar correctness baselines. The full harness also exercises the
default 280-token image budget, which is heavier than the 70-token measurement.

### Frozen-model parity replay

| Test | Status | Test time | Process wall | Max RSS |
|---|---|---:|---:|---:|
{chr(10).join(md_parity)}

Commands:

{FENCE}bash
cargo test --release --test text_embeddings real_model_parity -- --nocapture
cargo test --release --test image_embeddings image_embedding_matches_the_official_reference -- --nocapture
cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture
{FENCE}

## Comparison with documented baselines

The M2 scalar, M7 NEON, and M8 release numbers are taken from
{code('docs/benchmarks.md')}; they were not re-measured in this run. The current values
are this run's measurements.

| Metric | M2 scalar | M7 NEON | Current | Δ vs M7 |
|---|---:|---:|---:|---:|
| Text p50 | 1,485 ms | 159 ms | {fmt_ms(bench['p50_ms'])} | {fmt_pct(derived['text_p50_delta_vs_m7_pct'])} |
| Text p95 | 1,504 ms | 159 ms | {fmt_ms(bench['p95_ms'])} | {fmt_pct(derived['text_p95_delta_vs_m7_pct'])} |
| Throughput (4T/20) | 2.43 req/s | 24.47 req/s | {bench['throughput_rps']:.2f} req/s | {fmt_pct((derived['throughput_ratio_vs_m7'] - 1) * 100)} |
| Text parity | — | 12.98 s | {R['parity']['text']['test_seconds']:.2f} s | {fmt_pct((R['parity']['text']['test_seconds'] / hist['m7_text_parity_s'] - 1) * 100)} |
| Image parity (70 tok) | — | 22.02 s | {R['parity']['image']['test_seconds']:.2f} s | {fmt_pct((R['parity']['image']['test_seconds'] / hist['m7_image_parity_s'] - 1) * 100)} |
| Audio parity (short) | — | 15.03 s | {R['parity']['audio']['test_seconds']:.2f} s | {fmt_pct((R['parity']['audio']['test_seconds'] / hist['m7_audio_parity_s'] - 1) * 100)} |
| Binary size | — | — | {derived['binary_size_gib']:.2f} GiB | = M8 |
| Documents after restart | — | 3 | {e2e['documents']} | = |

Current text speedup vs M2 scalar: **{fmt_x(derived['text_speedup_vs_m2'])}**
(1,485 / {bench['p50_ms']:.1f}). Current throughput is
**{fmt_x(derived['throughput_ratio_vs_m2'])}** the M2 scalar rate. The text p50
is {fmt_pct(derived['text_p50_delta_vs_m7_pct'])} and throughput is
{fmt_pct((derived['throughput_ratio_vs_m7'] - 1) * 100)} relative to the
documented M7 baseline. {bench['note']}
Treat this as the new baseline for future comparisons rather than a regression
claim, because run-to-run variance, thermal state, and background load were not
controlled across the historical runs.

### Notes on comparability

- Text p50/p95/throughput use the same {code('text_bench')} and input as the M2/M7
  baselines, so they are directly comparable.
- The p95 sample is sensitive to a single outlier: sample 3 recorded 182.4 ms
  while sample 2 recorded 154.9 ms. Use p50 and throughput as the stable
  baseline; p95 is reported for completeness.
- Parity wall times use the same release-mode test commands as the M7 parity
  replay, so they are directly comparable.
- The standalone image/audio embed timings are not the same measurement as the
  parity test wall times; the parity tests include test setup and assertions.
  They are reported separately and should not be substituted into the parity
  columns.
- The release harness total includes a default 280-token image embedding, which
  is not the 70-token benchmark above.

## Notes and caveats

- **Fast suite parity skip:** {code('SELDON_SKIP_MODEL_PARITY=1')} makes the real-model
  parity tests return early in the fast suite. They are re-run separately in
  release mode and all pass.
- **Network isolation:** the harness records {code('network_isolation: disabled')}
  because {code('sandbox-exec')} was unavailable in this environment. Do not claim an
  isolated offline run from this report.
- **Gatekeeper:** the ad-hoc build is rejected by {code('spctl')}; this is expected for
  the local-use artifact. Local execution works without a developer account.
- **Attempt 1:** the first harness attempt completed every lifecycle check but
  exited 1 at the final report copy because the {code('--report')} path was relative and
  the harness had changed into its staging directory. Attempt 2 used an absolute
  path and passed. The attempt 1 log is retained as
  {code('logs/release-e2e-attempt1.log')}.
- **Memory:** the harness max RSS is higher than individual embeddings because
  the HTTP server holds text, image, and audio weights in one process.
- **Image/audio performance:** the towers remain scalar correctness baselines;
  the 70-token image and 100 ms audio measurements are not optimized results.

## Reproduction

{FENCE}bash
cd /Users/fouadassani/projets/seldon-vault-lite-rust

# Fast suite (includes the release E2E lifecycle test)
SELDON_SKIP_MODEL_PARITY=1 cargo test

# Release-mode text benchmark
cargo test --release --test text_bench -- --ignored --nocapture

# Standalone embedded embeddings
BIN=target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite
"$BIN" embed --text "What causes the northern lights?" --dim 768
"$BIN" embed --image fixtures/media/quadrants_16x16.png --dim 768 --vision-tokens 70
"$BIN" embed --audio fixtures/media/audio/short_tone_100ms.wav --dim 768

# Full release E2E harness (absolute --report path required)
scripts/release_e2e.sh \\
  --binary "$BIN" \\
  --report "$PWD/rapport/{run['id']}/release-e2e-report.json"

# Frozen-model parity replay
cargo test --release --test text_embeddings real_model_parity -- --nocapture
cargo test --release --test image_embeddings image_embedding_matches_the_official_reference -- --nocapture
cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture
{FENCE}

## Report files

| File | Contents |
|---|---|
| {code('benchmark.md')} | This report |
| {code('rapport.html')} | Self-contained visual report |
| {code('results.json')} | Machine-readable results for comparison |
| {code('release-e2e-report.json')} | Harness JSON report |
| {code('logs/')} | Raw test, benchmark, parity, and harness logs |
| {code('raw/')} | Timing files, parsed timings, embedding outputs |
| {code('collect_results.py')} | Assembles {code('results.json')} from logs |
| {code('generate_report.py')} | Generates {code('benchmark.md')} and {code('rapport.html')} |
"""

(D / "benchmark.md").write_text(markdown)

# ---------------------------------------------------------------------------
# HTML replacements
# ---------------------------------------------------------------------------

cards = "".join([
    html_card("Tests", f"{tests['passed']} / {tests['total']}", f"{tests['failed']} failed · {tests['ignored']} ignored", "100% pass rate", "good"),
    html_card("Release E2E", "PASSED", f"{e2e['wall_s']:.2f} s wall · 12/12 phases", "embedded artifact", "good"),
    html_card("Text p50", fmt_ms(bench["p50_ms"]), f"p95 {fmt_ms(bench['p95_ms'])} · median of 3", f"{derived['text_p50_delta_vs_m7_pct']:+.1f}% vs M7", "good"),
    html_card("Throughput", f"{bench['throughput_rps']:.2f} req/s", f"{bench['threads']} threads · {bench['concurrent_requests']} requests · median of 3", f"{derived['throughput_ratio_vs_m2']:.2f}x vs M2", "good"),
    html_card("Text speedup", fmt_x(derived["text_speedup_vs_m2"]), "vs M2 scalar 1,485 ms", "NEON path", "good"),
    html_card("Artifact", f"{derived['binary_size_gib']:.2f} GiB", f"SHA-256 {artifact['sha256'][:12]}…", "embedded model", "neutral"),
])

fingerprint = "".join([
    f'<div class="kv-item"><div class="k">Run ID</div><div class="v mono">{esc(run["id"])}</div></div>',
    '<div class="kv-item"><div class="k">Started</div><div class="v">2026-11-11 23:08:37 +01:00</div></div>',
    '<div class="kv-item"><div class="k">Finished</div><div class="v">2026-11-11 23:35:16 +01:00</div></div>',
    f'<div class="kv-item"><div class="k">Git commit</div><div class="v mono">{esc(run["git_short"])} · {esc(run["git_branch"])}</div></div>',
    f'<div class="kv-item"><div class="k">Model</div><div class="v mono">{esc(model["id"])} @ {esc(model["revision"][:12])}…</div></div>',
    f'<div class="kv-item"><div class="k">Target</div><div class="v mono">{esc(env["target"])}</div></div>',
    f'<div class="kv-item"><div class="k">Rust</div><div class="v">{esc(env["rustc"])}</div></div>',
    '<div class="kv-item"><div class="k">Cargo graph</div><div class="v">root crate only · zero third-party crates</div></div>',
])

env_rows = "".join([
    f'<tr><th>CPU</th><td>{esc(env["cpu"])}, {env["cores"]} cores ({env["performance_cores"]}P + {env["efficiency_cores"]}E)</td></tr>',
    f'<tr><th>Memory</th><td>{env["memory_bytes"] // (1024 ** 3)} GiB</td></tr>',
    f'<tr><th>OS</th><td>{esc(env["os"])} · {esc(env["kernel"])}</td></tr>',
    f'<tr><th>Architecture</th><td>{esc(env["arch"])}</td></tr>',
    f'<tr><th>Release profile</th><td>{esc(env["release_profile"])}</td></tr>',
    f'<tr><th>Test profile</th><td>{esc(env["test_profile"])}</td></tr>',
])

artifact_rows = "".join([
    f'<tr><th>Size</th><td>{fmt_int(artifact["bytes"])} B · {derived["binary_size_gib"]:.2f} GiB</td></tr>',
    f'<tr><th>SHA-256</th><td class="mono">{esc(artifact["sha256"][:24])}…</td></tr>',
    f'<tr><th>Linked libs</th><td class="mono">{esc(artifact["linked_libraries"][0])}</td></tr>',
    f'<tr><th>Deployment</th><td class="mono">minos {esc(artifact["minos"])}</td></tr>',
    f'<tr><th>Signature</th><td>{esc(artifact["signature"])} (linker-signed)</td></tr>',
    f'<tr><th>Gatekeeper</th><td>{esc(artifact["gatekeeper"])} — expected for ad-hoc local use</td></tr>',
    '<tr><th>Model source</th><td>embedded <code>.svlm</code> bundle</td></tr>',
])

suite_rows = []
for suite in suites:
    ignored = f'<span class="badge warn">{suite["ignored"]} ignored</span>' if suite["ignored"] else '<span class="badge neutral">0</span>'
    suite_rows.append(f"""
        <tr>
          <td>{esc(suite['friendly'])}</td>
          <td class="num">{suite['passed']}</td>
          <td class="num">{suite['failed']}</td>
          <td>{ignored}</td>
          <td class="num">{suite['seconds']:.2f} s</td>
          <td>{html_badge(suite['result'])}</td>
        </tr>""")

test_total_row = f"""
          <tr><th>Total</th><th class="num">{tests['passed']}</th><th class="num">{tests['failed']}</th><th>{tests['ignored']}</th><th class="num">{tests['sum_suite_seconds']:.2f} s</th><th>{html_badge('passed')}</th></tr>"""

phases = "".join(
    f"""
        <div class="timeline-item">
          <div class="phase">{esc(phase['phase'])}</div>
          <div class="status">{html_badge(phase['status'])} completed</div>
        </div>"""
    for phase in e2e["phases"]
)

artifact_checks = "".join([
    f'<tr><th>Linked libraries</th><td class="mono">{esc(artifact["linked_libraries"][0])} only</td></tr>',
    f'<tr><th>Deployment target</th><td class="mono">minos {esc(artifact["minos"])}</td></tr>',
    f'<tr><th>Signature</th><td>{esc(e2e["signing"])}</td></tr>',
    f'<tr><th>Gatekeeper</th><td>{esc(e2e["gatekeeper"])} (expected)</td></tr>',
    '<tr><th>Cargo graph</th><td>root crate only</td></tr>',
    f'<tr><th>Version</th><td class="mono">{esc(e2e["version"])}</td></tr>',
    f'<tr><th>Dimension</th><td>{e2e["dimension"]}</td></tr>',
])

text_rows = "".join([
    f'<tr><th>p50</th><td class="num">{fmt_ms(bench["p50_ms"])}</td></tr>',
    f'<tr><th>p95</th><td class="num">{fmt_ms(bench["p95_ms"])}</td></tr>',
    f'<tr><th>min / max</th><td class="num">{fmt_ms(bench["min_ms"])} / {fmt_ms(bench["max_ms"])}</td></tr>',
    f'<tr><th>Concurrent</th><td class="num">{bench["concurrent_requests"]} on {bench["threads"]} threads · {bench["concurrent_wall_s"]:.4f} s</td></tr>',
    f'<tr><th>Throughput</th><td class="num"><strong>{bench["throughput_rps"]:.2f} req/s</strong></td></tr>',
    f'<tr><th>Process wall</th><td class="num">{bench["wall_s"]:.2f} s (median; sample 1 includes 2.93 s compile)</td></tr>',
    f'<tr><th>Max RSS</th><td class="num">{fmt_gib(bench["max_rss_bytes"])}</td></tr>',
])

text_samples_rows = "".join(
    f"""
        <tr>
          <td>{sample['sample']}</td>
          <td class="num">{fmt_ms(sample['p50_ms'])}</td>
          <td class="num">{fmt_ms(sample['p95_ms'])}</td>
          <td class="num">{fmt_ms(sample['min_ms'])}</td>
          <td class="num">{fmt_ms(sample['max_ms'])}</td>
          <td class="num">{sample['throughput_rps']:.2f} req/s</td>
          <td class="num">{sample['wall_s']:.2f} s</td>
          <td class="num">{fmt_gib(sample['max_rss_bytes'])}</td>
        </tr>"""
    for sample in bench["samples"]
)
text_samples_rows += f"""
        <tr><th>Median</th><th class="num">{fmt_ms(bench['p50_ms'])}</th><th class="num">{fmt_ms(bench['p95_ms'])}</th><th class="num">{fmt_ms(bench['min_ms'])}</th><th class="num">{fmt_ms(bench['max_ms'])}</th><th class="num">{bench['throughput_rps']:.2f} req/s</th><th class="num">{bench['wall_s']:.2f} s</th><th class="num">{fmt_gib(bench['max_rss_bytes'])}</th></tr>"""

text_samples_note = (
    f"Median of {bench['sample_count']} release-mode samples. "
    f"Sample 1 ran immediately after the fast suite and was an outlier "
    f"({bench['samples'][0]['p50_ms']:.1f} ms p50, {bench['samples'][0]['throughput_rps']:.2f} req/s); "
    f"samples 2-3 are stable at ~154.5 ms p50 and ~24.6 req/s. "
    f"Sample 1 process wall includes 2.93 s of compilation; samples 2-3 are warm builds."
)

embed_rows = "".join(
    f"""
        <tr>
          <td>{esc(R['embeds'][key]['label'])}</td>
          <td class="mono">{esc(R['embeds'][key]['input'])}</td>
          <td class="num">{R['embeds'][key]['wall_s']:.2f} s</td>
          <td class="num">{R['embeds'][key]['cold_wall_s']:.2f} s</td>
          <td class="num">{fmt_gib(R['embeds'][key]['max_rss_bytes'])}</td>
        </tr>"""
    for key in ("text", "image70", "audio")
)

parity_rows = "".join(
    f"""
        <tr>
          <td>{esc(R['parity'][key]['label'])}</td>
          <td>{html_badge(R['parity'][key]['status'])}</td>
          <td class="num">{R['parity'][key]['test_seconds']:.2f} s</td>
          <td class="num">{R['parity'][key]['wall_s']:.2f} s</td>
          <td class="num">{fmt_gib(R['parity'][key]['max_rss_bytes'])}</td>
        </tr>"""
    for key in ("text", "image", "audio")
)

comparison_rows = "".join([
    f'<tr><td>Text p50</td><td class="num">1,485 ms</td><td class="num">159 ms</td><td class="num"><strong>{fmt_ms(bench["p50_ms"])}</strong></td><td class="num">{fmt_pct(derived["text_p50_delta_vs_m7_pct"])}</td></tr>',
    f'<tr><td>Text p95</td><td class="num">1,504 ms</td><td class="num">159 ms</td><td class="num">{fmt_ms(bench["p95_ms"])}</td><td class="num">{fmt_pct(derived["text_p95_delta_vs_m7_pct"])}</td></tr>',
    f'<tr><td>Throughput (4T/20)</td><td class="num">2.43 req/s</td><td class="num">24.47 req/s</td><td class="num">{bench["throughput_rps"]:.2f} req/s</td><td class="num">{fmt_pct((derived["throughput_ratio_vs_m7"] - 1) * 100)}</td></tr>',
    f'<tr><td>Text parity</td><td class="num">—</td><td class="num">12.98 s</td><td class="num">{R["parity"]["text"]["test_seconds"]:.2f} s</td><td class="num">{fmt_pct((R["parity"]["text"]["test_seconds"] / hist["m7_text_parity_s"] - 1) * 100)}</td></tr>',
    f'<tr><td>Image parity (70 tok)</td><td class="num">—</td><td class="num">22.02 s</td><td class="num">{R["parity"]["image"]["test_seconds"]:.2f} s</td><td class="num">{fmt_pct((R["parity"]["image"]["test_seconds"] / hist["m7_image_parity_s"] - 1) * 100)}</td></tr>',
    f'<tr><td>Audio parity (short)</td><td class="num">—</td><td class="num">15.03 s</td><td class="num">{R["parity"]["audio"]["test_seconds"]:.2f} s</td><td class="num">{fmt_pct((R["parity"]["audio"]["test_seconds"] / hist["m7_audio_parity_s"] - 1) * 100)}</td></tr>',
    f'<tr><td>Binary size</td><td class="num">—</td><td class="num">—</td><td class="num">{derived["binary_size_gib"]:.2f} GiB</td><td class="num">= M8</td></tr>',
    f'<tr><td>Documents after restart</td><td class="num">—</td><td class="num">3</td><td class="num">{e2e["documents"]}</td><td class="num">=</td></tr>',
])

comparison_note = (
    f'<strong>Interpretation.</strong> Text p50 is {fmt_pct(derived["text_p50_delta_vs_m7_pct"])} '
    f'and throughput is {fmt_pct((derived["throughput_ratio_vs_m7"] - 1) * 100)} versus '
    f'the documented M7 baseline. '
    f'{esc(bench["note"])} Treat this as the new baseline for the next '
    f'improvement rather than a regression claim: run-to-run variance, thermal '
    f'state, and background load were not controlled across historical runs. '
    f'Text speedup versus the M2 scalar baseline is {fmt_x(derived["text_speedup_vs_m2"])}.'
)

comparability_note = (
    'Parity wall times use the same release-mode commands as the M7 replay and '
    'are directly comparable. Standalone image/audio embed timings are a '
    'different measurement (they include embedded startup and no test '
    'assertions) and should not be substituted into the parity columns. '
    'The p95 sample is sensitive to a single outlier (sample 3: 182.4 ms; '
    'sample 2: 154.9 ms); use p50 and throughput as the stable baseline.'
)

notes = "".join([
    f'<li><strong>Fast suite parity skip:</strong> <code>SELDON_SKIP_MODEL_PARITY=1</code> makes real-model parity tests return early; they were replayed separately and all pass.</li>',
    f'<li><strong>Network isolation:</strong> disabled because <code>sandbox-exec</code> was unavailable; do not claim an isolated offline run.</li>',
    f'<li><strong>Gatekeeper:</strong> the ad-hoc build is rejected by <code>spctl</code>; expected for the local-use artifact.</li>',
    f'<li><strong>Attempt 1:</strong> all lifecycle checks passed, but the harness exited 1 at the final report copy because the <code>--report</code> path was relative. Attempt 2 used an absolute path and passed; the attempt 1 log is retained as <code>logs/release-e2e-attempt1.log</code>.</li>',
    f'<li><strong>Memory:</strong> the harness max RSS is higher than individual embeddings because the HTTP server holds text, image, and audio weights in one process.</li>',
    f'<li><strong>Image/audio performance:</strong> the towers remain scalar correctness baselines; the 70-token image and 100 ms audio measurements are not optimized results.</li>',
])

repro = esc(f"""cd /Users/fouadassani/projets/seldon-vault-lite-rust

# Fast suite (includes the release E2E lifecycle test)
SELDON_SKIP_MODEL_PARITY=1 cargo test

# Release-mode text benchmark
cargo test --release --test text_bench -- --ignored --nocapture

# Standalone embedded embeddings
BIN=target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite
"$BIN" embed --text "What causes the northern lights?" --dim 768
"$BIN" embed --image fixtures/media/quadrants_16x16.png --dim 768 --vision-tokens 70
"$BIN" embed --audio fixtures/media/audio/short_tone_100ms.wav --dim 768

# Full release E2E harness (absolute --report path required)
scripts/release_e2e.sh \\
  --binary "$BIN" \\
  --report "$PWD/rapport/{run['id']}/release-e2e-report.json"

# Frozen-model parity replay
cargo test --release --test text_embeddings real_model_parity -- --nocapture
cargo test --release --test image_embeddings image_embedding_matches_the_official_reference -- --nocapture
cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture""")

text_chart = html_bar_chart(
    "Text p50 speedup vs M2 scalar",
    [
        ("M2 scalar", 1.0, "1.00x", "m2"),
        ("M7 NEON", hist["m2_text_p50_ms"] / hist["m7_text_p50_ms"], f"{hist['m2_text_p50_ms'] / hist['m7_text_p50_ms']:.2f}x", "m7"),
        ("Current run", derived["text_speedup_vs_m2"], f"{derived['text_speedup_vs_m2']:.2f}x", "current"),
    ],
    "higher is better",
)

embed_chart = html_bar_chart(
    "Warm standalone embed wall time",
    [
        ("Text", R["embeds"]["text"]["wall_s"], f"{R['embeds']['text']['wall_s']:.2f} s", "text"),
        ("Image (70 tok)", R["embeds"]["image70"]["wall_s"], f"{R['embeds']['image70']['wall_s']:.2f} s", "image"),
        ("Audio (100 ms)", R["embeds"]["audio"]["wall_s"], f"{R['embeds']['audio']['wall_s']:.2f} s", "audio"),
    ],
    "lower is better",
)

rss_chart = html_bar_chart(
    "Peak RSS",
    [
        ("Text embed", R["embeds"]["text"]["max_rss_bytes"] / (1024 ** 3), f"{R['embeds']['text']['max_rss_bytes'] / (1024 ** 3):.2f} GiB", "text"),
        ("Image embed", R["embeds"]["image70"]["max_rss_bytes"] / (1024 ** 3), f"{R['embeds']['image70']['max_rss_bytes'] / (1024 ** 3):.2f} GiB", "image"),
        ("Audio embed", R["embeds"]["audio"]["max_rss_bytes"] / (1024 ** 3), f"{R['embeds']['audio']['max_rss_bytes'] / (1024 ** 3):.2f} GiB", "audio"),
        ("E2E harness", e2e["max_rss_bytes"] / (1024 ** 3), f"{e2e['max_rss_bytes'] / (1024 ** 3):.2f} GiB", "rss"),
    ],
    "lower is better",
)

ring_pct = f"{100.0 * tests['passed'] / tests['total']:.2f}%"

repl = {
    "{{RUN_ID}}": esc(run["id"]),
    "{{GIT_SHORT}}": esc(run["git_short"]),
    "{{CARDS}}": cards,
    "{{FINGERPRINT}}": fingerprint,
    "{{ENV_ROWS}}": env_rows,
    "{{ARTIFACT_ROWS}}": artifact_rows,
    "{{TEST_SUMMARY}}": f"<code>{esc(tests['command'])}</code> · {tests['passed']} passed, {tests['failed']} failed, {tests['ignored']} ignored in {tests['wall_s']:.2f} s wall.",
    "{{RING_PCT}}": ring_pct,
    "{{TEST_PASSED}}": str(tests["passed"]),
    "{{TEST_IGNORED}}": str(tests["ignored"]),
    "{{TEST_TOTAL}}": str(tests["total"]),
    "{{TEST_FAILED}}": str(tests["failed"]),
    "{{SUITE_ROWS}}": "".join(suite_rows),
    "{{TEST_TOTAL_ROW}}": test_total_row,
    "{{E2E_STATUS}}": html_badge("passed", "PASSED"),
    "{{E2E_WALL}}": f"{e2e['wall_s']:.2f} s",
    "{{E2E_RSS}}": fmt_gib(e2e["max_rss_bytes"]),
    "{{E2E_DOCS}}": str(e2e["documents"]),
    "{{E2E_NETWORK}}": f"{esc(e2e['network_isolation'])} · sandbox-exec unavailable",
    "{{E2E_TIMESTAMP}}": esc(e2e["timestamp"]),
    "{{PHASES}}": phases,
    "{{ARTIFACT_CHECKS}}": artifact_checks,
    "{{TEXT_ROWS}}": text_rows,
    "{{TEXT_SAMPLES}}": text_samples_rows,
    "{{TEXT_SAMPLES_NOTE}}": text_samples_note,
    "{{TEXT_CHART}}": text_chart,
    "{{EMBED_ROWS}}": embed_rows,
    "{{EMBED_CHART}}": embed_chart,
    "{{RSS_CHART}}": rss_chart,
    "{{PARITY_ROWS}}": parity_rows,
    "{{COMPARISON_ROWS}}": comparison_rows,
    "{{COMPARISON_NOTE}}": comparison_note,
    "{{COMPARABILITY_NOTE}}": comparability_note,
    "{{NOTES}}": notes,
    "{{REPRO}}": repro,
}

template = (D / "template.html").read_text()
for key, value in repl.items():
    template = template.replace(key, value)

leftover = [part for part in template.split("{{")[1:] if "}}" in part]
if leftover:
    raise SystemExit(f"unreplaced template placeholders: {leftover[:3]}")

(D / "rapport.html").write_text(template)
print("wrote", D / "benchmark.md")
print("wrote", D / "rapport.html")
print("markdown bytes:", (D / "benchmark.md").stat().st_size)
print("html bytes:", (D / "rapport.html").stat().st_size)
