---
name: report-builder
description: "Create reproducible test, benchmark, and E2E reports in a timestamped rapport/ folder: benchmark.md, self-contained rapport.html, results.json, logs, and raw data. Use when asked for a report, benchmark, E2E summary, performance snapshot, baseline comparison, or 'rapport'. Keywords: report, benchmark, E2E, test report, performance report, HTML report, Markdown report, results, baseline, comparison, rapport, compte rendu, bilan, rapport HTML"
argument-hint: "[scope] [--quick|--full]"
user-invocable: true
---

# Report Builder

> **Version:** 1.0.0 | **Last updated:** 2026-11-11
>
> Build a reproducible snapshot of tests, benchmarks, and end-to-end validation.
> The output must be easy to read, visually clear, and directly comparable with
> future runs.

## Purpose

Use this skill when the user asks for a report, a benchmark, an E2E summary, a
performance snapshot, a release validation, or a comparison between runs. The
deliverable is a timestamped folder under `rapport/` containing:

```text
rapport/YYYY-MM-DD_HH-MM-SS/
├── benchmark.md              # readable Markdown report
├── rapport.html              # self-contained visual report
├── results.json              # machine-readable source of truth
├── release-e2e-report.json   # optional: raw harness report
├── template.html             # optional: HTML template with placeholders
├── collect_results.py        # optional: builds results.json from logs
├── generate_report.py        # optional: builds Markdown + HTML from JSON
├── logs/                     # raw test, benchmark, parity, harness logs
└── raw/                      # timings, parsed JSON, screenshots, vectors
```

Do not replace the previous report. Create a new timestamped folder so runs can
be compared side by side.

## Non-negotiable rules

1. **Reproducibility first.** Record the exact commands, working directory, git
   commit, toolchain, OS, CPU, memory, model revision, artifact hash, and binary
   size. A report without a fingerprint is not comparable.
2. **No cherry-picking.** Run each benchmark at least three times, keep every
   sample, and report the median. If one sample is an outlier, say so.
3. **Do not hide skips or failures.** Report passed, failed, and ignored counts
   separately. Never describe an ignored or skipped test as executed.
4. **Separate the measurements.** Test time, process wall time, cold start, warm
   start, and harness wall time are different numbers. Label each one.
5. **Only compare apples to apples.** Do not compare a standalone embed timing
   with a parity test wall time, or a test-profile timing with a release-profile
   timing.
6. **Preserve raw evidence.** Write logs and timing output to `logs/` and
   `raw/`. Do not edit them after the run.
7. **Keep the schema stable.** `results.json` is the comparison contract. Add
   fields, do not rename or repurpose existing ones without a `schema_version` bump.
8. **Sanitize.** Do not include tokens, secrets, private keys, or unnecessary
   home-directory paths. Redact paths if the report will be shared.
9. **Keep the folder small.** Do not commit model weights, build artifacts, or
   large binaries. Logs and screenshots should stay in the low-megabyte range.
10. **Verify the report, not just the tests.** A report with unreplaced
    placeholders, broken links, or numbers that disagree with `results.json`
    is not done.

## Workflow

### 1. Scope the run

Write down:

- what is being validated (tests, benchmark, E2E lifecycle, release artifact);
- the exact commands and profiles;
- which tests are expected to be ignored or skipped;
- what historical baseline the run will be compared against.

If the user asks for a "quick" report, still produce `results.json` and
`benchmark.md`. The HTML report is strongly recommended because it is the
main visual deliverable.

### 2. Create the timestamped folder

Use local time and include the timezone in the report metadata:

```bash
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
DIR="rapport/$STAMP"
mkdir -p "$DIR/logs" "$DIR/raw"
printf '%s\n' "$DIR" > rapport/.latest
```

### 3. Run the tests sequentially

Do not run benchmarks in parallel with tests or with each other. CPU contention
destroys the comparison. Capture stdout, stderr, exit code, wall time, and peak
RSS:

```bash
/usr/bin/time -l command 2>&1 | tee "$DIR/logs/name.log"
```

Run the fast suite first, then the heavy E2E harness, then parity, then
benchmarks. Use background jobs only for a long harness, and do not start other
CPU-heavy work while it runs.

### 4. Run benchmarks three times

For each benchmark:

- run one warm-up if the tool needs it;
- run at least three measured samples;
- store each sample separately;
- compute the median for the report, but keep all samples in `results.json`;
- flag the outlier explicitly if one sample differs materially.

### 5. Collect environment and artifact metadata

At minimum:

- CPU model, physical/logical cores, performance/efficiency split;
- RAM;
- OS and kernel version;
- architecture and target triple;
- compiler and toolchain versions;
- release/test profile settings;
- git commit, branch, and dirty status;
- model or fixture revision;
- artifact path, size, SHA-256, linked libraries, deployment target, signature;
- for E2E: harness report, network-isolation mode, signing, Gatekeeper result.

### 6. Build `results.json`

Use a small Python script with only the standard library. Parse the logs, do not
copy numbers by hand. Keep the schema below and compute derived metrics from the
raw values.

### 7. Generate `benchmark.md` and `rapport.html`

Generate both from `results.json`. Do not maintain two independent sets of
numbers. Use an HTML template with `{{PLACEHOLDER}}` tokens, replace every token,
and fail the build if any placeholder remains.

### 8. Validate

Run the checklist at the end of this skill. Open the HTML in a browser or render
it headlessly, and inspect the screenshot. Check the bottom of the page, tables
with long paths, and mobile-width behavior.

### 9. Present and archive

Present `benchmark.md`, `rapport.html`, and `results.json` as the
deliverables. Keep `rapport/.latest` pointing at the newest folder. The report
folder is normally untracked; commit it only if the user wants the baseline in
Git.

## Data collection checklist

### Environment

- [ ] CPU and core split
- [ ] RAM
- [ ] OS and kernel
- [ ] Architecture and target
- [ ] Rust/compiler and build tool versions
- [ ] Release and test profile settings

### Artifact

- [ ] Path
- [ ] Size in bytes and GiB
- [ ] SHA-256
- [ ] Linked libraries
- [ ] Deployment target
- [ ] Signature and Gatekeeper result
- [ ] Model source (embedded bundle, directory, or cache)

### Tests

- [ ] Exact command
- [ ] Passed / failed / ignored / total
- [ ] Wall time and peak RSS
- [ ] Per-suite breakdown
- [ ] Notes for tests that skipped internally

### Benchmarks

- [ ] Input, prompt, dimension, and parameters
- [ ] Warm-up policy
- [ ] At least three samples
- [ ] p50, p95, min, max
- [ ] Throughput and concurrency
- [ ] Cold and warm wall time
- [ ] Peak RSS
- [ ] Median and outlier note

### E2E

- [ ] Harness command and absolute report path
- [ ] Status and wall time
- [ ] Phase list with pass/fail
- [ ] Documents before and after restart
- [ ] Network-isolation mode
- [ ] Sanitized environment details

### Parity

- [ ] Test command
- [ ] Status
- [ ] Test time and process wall time
- [ ] Peak RSS
- [ ] Tolerance or golden reference

### Historical baseline

- [ ] Source document and date
- [ ] Values used for comparison
- [ ] Whether they were re-measured or quoted
- [ ] Comparability notes

## `results.json` schema

Use a stable, explicit schema. A minimal example:

```json
{
  "schema_version": 1,
  "run": {
    "id": "YYYY-MM-DD_HH-MM-SS",
    "started_local": "2026-11-11T23:08:37+01:00",
    "started_utc": "2026-11-11T22:08:37Z",
    "finished_local": "2026-11-11T23:35:16+01:00",
    "git_commit": "...",
    "git_branch": "main",
    "git_status": "clean except rapport/"
  },
  "environment": {
    "cpu": "Apple M2 Pro",
    "cores": 10,
    "performance_cores": 6,
    "efficiency_cores": 4,
    "memory_bytes": 17179869184,
    "os": "macOS 26.5.1 (25F80)",
    "arch": "arm64",
    "target": "aarch64-apple-darwin",
    "rustc": "1.99.1",
    "cargo": "1.99.1",
    "release_profile": "opt-level=3, thin LTO, codegen-units=1",
    "test_profile": "opt-level=2"
  },
  "artifact": {
    "path": "target/.../seldon-vault-lite",
    "sha256": "...",
    "bytes": 1539215600,
    "linked_libraries": ["/usr/lib/libSystem.B.dylib"],
    "minos": "11.0",
    "signature": "adhoc",
    "gatekeeper": "rejected"
  },
  "tests": {
    "command": "SELDON_SKIP_MODEL_PARITY=1 cargo test",
    "passed": 438,
    "failed": 0,
    "ignored": 15,
    "total": 453,
    "wall_s": 15.14,
    "max_rss_bytes": 260734976,
    "suites": []
  },
  "benchmarks": {
    "text": {
      "command": "...",
      "sample_count": 3,
      "samples": [],
      "p50_ms": 154.8,
      "p95_ms": 173.2,
      "throughput_rps": 24.55,
      "note": "Median of three samples; sample 1 was an outlier."
    }
  },
  "e2e": {
    "status": "passed",
    "wall_s": 208.64,
    "max_rss_bytes": 4857331712,
    "phases": []
  },
  "parity": {},
  "historical": {},
  "derived": {}
}
```

Rules:

- Keep raw samples in `samples`; do not store only the median.
- Store bytes as integers, durations in seconds or milliseconds with a suffix
  in the key name.
- Store booleans and enums as strings only when they are labels (for example
  `signature: "adhoc"`).
- Compute derived values (speedup, delta, pass rate) from the raw fields.
- Never put secrets or tokens in `results.json`.

## `benchmark.md` outline

Use this order:

1. Title, run ID, start/finish, status.
2. **At a glance** table: tests, E2E, key latency, throughput, artifact, parity.
3. **Run fingerprint**: git commit, model revision, target, profiles.
4. **Environment** and **Artifact** tables.
5. **Test suite**: command, counts, wall time, suite breakdown.
6. **Release E2E harness**: command, status, phases, artifact checks.
7. **Performance**: latency/throughput, all samples, standalone embeddings,
   memory, parity replay.
8. **Comparison with documented baselines**: side-by-side table with deltas and
   a comparability note.
9. **Notes and caveats**: skips, network isolation, signing, outliers, attempt
   history.
10. **Reproduction**: exact copy-paste commands.
11. **Report files**: manifest of the folder.

Use tables for numbers. Use fenced code blocks for commands. Keep the prose
short; the tables and the HTML carry the detail.

## `rapport.html` requirements

The HTML report must be **self-contained**: inline CSS, no external fonts,
scripts, images, or CDNs. It must open offline and print cleanly.

### Layout

- Hero header with run ID, status pill, date, and commit.
- Six summary cards: tests, E2E, p50, throughput, speedup, artifact size.
- Run fingerprint grid.
- Environment and artifact tables side by side.
- Test suite section with a pass/ignored ring and a suite table.
- E2E timeline with one item per phase and artifact checks.
- Performance section with latency table, sample table, and horizontal bar
  charts.
- Parity table.
- Comparison table with deltas.
- Notes and caveats.
- Reproduction code block.
- Footer with generation source.

### Visual rules

- Use CSS variables for colors, spacing, radius, and shadows.
- Light theme with a dark gradient hero; accent colors for primary actions.
- Use badges for PASS/FAIL/SKIP, not color alone.
- Use `font-variant-numeric: tabular-nums` for numeric columns.
- Use responsive grids and stack columns below ~860px.
- Use `overflow-x: auto` for wide tables.
- Add `@media print` rules and avoid breaking panels across pages.
- Keep contrast accessible; do not rely on tiny gray text for critical data.
- Prefer CSS bar charts and a conic-gradient ring over external chart libraries.

### Template workflow

- Keep `template.html` with `{{PLACEHOLDER}}` tokens.
- Build a Python dict of replacements from `results.json`.
- Replace every token and raise an error if any `{{...}}` remains.
- Write `rapport.html` and `benchmark.md` from the same data.
- If Chrome is available, render a headless screenshot for visual verification:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --no-sandbox \
  --window-size=1440,6500 \
  --screenshot="$DIR/raw/report.png" \
  "file://$PWD/$DIR/rapport.html"
```

Inspect the screenshot. Check the hero, cards, tables, charts, the bottom of the
page, and a narrow viewport if practical.

## Statistics and comparability

- **Median, not best-of.** With three samples, the median is robust to one
  outlier. Report all samples and the median.
- **p50 vs p95.** p95 is sensitive to a single outlier. If p95 jumps while p50
  and throughput are stable, say so and point to the sample table.
- **Cold vs warm.** The first run after a build or reboot may page in a large
  binary. Label cold and warm runs separately.
- **Test time vs process wall.** A cargo test includes compilation and process
  startup; the test binary's `finished in` line is the test time. Report both
  when they differ.
- **Throughput.** Record threads, request count, wall time, and requests/second.
- **Memory.** Use `maximum resident set size` from `/usr/bin/time -l`. Note
  when a harness holds several models at once.
- **Deltas.** Compute `(current - baseline) / baseline * 100` and state the
  sign convention. For latency, negative is better; for throughput, positive is
  better.
- **Do not overclaim.** A delta within a few percent on an uncontrolled machine
  is noise. Call it a baseline, not a regression or an improvement.

## Comparison workflow

1. Keep `schema_version` and field names stable across runs.
2. Keep `rapport/.latest` pointing at the newest folder.
3. Compare `results.json` files with a small Python script, not by eye.
4. Compare only runs with the same command, profile, model revision, and
   artifact. If the environment changed, say so before comparing numbers.
5. Archive the raw `release-e2e-report.json` with the run.
6. If a historical value is quoted from documentation, label it as quoted and
   not re-measured.

## Validation checklist

- [ ] Folder name is `rapport/YYYY-MM-DD_HH-MM-SS/`.
- [ ] `benchmark.md`, `rapport.html`, and `results.json` exist.
- [ ] `python3 -m json.tool results.json` succeeds.
- [ ] No `{{...}}` placeholders remain in `rapport.html`.
- [ ] The HTML has no external `src=` or `href=` dependencies.
- [ ] The report opens offline and renders correctly at desktop and mobile width.
- [ ] Every number in `benchmark.md` matches `results.json`.
- [ ] Passed, failed, and ignored counts are consistent everywhere.
- [ ] Skipped tests are described as skipped, not as executed.
- [ ] Raw logs and timing files are present under `logs/` and `raw/`.
- [ ] No secrets, tokens, or private keys are present.
- [ ] The report folder is small enough to archive.
- [ ] The final response links the deliverables and summarizes the key numbers.

## Anti-patterns

- Reporting only the best benchmark sample.
- Reporting only the median and hiding the raw samples.
- Mixing test-profile and release-profile numbers.
- Calling a skipped parity test "passed" without a separate replay.
- Comparing standalone embed timings with parity test wall times.
- Claiming network isolation when the harness recorded it as disabled.
- Claiming Gatekeeper acceptance for an ad-hoc build.
- Using a relative `--report` path for a harness that changes directory.
- Copying numbers by hand into the Markdown and the HTML.
- Shipping an HTML report with external CDN assets.
- Leaving `TODO`, `FIXME`, or `{{PLACEHOLDER}}` tokens in the final report.
- Committing large binaries or model weights with the report.

## Project example: Seldon Vault Lite

For this repository, the reference run used:

```bash
# Fast suite (parity tests return early via the environment variable)
SELDON_SKIP_MODEL_PARITY=1 cargo test

# Text benchmark: three samples, release profile
for i in 1 2 3; do
  cargo test --release --test text_bench -- --ignored --nocapture \
    2>&1 | tee "rapport/$STAMP/logs/text-bench-$i.log"
done

# Full embedded release E2E; --report must be absolute because the harness
# changes into its staging directory before copying the report.
scripts/release_e2e.sh \
  --binary target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite \
  --report "$PWD/rapport/$STAMP/release-e2e-report.json"

# Frozen-model parity replay
cargo test --release --test text_embeddings real_model_parity -- --nocapture
cargo test --release --test image_embeddings image_embedding_matches_the_official_reference -- --nocapture
cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture
```

Standalone embed timings should use `/usr/bin/time -l` around the embedded
binary, once for cold and once for warm:

```bash
BIN=target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite
/usr/bin/time -l -o "$DIR/raw/text.time" "$BIN" embed \
  --text "What causes the northern lights?" --dim 768 > "$DIR/raw/text.json"
/usr/bin/time -l -o "$DIR/raw/image70.time" "$BIN" embed \
  --image fixtures/media/quadrants_16x16.png --dim 768 --vision-tokens 70 > "$DIR/raw/image70.json"
/usr/bin/time -l -o "$DIR/raw/audio.time" "$BIN" embed \
  --audio fixtures/media/audio/short_tone_100ms.wav --dim 768 > "$DIR/raw/audio.json"
```

Known caveats for this project:

- `sandbox-exec` may be unavailable; record `network_isolation: disabled`
  and do not claim an isolated offline run.
- The ad-hoc linker-signed build is expected to be rejected by `spctl`.
- The image and audio towers are scalar correctness baselines; their timings
  are not optimized results.
- `SELDON_SKIP_MODEL_PARITY=1` makes the real-model parity tests return early;
  always replay them separately in release mode.

## Minimal Python helpers

Use only the standard library. Median:

```python
import statistics

p50 = statistics.median([sample["p50_ms"] for sample in samples])
throughput = statistics.median([sample["throughput_rps"] for sample in samples])
```

Safe HTML escaping:

```python
import html

def esc(value):
    return html.escape(str(value), quote=True)
```

Template replacement with a guard:

```python
template = (folder / "template.html").read_text()
for key, value in replacements.items():
    template = template.replace(key, value)
if "{{" in template:
    raise SystemExit("unreplaced template placeholder")
(folder / "rapport.html").write_text(template)
```

## Definition of done

A report is done when:

- the run is reproducible from the recorded commands;
- every number is traceable to a raw log or `results.json`;
- pass, fail, skip, cold, warm, and outlier cases are explicit;
- `benchmark.md` and `rapport.html` tell the same story;
- the HTML is self-contained, responsive, and visually verified;
- the folder is archived under `rapport/` with a timestamp;
- the final response links the files and states the key results and caveats.
