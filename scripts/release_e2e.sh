#!/usr/bin/env bash
# Local-use end-to-end validation for the macOS release artifact.
#
# This harness is intentionally independent of Cargo and of the development
# model cache. It stages the release binary and the committed media fixtures in
# a temporary install root, runs the binary with a sanitized environment, and
# exercises the full text, image, audio, storage, and HTTP lifecycle.
#
# This is the release validation harness for the development Apple Silicon
# machine. No separate clean-machine gate is required.
#
# Usage:
#   scripts/release_e2e.sh [--binary PATH] [--model PATH] [--workdir DIR]
#                          [--report PATH] [--skip-network-isolation] [--keep]
#
# --model is only for a non-embedded development binary. A release validation
# must omit it and use the embedded-model release build.
#
# The script uses only macOS system tools (bash, env, awk, sed, grep, curl,
# codesign, otool, vtool, spctl, shasum). It does not require Homebrew, Python,
# Cargo, or an inference engine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FIXTURE_MEDIA="$REPO_ROOT/fixtures/media"
DIMENSION=768

BINARY=""
MODEL=""
WORKDIR=""
REPORT=""
KEEP=0
SKIP_NETWORK=0

usage() {
    cat <<'EOF'
Usage: scripts/release_e2e.sh [options]

  --binary PATH              release binary (default: embedded-full build if present,
                             otherwise target/aarch64-apple-darwin/release/seldon-vault-lite)
  --model PATH               optional model directory or .svlm for a non-embedded binary;
                             omit for a single-file release validation
  --workdir DIR              staging directory (kept on exit)
  --report PATH              copy the JSON report to PATH
  --skip-network-isolation   do not attempt sandbox-exec network denial
  --keep                     keep the staging directory on success
  -h, --help                 show this help

A release validation must use an embedded-model build, no model cache, and a
system-only PATH. See docs/release.md.
EOF
}

fail() {
    echo "release-e2e: ERROR: $*" >&2
    exit 1
}

warn() {
    echo "release-e2e: WARNING: $*" >&2
}

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" ;;
    esac
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary)
            [ "$#" -ge 2 ] || fail "--binary requires a value"
            BINARY="$2"
            shift 2
            ;;
        --model)
            [ "$#" -ge 2 ] || fail "--model requires a value"
            MODEL="$2"
            shift 2
            ;;
        --workdir)
            [ "$#" -ge 2 ] || fail "--workdir requires a value"
            WORKDIR="$2"
            KEEP=1
            shift 2
            ;;
        --report)
            [ "$#" -ge 2 ] || fail "--report requires a value"
            REPORT="$2"
            shift 2
            ;;
        --skip-network-isolation)
            SKIP_NETWORK=1
            shift
            ;;
        --keep)
            KEEP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "release-e2e: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$BINARY" ]; then
    if [ -x "$REPO_ROOT/target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite" ]; then
        BINARY="$REPO_ROOT/target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite"
    elif [ -x "$REPO_ROOT/target/aarch64-apple-darwin/release/seldon-vault-lite" ]; then
        BINARY="$REPO_ROOT/target/aarch64-apple-darwin/release/seldon-vault-lite"
    else
        fail "no release binary found; build with --release or pass --binary"
    fi
fi

BINARY="$(absolute_path "$BINARY")"
[ -x "$BINARY" ] || fail "binary is not executable: $BINARY"
[ -d "$FIXTURE_MEDIA" ] || fail "media fixtures are missing: $FIXTURE_MEDIA"

if [ -n "$MODEL" ]; then
    MODEL="$(absolute_path "$MODEL")"
    [ -e "$MODEL" ] || fail "model path does not exist: $MODEL"
    warn "--model was provided; this run is not a single-file release validation"
fi

if [ -n "$WORKDIR" ]; then
    WORKDIR="$(absolute_path "$WORKDIR")"
    mkdir -p "$WORKDIR"
else
    WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/seldon-release-e2e.XXXXXX")"
fi

INSTALL="$WORKDIR/install"
case "$INSTALL" in
    "$WORKDIR"/*) ;;
    *) fail "refusing to use staging path $INSTALL" ;;
esac
rm -rf "$INSTALL"
mkdir -p "$INSTALL/fixtures" "$WORKDIR/home" "$WORKDIR/tmp"
cp "$BINARY" "$INSTALL/seldon-vault-lite"
chmod 755 "$INSTALL/seldon-vault-lite"
cp -R "$FIXTURE_MEDIA" "$INSTALL/fixtures/media"

SERVER_PID=""
SERVER_OUT="$WORKDIR/serve.out"
SERVER_ERR="$WORKDIR/serve.err"
ADDRESS=""
NETWORK_ISOLATION="disabled"
SANDBOX_PROFILE='(version 1)(allow default)(deny network*)'
SIGNING="unknown"
GATEKEEPER="unknown"

cleanup() {
    status=$?
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ "$status" -ne 0 ]; then
        echo "release-e2e: staging directory preserved at $WORKDIR" >&2
    elif [ "$KEEP" -eq 0 ]; then
        rm -rf "$WORKDIR"
    else
        echo "release-e2e: staging directory kept at $WORKDIR"
    fi
}
trap cleanup EXIT

clean_env() {
    env -i \
        HOME="$WORKDIR/home" \
        TMPDIR="$WORKDIR/tmp" \
        PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
        LANG="en_US.UTF-8" \
        LC_ALL="en_US.UTF-8" \
        "$@"
}

run_cli() {
    if [ "$NETWORK_ISOLATION" = "sandbox-exec" ]; then
        clean_env /usr/bin/sandbox-exec -p "$SANDBOX_PROFILE" "$INSTALL/seldon-vault-lite" "$@"
    else
        clean_env "$INSTALL/seldon-vault-lite" "$@"
    fi
}

run_model_cli() {
    if [ -n "$MODEL" ]; then
        run_cli --model-dir "$MODEL" "$@"
    else
        run_cli "$@"
    fi
}

curl_clean() {
    env -i \
        HOME="$WORKDIR/home" \
        TMPDIR="$WORKDIR/tmp" \
        PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
        /usr/bin/curl "$@"
}

assert_contains() {
    text="$1"
    needle="$2"
    label="$3"
    case "$text" in
        *"$needle"*) ;;
        *) fail "$label: expected '$needle' in: $text" ;;
    esac
}

assert_vector() {
    vector="$1"
    dimension="$2"
    label="$3"
    case "$vector" in
        "["*"]") ;;
        *) fail "$label: output is not a JSON array: $vector" ;;
    esac
    commas="$(printf '%s' "$vector" | tr -cd ',' | wc -c | tr -d ' ')"
    count=$((commas + 1))
    [ "$count" -eq "$dimension" ] || fail "$label: expected $dimension values, got $count"
    norm="$(printf '%s\n' "$vector" | sed -e 's/^\[//' -e 's/\]$//' | awk -F, '{s=0; for (i=1; i<=NF; i++) {v=$i+0; s+=v*v}; printf "%.8f", sqrt(s)}')"
    awk -v n="$norm" 'BEGIN { if (n < 0.9999 || n > 1.0001) exit 1 }' \
        || fail "$label: vector norm $norm is not 1 within tolerance"
}

json_number() {
    json="$1"
    key="$2"
    printf '%s' "$json" | sed -n "s/.*\"$key\":\([0-9][0-9]*\).*/\1/p" | head -1
}

assert_json_number() {
    json="$1"
    key="$2"
    expected="$3"
    label="$4"
    actual="$(json_number "$json" "$key")"
    [ -n "$actual" ] || fail "$label: missing number $key in: $json"
    [ "$actual" -eq "$expected" ] || fail "$label: expected $key=$expected, got $actual"
}

assert_json_positive() {
    json="$1"
    key="$2"
    label="$3"
    actual="$(json_number "$json" "$key")"
    [ -n "$actual" ] || fail "$label: missing number $key in: $json"
    [ "$actual" -gt 0 ] || fail "$label: expected $key > 0, got $actual"
}

first_document_id() {
    printf '%s' "$1" | grep -o '"document_id":"[^"]*"' | head -1 | sed -e 's/^"document_id":"//' -e 's/"$//'
}

assert_first_document_id() {
    json="$1"
    expected="$2"
    label="$3"
    actual="$(first_document_id "$json")"
    [ "$actual" = "$expected" ] || fail "$label: expected first document $expected, got $actual"
}

start_server() {
    : > "$SERVER_OUT"
    : > "$SERVER_ERR"
    if [ -n "$MODEL" ]; then
        clean_env "$INSTALL/seldon-vault-lite" serve \
            --data-dir "$INSTALL/data" \
            --bind 127.0.0.1:0 \
            --dim "$DIMENSION" \
            --model-dir "$MODEL" \
            >"$SERVER_OUT" 2>"$SERVER_ERR" &
    else
        clean_env "$INSTALL/seldon-vault-lite" serve \
            --data-dir "$INSTALL/data" \
            --bind 127.0.0.1:0 \
            --dim "$DIMENSION" \
            >"$SERVER_OUT" 2>"$SERVER_ERR" &
    fi
    SERVER_PID=$!
    waited=0
    while [ "$waited" -lt 1800 ]; do
        if grep -q '"status":"listening"' "$SERVER_OUT" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            fail "server exited before listening: $(cat "$SERVER_ERR")"
        fi
        sleep 0.1
        waited=$((waited + 1))
    done
    grep -q '"status":"listening"' "$SERVER_OUT" \
        || fail "server did not print a listening record: $(cat "$SERVER_ERR")"
    ADDRESS="$(sed -n 's/.*"address":"\([^"]*\)".*/\1/p' "$SERVER_OUT" | head -1)"
    [ -n "$ADDRESS" ] || fail "server listening record has no address"
    echo "release-e2e: server listening at $ADDRESS"
}

stop_server() {
    code="$(curl_clean -sS --max-time 60 -o "$WORKDIR/shutdown.json" -w '%{http_code}' \
        -X POST "http://$ADDRESS/v1/shutdown" \
        -H 'Content-Type: application/json' \
        --data-binary '')"
    [ "$code" = "202" ] || fail "shutdown returned HTTP $code: $(cat "$WORKDIR/shutdown.json")"
    waited=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        [ "$waited" -lt 600 ] || fail "server did not stop within 60 seconds"
        sleep 0.1
        waited=$((waited + 1))
    done
    wait "$SERVER_PID" || fail "server exited with a failure status"
    SERVER_PID=""
    grep -q '"status":"stopped"' "$SERVER_OUT" || fail "server did not print a stopped record"
}

artifact_checks() {
    echo "release-e2e: artifact checks"
    linked="$(otool -L "$BINARY" | tail -n +2 | awk '{print $1}' | sort -u)"
    [ "$linked" = "/usr/lib/libSystem.B.dylib" ] \
        || fail "unexpected linked libraries: $linked"
    build_info="$(vtool -show-build "$BINARY")"
    assert_contains "$build_info" "minos 11.0" "deployment target"
    codesign --verify --strict "$BINARY" >/dev/null 2>&1 \
        || fail "codesign verification failed"
    signature_info="$(codesign -dv --verbose=4 "$BINARY" 2>&1 || true)"
    case "$signature_info" in
        *"Signature=adhoc"*)
            SIGNING="adhoc"
            ;;
        *)
            fail "release validation expects an ad-hoc local-use signature: $signature_info"
            ;;
    esac
    if spctl --assess --type execute -vv "$BINARY" >"$WORKDIR/spctl.txt" 2>&1; then
        GATEKEEPER="accepted"
    else
        GATEKEEPER="rejected"
    fi
    if [ "$GATEKEEPER" = "accepted" ]; then
        warn "Gatekeeper accepted the ad-hoc build; assessment recorded"
    else
        echo "release-e2e: Gatekeeper rejected the ad-hoc build (expected for local use)"
    fi
    if command -v cargo >/dev/null 2>&1 && [ -f "$REPO_ROOT/Cargo.toml" ]; then
        cargo_tree="$(cd "$REPO_ROOT" && cargo tree --edges normal,build,dev 2>/dev/null)" \
            || fail "cargo tree failed"
        lines="$(printf '%s\n' "$cargo_tree" | wc -l | tr -d ' ')"
        [ "$lines" -eq 1 ] || fail "the Cargo graph contains dependencies"
        echo "release-e2e: Cargo graph contains only the root crate"
    else
        echo "release-e2e: Cargo is unavailable; the zero-crate check is skipped"
    fi
    echo "release-e2e: linked libraries: $linked"
    echo "release-e2e: deployment target: minos 11.0"
    echo "release-e2e: signature: $SIGNING; Gatekeeper: $GATEKEEPER"
}

echo "release-e2e: binary=$BINARY"
echo "release-e2e: staging=$INSTALL"
echo "release-e2e: sanitized environment HOME=$WORKDIR/home TMPDIR=$WORKDIR/tmp PATH=/usr/bin:/bin:/usr/sbin:/sbin"

artifact_checks

if [ "$SKIP_NETWORK" -eq 0 ]; then
    if command -v sandbox-exec >/dev/null 2>&1; then
        if clean_env /usr/bin/sandbox-exec -p "$SANDBOX_PROFILE" "$INSTALL/seldon-vault-lite" version >/dev/null 2>&1; then
            NETWORK_ISOLATION="sandbox-exec"
            echo "release-e2e: CLI network isolation enabled with sandbox-exec"
        else
            warn "sandbox-exec is unavailable in this environment; CLI offline check runs without network isolation"
        fi
    else
        warn "sandbox-exec not found; CLI offline check runs without network isolation"
    fi
fi

version="$(run_cli version)"
case "$version" in
    "seldon-vault-lite "*" (aarch64-apple-darwin)") ;;
    *) fail "unexpected version output: $version" ;;
esac
help="$(run_cli help)"
assert_contains "$help" "USAGE:" "help"
for command in embed index search verify compact serve version; do
    assert_contains "$help" "$command" "help must mention $command"
done

if [ -z "$MODEL" ]; then
    probe="$(run_cli embed --ids 2,5,6,7,1 --dim 8 2>"$WORKDIR/probe.err")" \
        || fail "binary is not embedded and no --model was given; use the embedded-model build or pass --model"
    assert_vector "$probe" 8 "embedded model probe"
    echo "release-e2e: embedded model probe passed"
fi

cd "$INSTALL"

echo "release-e2e: text embedding"
text_vector="$(run_model_cli embed --text 'The northern lights are caused by solar wind.' --dim "$DIMENSION")"
assert_vector "$text_vector" "$DIMENSION" "text embedding"
text_vector_again="$(run_model_cli embed --text 'The northern lights are caused by solar wind.' --dim "$DIMENSION")"
[ "$text_vector" = "$text_vector_again" ] || fail "text embedding is not deterministic"

echo "release-e2e: image embedding"
image_vector="$(run_model_cli embed --image "fixtures/media/quadrants_16x16.png" --dim "$DIMENSION")"
assert_vector "$image_vector" "$DIMENSION" "image embedding"

echo "release-e2e: audio embedding"
audio_vector="$(run_model_cli embed --audio "fixtures/media/audio/short_tone_100ms.wav" --dim "$DIMENSION")"
assert_vector "$audio_vector" "$DIMENSION" "audio embedding"

cat > documents.jsonl <<'EOF'
{"id":"text-northern-lights","text":"The northern lights are caused by solar wind.","title":"Northern lights"}
{"id":"image-quadrants","image":"fixtures/media/quadrants_16x16.png","title":"Quadrants"}
{"id":"audio-tone","audio":"fixtures/media/audio/short_tone_100ms.wav","title":"Short tone"}
EOF

echo "release-e2e: indexing text, image, and audio documents"
summary="$(run_model_cli index --data-dir "$INSTALL/data" --vision-tokens 70 "$INSTALL/documents.jsonl")"
assert_json_number "$summary" documents 3 "index summary"
assert_json_number "$summary" chunks 3 "index summary"
assert_json_number "$summary" dimension "$DIMENSION" "index summary"

echo "release-e2e: lexical, vector, and hybrid search"
lexical="$(run_cli search --data-dir "$INSTALL/data" --mode lexical 'northern lights')"
assert_contains "$lexical" '"document_id":"text-northern-lights"' "lexical search"
assert_first_document_id "$lexical" "text-northern-lights" "lexical search"

vector_text="$(run_model_cli search --data-dir "$INSTALL/data" --mode vector --dim "$DIMENSION" 'northern lights')"
assert_first_document_id "$vector_text" "text-northern-lights" "vector text search"

vector_image="$(run_model_cli search --data-dir "$INSTALL/data" --mode vector --dim "$DIMENSION" --vision-tokens 70 --image "fixtures/media/quadrants_16x16.png")"
assert_first_document_id "$vector_image" "image-quadrants" "vector image search"

vector_audio="$(run_model_cli search --data-dir "$INSTALL/data" --mode vector --dim "$DIMENSION" --audio "fixtures/media/audio/short_tone_100ms.wav")"
assert_first_document_id "$vector_audio" "audio-tone" "vector audio search"

hybrid="$(run_model_cli search --data-dir "$INSTALL/data" --mode hybrid --dim "$DIMENSION" 'northern lights')"
assert_contains "$hybrid" '"document_id":"text-northern-lights"' "hybrid search"

echo "release-e2e: verification and compaction"
verification="$(run_cli verify --data-dir "$INSTALL/data")"
assert_json_number "$verification" documents 3 "verify"
assert_json_number "$verification" chunks 3 "verify"
assert_json_number "$verification" truncated_bytes 0 "verify"

compaction="$(run_cli compact --data-dir "$INSTALL/data")"
assert_json_number "$compaction" documents 3 "compact"
assert_json_number "$compaction" chunks 3 "compact"
assert_json_positive "$compaction" snapshot_bytes "compact"

verification="$(run_cli verify --data-dir "$INSTALL/data")"
assert_json_number "$verification" documents 3 "verify after compaction"
assert_json_number "$verification" chunks 3 "verify after compaction"
assert_json_number "$verification" records 0 "verify after compaction"
assert_json_number "$verification" snapshot_records 3 "verify after compaction"

vector_after="$(run_model_cli search --data-dir "$INSTALL/data" --mode vector --dim "$DIMENSION" 'northern lights')"
assert_first_document_id "$vector_after" "text-northern-lights" "search after compaction"
echo "release-e2e: restart after compaction preserved all documents"

echo "release-e2e: HTTP API lifecycle"
start_server
health="$(curl_clean -fsS --max-time 60 "http://$ADDRESS/health")"
assert_contains "$health" '"status":"ok"' "health"
ready="$(curl_clean -fsS --max-time 60 "http://$ADDRESS/ready")"
assert_contains "$ready" '"ready":true' "ready"
assert_json_number "$ready" documents 3 "ready before ingestion"
models="$(curl_clean -fsS --max-time 60 "http://$ADDRESS/v1/models")"
assert_contains "$models" '"embedding_dimension":768' "models"
assert_contains "$models" '"text":true' "models text"
assert_contains "$models" '"image":true' "models image"
assert_contains "$models" '"audio":true' "models audio"
embedding="$(curl_clean -fsS --max-time 120 -X POST "http://$ADDRESS/v1/embeddings" \
    -H 'Content-Type: application/json' \
    --data-binary '{"input":{"type":"text","text":"The northern lights are caused by solar wind."},"dimension":768}')"
assert_contains "$embedding" '"embedding":[' "HTTP text embedding"
assert_contains "$embedding" '"dimension":768' "HTTP text embedding"
document="$(curl_clean -fsS --max-time 120 -X POST "http://$ADDRESS/v1/documents" \
    -H 'Content-Type: application/json' \
    --data-binary '{"id":"http-aurora","input":{"type":"text","text":"Aurora borealis over the Arctic sky."},"title":"HTTP aurora"}')"
job_id="$(json_number "$document" id)"
[ -n "$job_id" ] || fail "document ingestion response has no job id: $document"
job_status=""
for _ in $(seq 1 240); do
    job="$(curl_clean -fsS --max-time 60 "http://$ADDRESS/v1/jobs/$job_id")"
    case "$job" in
        *'"status":"succeeded"'*)
            job_status="succeeded"
            break
            ;;
        *'"status":"failed"'*)
            fail "ingestion job failed: $job"
            ;;
    esac
    sleep 0.25
done
[ "$job_status" = "succeeded" ] || fail "ingestion job did not finish"
search="$(curl_clean -fsS --max-time 120 -X POST "http://$ADDRESS/v1/search" \
    -H 'Content-Type: application/json' \
    --data-binary '{"query":{"type":"text","text":"aurora borealis"},"mode":"hybrid"}')"
assert_contains "$search" '"document_id":"http-aurora"' "HTTP hybrid search"
deleted="$(curl_clean -fsS --max-time 60 -X DELETE "http://$ADDRESS/v1/documents/http-aurora")"
assert_contains "$deleted" '"deleted":true' "HTTP document deletion"
ready="$(curl_clean -fsS --max-time 60 "http://$ADDRESS/ready")"
assert_json_number "$ready" documents 3 "ready after deletion"
stop_server
echo "release-e2e: HTTP server stopped cleanly"

echo "release-e2e: HTTP restart persistence"
start_server
ready="$(curl_clean -fsS --max-time 60 "http://$ADDRESS/ready")"
assert_json_number "$ready" documents 3 "ready after restart"
search="$(curl_clean -fsS --max-time 120 -X POST "http://$ADDRESS/v1/search" \
    -H 'Content-Type: application/json' \
    --data-binary '{"query":{"type":"text","text":"northern lights"},"mode":"hybrid"}')"
assert_contains "$search" '"document_id":"text-northern-lights"' "HTTP search after restart"
stop_server
echo "release-e2e: HTTP restart preserved all documents"

sha256="$(shasum -a 256 "$BINARY" | awk '{print $1}')"
bytes="$(stat -f%z "$BINARY")"
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
os_version="$(sw_vers -productVersion)"
machine="$(uname -m)"
report_path="$WORKDIR/release-e2e-report.json"
cat > "$report_path" <<EOF
{"status":"passed","timestamp":"$timestamp","platform":"$machine","os":"$os_version","binary":"$BINARY","binary_sha256":"$sha256","binary_bytes":$bytes,"version":"$version","signing":"$SIGNING","gatekeeper":"$GATEKEEPER","network_isolation":"$NETWORK_ISOLATION","dimension":$DIMENSION,"documents":3,"fixtures":"$FIXTURE_MEDIA","staging":"$INSTALL"}
EOF

if [ -n "$REPORT" ]; then
    REPORT="$(absolute_path "$REPORT")"
    cp "$report_path" "$REPORT"
    echo "release-e2e: report copied to $REPORT"
fi

echo "release-e2e: PASSED"
cat "$report_path"
