#!/usr/bin/env bash
# benchmark.sh — Compare llama-server vs inferrs serve vs inferrs serve --turbo-quant --quantize
# for google/gemma-4-E2B-it using the Decode, Prefill, and TTFT metrics.
#
# Usage:
#   ./benchmark.sh [--runs N] [--prompt-len N] [--max-tokens N] [--warmup N]
#
# Requirements:
#   - llama-server  (from llama.cpp, must be on PATH)
#   - inferrs       (must be on PATH)
#   - curl, python3
#
# The script:
#   1. Starts `inferrs serve` for both inferrs variants, sends timed requests, then stops it
#   2. Starts llama-server, sends timed /v1/chat/completions requests, then stops it
#   3. Prints a summary table

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
RUNS=5
WARMUP=1
PROMPT_LEN=128
MAX_TOKENS=128
LLAMA_PORT=8181
INFERRS_PORT=8080
INFERRS_TQ_PORT=8082
LLAMA_MODEL="ggml-org/gemma-4-E2B-it-GGUF"
INFERRS_MODEL="google/gemma-4-E2B-it"
SERVER_READY_TIMEOUT=120   # seconds to wait for server /health

# Use the locally built inferrs binary if it exists; fall back to PATH.
INFERRS_BIN="${INFERRS_BIN:-$(dirname "${BASH_SOURCE[0]}")/../target/release/inferrs}"
if [[ ! -x "$INFERRS_BIN" ]]; then
  INFERRS_BIN="inferrs"
fi

# ── parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)        RUNS="$2";       shift 2 ;;
    --warmup)      WARMUP="$2";     shift 2 ;;
    --prompt-len)  PROMPT_LEN="$2"; shift 2 ;;
    --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ── helpers ─────────────────────────────────────────────────────────────────
log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

wait_for_health() {
  local url="$1"
  local timeout="$2"
  local start=$SECONDS
  printf "    Waiting for %s " "$url"
  while true; do
    if curl -sf "$url" > /dev/null 2>&1; then
      printf ' ready\n'
      return 0
    fi
    if (( SECONDS - start > timeout )); then
      printf ' TIMEOUT\n'
      return 1
    fi
    printf '.'
    sleep 2
  done
}

kill_server() {
  local pid="$1"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

# Build a synthetic prompt of approximately PROMPT_LEN tokens.
# Simple heuristic: ~4 chars per token.
PROMPT_CHARS=$(( PROMPT_LEN * 4 ))
SYNTHETIC_PROMPT=$(python3 -c "
import random
words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'a', 'lazy', 'dog',
         'machine', 'learning', 'model', 'performance', 'benchmark', 'inference',
         'speed', 'latency', 'throughput', 'token', 'generation', 'prefill',
         'decode', 'attention', 'transformer', 'neural', 'network', 'parameter']
out = ' '.join(random.choices(words, k=${PROMPT_LEN}))
print(out[:${PROMPT_CHARS}])
")

# ── unified benchmark helper ──────────────────────────────────────────────────
# Works against any OpenAI-compatible server. Methodology (identical for all):
#   - One non-streaming probe to get the true prompt token count (n_prompt).
#   - SSE streaming for each run: wall-clock TTFT = time to first content chunk;
#     n_gen = number of content chunks (one chunk ≈ one token).
#   - prefill = n_prompt / ttft
#   - decode  = n_gen    / (total_elapsed - ttft)
bench_http() {
  local host="$1"
  local port="$2"
  local warmup="$3"
  local runs="$4"

  python3 - "$host" "$port" "$warmup" "$runs" "$MAX_TOKENS" "$SYNTHETIC_PROMPT" <<'PYEOF'
import sys, json, urllib.request, time, statistics, os

host, port, warmup, runs, max_tokens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
url = f"http://{host}:{port}/v1/chat/completions"
prompt = sys.argv[6] if len(sys.argv) > 6 else "Write a short story about a robot."

def http_post(payload_dict):
    payload = json.dumps(payload_dict).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())

# One non-streaming probe to get the true prompt token count (constant for our prompt).
probe = http_post({
    "model": "benchmark",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1,
    "temperature": 0.0,
    "stream": False,
})
n_prompt = probe.get("usage", {}).get("prompt_tokens", 0)

def do_stream(prompt_text):
    """Stream one request; return (ttft_ms, total_ms, n_gen)."""
    payload = json.dumps({
        "model": "benchmark",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft_ms = None
    n_gen = 0

    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            # Count any generated token: content or reasoning_content (thinking models
            # like Gemma-4 emit reasoning_content instead of content for CoT tokens).
            token = delta.get("content") or delta.get("reasoning_content") or ""
            if token:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                n_gen += 1  # each non-empty chunk ≈ 1 token

    total_ms = (time.perf_counter() - t0) * 1000
    if ttft_ms is None:
        ttft_ms = total_ms
    return ttft_ms, total_ms, n_gen

ttfts, prefills, decodes = [], [], []
for i in range(warmup + runs):
    ttft, total_ms, n_gen = do_stream(prompt)
    decode_ms   = max(total_ms - ttft, 1e-6)
    prefill_tps = (n_prompt / ttft      * 1000) if ttft      > 0 else 0.0
    decode_tps  = (n_gen    / decode_ms * 1000) if decode_ms > 0 else 0.0

    if i < warmup:
        print(f"  [warmup] TTFT={ttft:.1f}ms  prefill={prefill_tps:.1f}t/s  decode={decode_tps:.1f}t/s  (n_prompt={n_prompt}, n_gen={n_gen})")
        continue

    run = i - warmup + 1
    ttfts.append(ttft);  prefills.append(prefill_tps);  decodes.append(decode_tps)
    print(f"  [run {run}/{runs}] TTFT={ttft:.1f}ms  prefill={prefill_tps:.1f}t/s  decode={decode_tps:.1f}t/s  (n_prompt={n_prompt}, n_gen={n_gen})")

def stats(vals, unit):
    if not vals:
        return f"N/A {unit}"
    return f"{statistics.mean(vals):.2f} ± {(statistics.stdev(vals) if len(vals) > 1 else 0):.2f} {unit}"

print()
print(f"  TTFT    : {stats(ttfts,    'ms')}")
print(f"  Prefill : {stats(prefills, 'tok/s')}")
print(f"  Decode  : {stats(decodes,  'tok/s')}")

summary = {
    "ttft_ms":     statistics.mean(ttfts)    if ttfts    else None,
    "prefill_tps": statistics.mean(prefills) if prefills else None,
    "decode_tps":  statistics.mean(decodes)  if decodes  else None,
}
summary_file = os.environ.get("BENCH_SUMMARY_FILE")
if summary_file:
    with open(summary_file, "w") as f:
        json.dump(summary, f)
PYEOF
}

# ── summary storage ──────────────────────────────────────────────────────────
TMPDIR_BENCH=$(mktemp -d)
INFERRS_PID=""
INFERRS_TQ_PID=""
LLAMA_PID=""
cleanup() {
  kill_server "${INFERRS_PID}"
  kill_server "${INFERRS_TQ_PID}"
  kill_server "${LLAMA_PID}"
  rm -rf "$TMPDIR_BENCH"
}
trap cleanup EXIT

SUMMARY_LLAMA="$TMPDIR_BENCH/llama.json"
SUMMARY_INFERRS="$TMPDIR_BENCH/inferrs.json"
SUMMARY_INFERRS_TQ="$TMPDIR_BENCH/inferrs_tq.json"

# ════════════════════════════════════════════════════════════════════════════
# 1. inferrs serve --quantize (plain bf16)
# ════════════════════════════════════════════════════════════════════════════
log "Benchmark 1/3 — inferrs serve --quantize $INFERRS_MODEL (plain bf16)"

INFERRS_LOG="$TMPDIR_BENCH/inferrs-serve.log"

"$INFERRS_BIN" serve "$INFERRS_MODEL" \
  --host 127.0.0.1 \
  --port "$INFERRS_PORT" \
  --quantize \
  > "$INFERRS_LOG" 2>&1 &
INFERRS_PID=$!
ok "inferrs serve --quantize started (pid $INFERRS_PID)"

if ! wait_for_health "http://127.0.0.1:${INFERRS_PORT}/health" "$SERVER_READY_TIMEOUT"; then
  err "inferrs serve --quantize failed to start. Log tail:"
  tail -20 "$INFERRS_LOG" >&2
  kill_server "$INFERRS_PID"
  exit 1
fi

BENCH_SUMMARY_FILE="$SUMMARY_INFERRS" \
  bench_http "127.0.0.1" "$INFERRS_PORT" "$WARMUP" "$RUNS"

kill_server "$INFERRS_PID"
ok "inferrs serve --quantize stopped"

# ════════════════════════════════════════════════════════════════════════════
# 2. inferrs serve --turbo-quant --quantize
# ════════════════════════════════════════════════════════════════════════════
log "Benchmark 2/3 — inferrs serve --turbo-quant --quantize $INFERRS_MODEL"

INFERRS_TQ_LOG="$TMPDIR_BENCH/inferrs-serve-tq.log"

"$INFERRS_BIN" serve "$INFERRS_MODEL" \
  --host 127.0.0.1 \
  --port "$INFERRS_TQ_PORT" \
  --turbo-quant \
  --quantize \
  > "$INFERRS_TQ_LOG" 2>&1 &
INFERRS_TQ_PID=$!
ok "inferrs serve --turbo-quant --quantize started (pid $INFERRS_TQ_PID)"

if ! wait_for_health "http://127.0.0.1:${INFERRS_TQ_PORT}/health" "$SERVER_READY_TIMEOUT"; then
  err "inferrs serve --turbo-quant --quantize failed to start. Log tail:"
  tail -20 "$INFERRS_TQ_LOG" >&2
  kill_server "$INFERRS_TQ_PID"
  exit 1
fi

BENCH_SUMMARY_FILE="$SUMMARY_INFERRS_TQ" \
  bench_http "127.0.0.1" "$INFERRS_TQ_PORT" "$WARMUP" "$RUNS"

kill_server "$INFERRS_TQ_PID"
ok "inferrs serve --turbo-quant --quantize stopped"

# ════════════════════════════════════════════════════════════════════════════
# 3. llama-server -hf ggml-org/gemma-4-E2B-it-GGUF
# ════════════════════════════════════════════════════════════════════════════
log "Benchmark 3/3 — llama-server -hf $LLAMA_MODEL"

LLAMA_LOG="$TMPDIR_BENCH/llama-server.log"

# Start llama-server in background
llama-server \
  -hf "$LLAMA_MODEL" \
  --host 127.0.0.1 \
  --port "$LLAMA_PORT" \
  > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
ok "llama-server started (pid $LLAMA_PID)"

if ! wait_for_health "http://127.0.0.1:${LLAMA_PORT}/health" "$SERVER_READY_TIMEOUT"; then
  err "llama-server failed to start. Log tail:"
  tail -20 "$LLAMA_LOG" >&2
  kill_server "$LLAMA_PID"
  exit 1
fi

BENCH_SUMMARY_FILE="$SUMMARY_LLAMA" \
  bench_http "127.0.0.1" "$LLAMA_PORT" "$WARMUP" "$RUNS"

kill_server "$LLAMA_PID"
ok "llama-server stopped"

# ════════════════════════════════════════════════════════════════════════════
# Summary table
# ════════════════════════════════════════════════════════════════════════════
log "Results"
python3 - "$SUMMARY_LLAMA" "$SUMMARY_INFERRS" "$SUMMARY_INFERRS_TQ" \
          "$RUNS" "$WARMUP" "$PROMPT_LEN" "$MAX_TOKENS" <<'PYEOF'
import sys, json

llama_f, inferrs_f, inferrs_tq_f, runs, warmup, prompt_len, max_tokens = sys.argv[1:]

def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}

llama     = load(llama_f)
inferrs   = load(inferrs_f)
inferrs_tq = load(inferrs_tq_f)

def fmt(v, unit=""):
    if v is None:
        return "N/A"
    return f"{v:.2f}{' ' + unit if unit else ''}"

rows = [
    ("llama-server -hf ggml-org/gemma-4-E2B-it-GGUF (Q4_K_M)",
     llama.get("ttft_ms"), llama.get("prefill_tps"), llama.get("decode_tps")),
    ("inferrs serve --quantize google/gemma-4-E2B-it",
     inferrs.get("ttft_ms"), inferrs.get("prefill_tps"), inferrs.get("decode_tps")),
    ("inferrs serve --turbo-quant --quantize google/gemma-4-E2B-it",
     inferrs_tq.get("ttft_ms"), inferrs_tq.get("prefill_tps"), inferrs_tq.get("decode_tps")),
]

W = 56
print()
print(f"Benchmark settings: prompt_len={prompt_len} tokens, max_tokens={max_tokens}, runs={runs}, warmup={warmup}")
print()
print(f"{'Backend':<{W}}  {'TTFT (ms)':>12}  {'Prefill (t/s)':>14}  {'Decode (t/s)':>13}")
print("-" * (W + 45))
for name, ttft, pfill, dec in rows:
    print(f"{name:<{W}}  {fmt(ttft,'ms'):>12}  {fmt(pfill,'t/s'):>14}  {fmt(dec,'t/s'):>13}")
print()
# relative comparison vs llama
base_ttft   = llama.get("ttft_ms")
base_pfill  = llama.get("prefill_tps")
base_dec    = llama.get("decode_tps")
if base_ttft and base_dec:
    print("Relative to llama-server (higher prefill/decode is better; lower TTFT is better):")
    for name, ttft, pfill, dec in rows[1:]:
        if ttft and pfill and dec:
            d_ttft  = (ttft   - base_ttft)  / base_ttft  * 100
            d_pfill = (pfill  - base_pfill) / base_pfill * 100
            d_dec   = (dec    - base_dec)   / base_dec   * 100
            sign = lambda x: "+" if x >= 0 else ""
            print(f"  {name}")
            print(f"    TTFT:    {sign(d_ttft)}{d_ttft:.1f}%")
            print(f"    Prefill: {sign(d_pfill)}{d_pfill:.1f}%")
            print(f"    Decode:  {sign(d_dec)}{d_dec:.1f}%")
PYEOF
