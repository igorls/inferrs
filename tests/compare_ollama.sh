#!/usr/bin/env bash
# Reproducible comparison: inferrs vs ollama Ollama-compatible API.
# Prerequisites: ollama serving gemma4:e2b on :11434, inferrs on :11435.
set -uo pipefail

OLLAMA="http://localhost:11434"
INFERRS="http://localhost:11435"
OLLAMA_MODEL="gemma4:e2b"
INFERRS_MODEL="google/gemma-4-E2B-it"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

pass=0
fail=0

# ─── Test: Non-streaming /api/chat ───────────────────────────────────────────
test_chat_nonstream() {
  local label="$1"
  local prompt="$2"
  printf "\n${YELLOW}━━━ %s ━━━${NC}\n" "$label"

  local ollama_resp inferrs_resp
  ollama_resp=$(curl -sf --max-time 60 -H "Content-Type: application/json" \
    "${OLLAMA}/api/chat" -d "$(jq -nc --arg m "$OLLAMA_MODEL" --arg p "$prompt" \
    '{model:$m,stream:false,messages:[{role:"user",content:$p}],options:{temperature:0.01,num_predict:256}}')" 2>&1)
  inferrs_resp=$(curl -sf --max-time 60 -H "Content-Type: application/json" \
    "${INFERRS}/api/chat" -d "$(jq -nc --arg m "$INFERRS_MODEL" --arg p "$prompt" \
    '{model:$m,stream:false,messages:[{role:"user",content:$p}],options:{temperature:0.01,num_predict:256}}')" 2>&1)

  local o_content o_thinking i_content i_thinking
  o_content=$(echo "$ollama_resp" | jq -r '.message.content // "(empty)"')
  o_thinking=$(echo "$ollama_resp" | jq -r '.message.thinking // "(null)"' | head -c 120)
  i_content=$(echo "$inferrs_resp" | jq -r '.message.content // "(empty)"')
  i_thinking=$(echo "$inferrs_resp" | jq -r '.message.thinking // "(null)"' | head -c 120)

  printf "  ${CYAN}Ollama${NC}:  content=%s\n" "$(echo "$o_content" | head -c 150)"
  printf "           thinking=%s\n" "$o_thinking"
  printf "  ${CYAN}Inferrs${NC}: content=%s\n" "$(echo "$i_content" | head -c 150)"
  printf "           thinking=%s\n" "$i_thinking"

  # Check 1: thinking field exists in ollama but missing in inferrs
  if [[ "$o_thinking" != "(null)" && "$i_thinking" == "(null)" ]]; then
    printf "  ${RED}FAIL${NC}: inferrs missing 'thinking' field (ollama has it)\n"
    ((fail++))
    return
  fi

  # Check 2: content quality — detect degeneration
  local degen_count
  degen_count=$(echo "$i_content" | grep -oP '\b(\w+)\b' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
  if [[ "${degen_count:-0}" -gt 8 ]]; then
    printf "  ${RED}FAIL${NC}: inferrs output degenerated\n"
    ((fail++))
    return
  fi

  printf "  ${GREEN}PASS${NC}\n"
  ((pass++))
}

# ─── Test: Streaming /api/chat ───────────────────────────────────────────────
test_chat_stream() {
  local label="$1"
  local prompt="$2"
  printf "\n${YELLOW}━━━ %s ━━━${NC}\n" "$label"

  local tmpdir
  tmpdir=$(mktemp -d)

  # Capture streaming NDJSON lines
  curl -sN --max-time 30 -H "Content-Type: application/json" \
    "${OLLAMA}/api/chat" -d "$(jq -nc --arg m "$OLLAMA_MODEL" --arg p "$prompt" \
    '{model:$m,stream:true,messages:[{role:"user",content:$p}],options:{temperature:0.01,num_predict:128}}')" \
    > "$tmpdir/ollama_stream.ndjson" 2>&1 &
  local ollama_pid=$!

  curl -sN --max-time 30 -H "Content-Type: application/json" \
    "${INFERRS}/api/chat" -d "$(jq -nc --arg m "$INFERRS_MODEL" --arg p "$prompt" \
    '{model:$m,stream:true,messages:[{role:"user",content:$p}],options:{temperature:0.01,num_predict:128}}')" \
    > "$tmpdir/inferrs_stream.ndjson" 2>&1 &
  local inferrs_pid=$!

  wait $ollama_pid 2>/dev/null || true
  wait $inferrs_pid 2>/dev/null || true

  # Reassemble content
  local o_content o_thinking i_content i_thinking
  o_content=$(jq -rj '.message.content // empty' < "$tmpdir/ollama_stream.ndjson" 2>/dev/null)
  o_thinking=$(jq -rj '.message.thinking // empty' < "$tmpdir/ollama_stream.ndjson" 2>/dev/null | head -c 120)
  i_content=$(jq -rj '.message.content // empty' < "$tmpdir/inferrs_stream.ndjson" 2>/dev/null)
  i_thinking=$(jq -rj '.message.thinking // empty' < "$tmpdir/inferrs_stream.ndjson" 2>/dev/null | head -c 120)

  # Check for thinking field in stream chunks
  local o_has_thinking i_has_thinking
  o_has_thinking=$(grep -c '"thinking"' "$tmpdir/ollama_stream.ndjson" 2>/dev/null || echo "0")
  i_has_thinking=$(grep -c '"thinking"' "$tmpdir/inferrs_stream.ndjson" 2>/dev/null || echo "0")

  printf "  ${CYAN}Ollama${NC}:  content=%s\n" "$(echo "$o_content" | head -c 150)"
  printf "           thinking_chunks=%s, thinking=%s...\n" "$o_has_thinking" "$(echo "$o_thinking" | head -c 80)"
  printf "  ${CYAN}Inferrs${NC}: content=%s\n" "$(echo "$i_content" | head -c 150)"
  printf "           thinking_chunks=%s, thinking=%s...\n" "$i_has_thinking" "$(echo "$i_thinking" | head -c 80)"

  local failures=0

  # Check 1: thinking field present in stream
  if [[ "$o_has_thinking" -gt 0 && "$i_has_thinking" -eq 0 ]]; then
    printf "  ${RED}ISSUE${NC}: inferrs stream missing 'thinking' field (ollama has %s chunks with it)\n" "$o_has_thinking"
    ((failures++))
  fi

  # Check 2: degeneration
  local degen_count
  degen_count=$(echo "$i_content" | grep -oP '\b(\w+)\b' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
  if [[ "${degen_count:-0}" -gt 8 ]]; then
    printf "  ${RED}ISSUE${NC}: inferrs stream output degenerated (repeated word %s times)\n" "$degen_count"
    ((failures++))
  fi

  # Check 3: empty content
  if [[ -z "$i_content" || "$i_content" == "(empty)" ]]; then
    printf "  ${RED}ISSUE${NC}: inferrs returned empty content\n"
    ((failures++))
  fi

  if [[ $failures -gt 0 ]]; then
    printf "  ${RED}FAIL${NC} (%d issues)\n" "$failures"
    ((fail++))
  else
    printf "  ${GREEN}PASS${NC}\n"
    ((pass++))
  fi

  rm -rf "$tmpdir"
}

# ─── Test: /api/generate endpoint ────────────────────────────────────────────
test_generate() {
  local label="$1"
  local prompt="$2"
  local stream="$3"
  printf "\n${YELLOW}━━━ %s ━━━${NC}\n" "$label"

  local inferrs_resp
  inferrs_resp=$(curl -sf --max-time 60 -H "Content-Type: application/json" \
    "${INFERRS}/api/generate" -d "$(jq -nc --arg m "$INFERRS_MODEL" --arg p "$prompt" --argjson s "$stream" \
    '{model:$m,stream:$s,prompt:$p,options:{temperature:0.01,num_predict:64}}')" 2>&1) || {
    printf "  ${RED}FAIL${NC}: request failed (404 or error)\n"
    ((fail++))
    return
  }

  if [[ "$stream" == "false" ]]; then
    local content
    content=$(echo "$inferrs_resp" | jq -r '.response // "(empty)"')
    printf "  Inferrs: %s\n" "$(echo "$content" | head -c 200)"
    if [[ -z "$content" || "$content" == "(empty)" ]]; then
      printf "  ${RED}FAIL${NC}: empty response\n"
      ((fail++))
    else
      printf "  ${GREEN}PASS${NC}\n"
      ((pass++))
    fi
  else
    local content
    content=$(echo "$inferrs_resp" | jq -rj '.response // empty' 2>/dev/null)
    printf "  Inferrs: %s\n" "$(echo "$content" | head -c 200)"
    if [[ -z "$content" ]]; then
      printf "  ${RED}FAIL${NC}: empty streaming response\n"
      ((fail++))
    else
      printf "  ${GREEN}PASS${NC}\n"
      ((pass++))
    fi
  fi
}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  inferrs vs ollama — Ollama API Compatibility Test Suite  ║"
echo "║  Ollama:  ${OLLAMA}  model: ${OLLAMA_MODEL}              ║"
echo "║  Inferrs: ${INFERRS}  model: ${INFERRS_MODEL}  ║"
echo "╚════════════════════════════════════════════════════════════╝"

# ── Non-streaming chat ───────────────────────────────────────────────────────
test_chat_nonstream "CHAT non-stream: math"     "What is 2+2? Answer in one word."
test_chat_nonstream "CHAT non-stream: greeting" "Hello, how are you today?"
test_chat_nonstream "CHAT non-stream: code"     "Write a Python function that returns the factorial of n."

# ── Streaming chat ───────────────────────────────────────────────────────────
test_chat_stream "CHAT stream: math"     "What is 2+2? Answer in one word."
test_chat_stream "CHAT stream: greeting" "Hello, how are you today?"
test_chat_stream "CHAT stream: code"     "Write a Python function that returns the factorial of n."

# ── Generate endpoint ────────────────────────────────────────────────────────
test_generate "GENERATE non-stream" "What is 2+2?" false
test_generate "GENERATE stream"     "What is 2+2?" true

echo ""
echo "════════════════════════════════════════════════════════════"
printf "Results: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}\n" "$pass" "$fail"
echo "════════════════════════════════════════════════════════════"
exit $fail
