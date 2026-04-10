#!/usr/bin/env bash
# Reproducible comparison: inferrs vs ollama Ollama-compatible API.
# Prerequisites: ollama serving gemma4:e2b on :11434, inferrs on :11435.
set -uo pipefail

OLLAMA="${OLLAMA:-http://localhost:11434}"
INFERRS="${INFERRS:-http://localhost:11435}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:e2b}"
INFERRS_MODEL="${INFERRS_MODEL:-google/gemma-4-E2B-it}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

pass=0
fail=0

check_schema() {
  local resp="$1"
  local src="$2"
  local d_reason e_count p_count total_d
  d_reason=$(echo "$resp" | jq -r '.done_reason // ""')
  e_count=$(echo "$resp" | jq -r '.eval_count // 0')
  p_count=$(echo "$resp" | jq -r '.prompt_eval_count // 0')
  total_d=$(echo "$resp" | jq -r '.total_duration // 0')
  
  if [[ -z "$d_reason" || "$e_count" == "0" || "$p_count" == "0" || "$total_d" == "0" ]]; then
    printf "  ${RED}ISSUE${NC}: %s missing/zero metrics fields (d_reason='%s', e_count=%s, p_count=%s, total_d=%s)\n" "$src" "$d_reason" "$e_count" "$p_count" "$total_d"
    return 1
  fi
  return 0
}

# ─── Test: Non-streaming /api/chat ───────────────────────────────────────────
test_chat_nonstream() {
  local label="$1"
  local prompt="$2"
  local think_flag="$3"
  printf "\n${YELLOW}━━━ %s ━━━${NC}\n" "$label"

  local filter
  if [[ "$think_flag" == "default" ]]; then
    filter='{model:$m,stream:false,messages:[{role:"user",content:$p}],options:{temperature:0.01,num_predict:256}}'
  elif [[ "$think_flag" == "true" ]]; then
    filter='{model:$m,stream:false,think:true,messages:[{role:"user",content:$p}],options:{temperature:0.01,num_predict:256}}'
  else
    filter='{model:$m,stream:false,think:false,messages:[{role:"user",content:$p}],options:{temperature:0.01,num_predict:256}}'
  fi

  local o_req i_req
  o_req=$(jq -nc --arg m "$OLLAMA_MODEL" --arg p "$prompt" "$filter")
  i_req=$(jq -nc --arg m "$INFERRS_MODEL" --arg p "$prompt" "$filter")

  local ollama_resp inferrs_resp
  ollama_resp=$(curl -sf --max-time 120 -H "Content-Type: application/json" "${OLLAMA}/api/chat" -d "$o_req" 2>&1)
  inferrs_resp=$(curl -sf --max-time 120 -H "Content-Type: application/json" "${INFERRS}/api/chat" -d "$i_req" 2>&1)

  local o_content o_thinking i_content i_thinking
  o_content=$(echo "$ollama_resp" | jq -r '.message.content // "(empty)"')
  o_thinking=$(echo "$ollama_resp" | jq -r '.message.thinking // "(null)"')
  i_content=$(echo "$inferrs_resp" | jq -r '.message.content // "(empty)"')
  i_thinking=$(echo "$inferrs_resp" | jq -r '.message.thinking // "(null)"')

  printf "  ${CYAN}Ollama${NC}:  content=%s\n" "$(echo "$o_content" | head -c 100)"
  printf "           thinking=%s\n" "$(echo "$o_thinking" | head -c 100 | tr '\n' ' ')"
  printf "  ${CYAN}Inferrs${NC}: content=%s\n" "$(echo "$i_content" | head -c 100)"
  printf "           thinking=%s\n" "$(echo "$i_thinking" | head -c 100 | tr '\n' ' ')"

  local failures=0

  # Check 1: Valid schema metrics
  if ! check_schema "$inferrs_resp" "inferrs"; then
    ((failures++))
  fi

  # Check 2: Thinking field logic
  if [[ "$think_flag" == "true" || "$think_flag" == "default" ]]; then
    # In ollama, default behavior is to extract thinking if the model produces it (e2b always does for math)
    if [[ "$o_thinking" != "(null)" && "$i_thinking" == "(null)" ]]; then
      printf "  ${RED}ISSUE${NC}: inferrs missing 'thinking' field (ollama has it)\n"
      ((failures++))
    elif [[ "$i_thinking" != "(null)" && ${#i_thinking} -lt 15 ]]; then
      printf "  ${RED}ISSUE${NC}: inferrs thinking field is suspiciously short (%d chars). Did it really think?\n" "${#i_thinking}"
      ((failures++))
    fi
  elif [[ "$think_flag" == "false" ]]; then
    if [[ "$i_thinking" != "(null)" && -n "$i_thinking" ]]; then
      printf "  ${RED}ISSUE${NC}: inferrs returned 'thinking' field despite think:false\n"
      ((failures++))
    fi
  fi

  # Check 3: Content quality (detect degeneration)
  local degen_count
  degen_count=$(printf '%s\n' "$i_content" | tr -cs '[:alnum:]_' '\n' | grep -E '.' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
  if [[ "${degen_count:-0}" -gt 40 ]]; then
    printf "  ${RED}ISSUE${NC}: inferrs output degenerated (repeated word %s times)\n" "$degen_count"
    ((failures++))
  fi

  if [[ $failures -gt 0 ]]; then
    printf "  ${RED}FAIL${NC} (%d issues)\n" "$failures"
    ((fail++))
  else
    printf "  ${GREEN}PASS${NC}\n"
    ((pass++))
  fi
}

# ─── Test: Streaming /api/chat ───────────────────────────────────────────────
test_chat_stream() {
  local label="$1"
  local prompt="$2"
  printf "\n${YELLOW}━━━ %s ━━━${NC}\n" "$label"

  local tmpdir
  tmpdir=$(mktemp -d)

  # Capture streaming NDJSON lines
  curl -sN --max-time 120 -H "Content-Type: application/json" \
    "${OLLAMA}/api/chat" -d "$(jq -nc --arg m "$OLLAMA_MODEL" --arg p "$prompt" \
    '{model:$m,stream:true,think:true,messages:[{role:"user",content:$p}],options:{temperature:0.01,num_predict:128}}')" \
    > "$tmpdir/ollama_stream.ndjson" 2>&1 &
  local ollama_pid=$!

  curl -sN --max-time 120 -H "Content-Type: application/json" \
    "${INFERRS}/api/chat" -d "$(jq -nc --arg m "$INFERRS_MODEL" --arg p "$prompt" \
    '{model:$m,stream:true,think:true,messages:[{role:"user",content:$p}],options:{temperature:0.01,num_predict:128}}')" \
    > "$tmpdir/inferrs_stream.ndjson" 2>&1 &
  local inferrs_pid=$!

  wait $ollama_pid 2>/dev/null || true
  wait $inferrs_pid 2>/dev/null || true

  # Reassemble content
  local o_content o_thinking i_content i_thinking
  o_content=$(jq -rj '.message.content // empty' < "$tmpdir/ollama_stream.ndjson" 2>/dev/null)
  o_thinking=$(jq -rj '.message.thinking // empty' < "$tmpdir/ollama_stream.ndjson" 2>/dev/null)
  i_content=$(jq -rj '.message.content // empty' < "$tmpdir/inferrs_stream.ndjson" 2>/dev/null)
  i_thinking=$(jq -rj '.message.thinking // empty' < "$tmpdir/inferrs_stream.ndjson" 2>/dev/null)

  # Check for thinking field in stream chunks
  local o_has_thinking i_has_thinking
  o_has_thinking=$(grep -c '"thinking"' "$tmpdir/ollama_stream.ndjson" 2>/dev/null || echo "0")
  i_has_thinking=$(grep -c '"thinking"' "$tmpdir/inferrs_stream.ndjson" 2>/dev/null || echo "0")

  printf "  ${CYAN}Ollama${NC}:  content=%s\n" "$(echo "$o_content" | head -c 100)"
  printf "           thinking_chunks=%s, thinking=%s\n" "$o_has_thinking" "$(echo "$o_thinking" | head -c 50 | tr '\n' ' ')"
  printf "  ${CYAN}Inferrs${NC}: content=%s\n" "$(echo "$i_content" | head -c 100)"
  printf "           thinking_chunks=%s, thinking=%s\n" "$i_has_thinking" "$(echo "$i_thinking" | head -c 50 | tr '\n' ' ')"

  local failures=0

  # Check 1: Final token schema
  local last_token
  last_token=$(tail -n 1 "$tmpdir/inferrs_stream.ndjson" 2>/dev/null)
  if ! check_schema "$last_token" "inferrs (stream final)"; then
    ((failures++))
  fi

  # Check 2: thinking field present in stream and sane length
  if [[ "$o_has_thinking" -gt 0 && "$i_has_thinking" -eq 0 ]]; then
    printf "  ${RED}ISSUE${NC}: inferrs stream missing 'thinking' field (ollama has %s chunks with it)\n" "$o_has_thinking"
    ((failures++))
  elif [[ "$i_has_thinking" -gt 0 && "$i_has_thinking" -lt 5 ]]; then
    printf "  ${RED}ISSUE${NC}: inferrs thinking stream suspiciously short (%d chunks). Real reasoning expected.\n" "$i_has_thinking"
    ((failures++))
  fi

  # Check 3: degeneration
  local degen_count
  degen_count=$(printf '%s\n' "$i_content" | tr -cs '[:alnum:]_' '\n' | grep -E '.' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
  if [[ "${degen_count:-0}" -gt 40 ]]; then
    printf "  ${RED}ISSUE${NC}: inferrs stream output degenerated (repeated word %s times)\n" "$degen_count"
    ((failures++))
  fi

  # Check 4: empty content (only an issue if there's also no thinking output)
  if [[ (-z "$i_content" || "$i_content" == "(empty)") && "$i_has_thinking" -eq 0 ]]; then
    printf "  ${RED}ISSUE${NC}: inferrs returned empty content (and no thinking)\n"
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

  local o_req i_req
  o_req=$(jq -nc --arg m "$OLLAMA_MODEL" --arg p "$prompt" --argjson s "$stream" '{model:$m,stream:$s,prompt:$p,options:{temperature:0.01,num_predict:64}}')
  i_req=$(jq -nc --arg m "$INFERRS_MODEL" --arg p "$prompt" --argjson s "$stream" '{model:$m,stream:$s,prompt:$p,options:{temperature:0.01,num_predict:64}}')

  local ollama_resp inferrs_resp
  if [[ "$stream" == "false" ]]; then
    ollama_resp=$(curl -sf --max-time 60 -H "Content-Type: application/json" "${OLLAMA}/api/generate" -d "$o_req" 2>&1)
    inferrs_resp=$(curl -sf --max-time 60 -H "Content-Type: application/json" "${INFERRS}/api/generate" -d "$i_req" 2>&1)

    local o_content i_content failures=0
    o_content=$(echo "$ollama_resp" | jq -r '.response // "(empty)"')
    i_content=$(echo "$inferrs_resp" | jq -r '.response // "(empty)"')
    
    printf "  ${CYAN}Ollama${NC}:  content=%s\n" "$(echo "$o_content" | head -c 100)"
    printf "  ${CYAN}Inferrs${NC}: content=%s\n" "$(echo "$i_content" | head -c 100)"

    if ! check_schema "$inferrs_resp" "inferrs (generate)"; then ((failures++)); fi
    if [[ -z "$i_content" || "$i_content" == "(empty)" ]]; then
      printf "  ${RED}ISSUE${NC}: inferrs returned empty response\n"
      ((failures++))
    fi

    if [[ $failures -gt 0 ]]; then ((fail++)); printf "  ${RED}FAIL${NC} (%d issues)\n" "$failures"; else ((pass++)); printf "  ${GREEN}PASS${NC}\n"; fi
  else
    local tmpdir
    tmpdir=$(mktemp -d)

    curl -sN --max-time 60 -H "Content-Type: application/json" "${OLLAMA}/api/generate" -d "$o_req" > "$tmpdir/ollama_stream.ndjson" 2>&1 &
    local ollama_pid=$!
    curl -sN --max-time 60 -H "Content-Type: application/json" "${INFERRS}/api/generate" -d "$i_req" > "$tmpdir/inferrs_stream.ndjson" 2>&1 &
    local inferrs_pid=$!
    
    wait $ollama_pid 2>/dev/null || true
    wait $inferrs_pid 2>/dev/null || true

    local o_content i_content failures=0
    o_content=$(jq -rj '.response // empty' < "$tmpdir/ollama_stream.ndjson" 2>/dev/null)
    i_content=$(jq -rj '.response // empty' < "$tmpdir/inferrs_stream.ndjson" 2>/dev/null)

    printf "  ${CYAN}Ollama${NC}:  content=%s\n" "$(echo "$o_content" | head -c 100)"
    printf "  ${CYAN}Inferrs${NC}: content=%s\n" "$(echo "$i_content" | head -c 100)"

    local last_token
    last_token=$(tail -n 1 "$tmpdir/inferrs_stream.ndjson" 2>/dev/null)
    if ! check_schema "$last_token" "inferrs (generate stream final)"; then ((failures++)); fi
    if [[ -z "$i_content" ]]; then
      printf "  ${RED}ISSUE${NC}: inferrs empty streaming response\n"
      ((failures++))
    fi

    if [[ $failures -gt 0 ]]; then ((fail++)); printf "  ${RED}FAIL${NC} (%d issues)\n" "$failures"; else ((pass++)); printf "  ${GREEN}PASS${NC}\n"; fi
    rm -rf "$tmpdir"
  fi
}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  inferrs vs ollama — Ollama API Compatibility Test Suite  ║"
echo "║  Ollama:  ${OLLAMA}  model: ${OLLAMA_MODEL}              ║"
echo "║  Inferrs: ${INFERRS}  model: ${INFERRS_MODEL}  ║"
echo "╚════════════════════════════════════════════════════════════╝"

# ── Non-streaming chat ───────────────────────────────────────────────────────
test_chat_nonstream "CHAT non-stream: default logic (math)" "What is 2+2? Explain your reasoning." "default"
test_chat_nonstream "CHAT non-stream: think=true (math)" "Solve step by step: What is the derivative of f(x) = x^3 + 2x^2 - 5x + 3?" "true"
test_chat_nonstream "CHAT non-stream: think=false (math)" "Solve step by step: What is the integral of 2x?" "false"
test_chat_nonstream "CHAT non-stream: think=true (greeting)" "Hello, how are you today?" "true"
test_chat_nonstream "CHAT non-stream: think=false (code)" "Write a Python function that returns the factorial of n." "false"

# ── Streaming chat ───────────────────────────────────────────────────────────
test_chat_stream "CHAT stream: math" "Solve step by step: What is the derivative of x^2?"
test_chat_stream "CHAT stream: code" "Write a fast inverse square root in C."

# ── Generate endpoint ────────────────────────────────────────────────────────
test_generate "GENERATE non-stream" "What is 2+2?" false
test_generate "GENERATE stream"     "Write a greeting to a user." true

echo ""
echo "════════════════════════════════════════════════════════════"
printf "Results: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}\n" "$pass" "$fail"
echo "════════════════════════════════════════════════════════════"
exit $fail
