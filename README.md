# inferrs

A TurboQuant LLM inference engine. Ships as a single binary that downloads a
model and exposes an OpenAI-compatible API.

## Why inferrs?

Most LLM serving stacks force a trade-off between features and resource usage.
**inferrs** targets both:

| | inferrs | vLLM | llama.cpp |
|---|---|---|---|
| **Language** | Rust | Python/C++ | C/C++ |
| **Streaming (SSE)** | ✓ | ✓ | ✓ |
| **KV cache management** | TurboQuant, Per-context alloc, PagedAttention | PagedAttention | Per-context alloc |
| **Desktop friendly** | ✓ — lightweight | ✗ — claims most GPU memory | ✓ — lightweight |
| **Binary footprint** | Single binary | Python environment + deps | Single binary |

## Features

- **OpenAI-compatible API** — `/v1/completions`, `/v1/chat/completions`,
  `/v1/models`, `/health`
- **Hardware backends** — CPU, CUDA (NVIDIA), Metal (Apple Silicon)
  decoder-only transformers (safetensors format)

## Quick start

### Install

```bash
brew tap ericcurtin/inferrs
brew install inferrs
```

### Build from source

```bash
cargo build
```

Enable GPU acceleration with a feature flag:

```bash
# NVIDIA
cargo build --release --features cuda

# Apple Silicon
cargo build --release --features metal
```

### Run

```bash
# Serve a model (downloads automatically from HuggingFace Hub)
inferrs serve Qwen/Qwen3.5-0.8B

# Specify dtype and port
inferrs serve Qwen/Qwen3.5-0.8B --dtype f16 --port 8080
```

### Query

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

## Architecture

```
┌─────────┐      HTTP       ┌────────┐  channel  ┌────────┐
│  Client │ ──────────────▶ │ Server │ ────────▶ │ Engine │
└─────────┘  (axum + SSE)   └────────┘           └────────┘
                                                     │
                               ┌──────────┬──────────┼──────────┐
                               ▼          ▼          ▼          ▼
                          Scheduler    Transformer  KV Cache  Sampler
```

