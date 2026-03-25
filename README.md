# inferrs

A fast LLM inference engine. Ships as a single binary that downloads a model
and exposes an OpenAI-compatible API.

## Why inferrs?

Most LLM serving stacks force a trade-off between features and resource usage.
**inferrs** targets the sweet spot for desktop and single-GPU users:

| | inferrs | vLLM | llama.cpp |
|---|---|---|---|
| **Language** | Rust | Python/C++ | C/C++ |
| **Memory strategy** | Grow on demand | Pre-allocates ~90% of GPU memory at startup | Fixed per-request allocation |
| **Continuous batching** | ✓ | ✓ | ✗ |
| **Streaming (SSE)** | ✓ | ✓ | ✓ |
| **Chunked prefill** | ✓ | ✓ | ✗ |
| **KV cache management** | Per-context concatenation/PagedAttention | PagedAttention | Per-context allocation |
| **Multi-GPU / distributed** | Single device | Multi-GPU, tensor parallel | Partial (model splitting) |
| **Desktop friendly** | ✓ — lightweight | ✗ — claims most GPU memory | ✓ — lightweight |
| **Binary footprint** | Single static binary | Python environment + deps | Single binary |

## Features

- **OpenAI-compatible API** — `/v1/completions`, `/v1/chat/completions`,
  `/v1/models`, `/health`
- **Server-Sent Events** streaming
- **Continuous batching** with chunked prefill and preemption
- **Block-based KV cache** with free-list reuse
- **Chat templates** — ChatML and Llama formats
- **Sampling** — temperature, top-k, top-p, repetition penalty
- **HuggingFace Hub** — downloads models and tokenizers on first run
- **Hardware backends** — CPU, CUDA (NVIDIA), Metal (Apple Silicon)
- **Model architectures** — Qwen2, Llama, Mistral, and other standard
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

- **Server** — Axum HTTP server, parses requests, tokenizes input, streams
  tokens back via SSE.
- **Engine** — Owns the model and runs the inference loop on a dedicated thread.
- **Scheduler** — Continuous batching with chunked prefill and preemption.
- **KV Cache** — Block-based, grow-on-demand allocation with free-list reuse.
- **Sampler** — Temperature, top-k, top-p, and repetition penalty.

