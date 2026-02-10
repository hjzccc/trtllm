# Learnings & Conventions

## [TIMESTAMP] Session Start
- Plan: qwen3-moe-precision-benchmark
- Goal: Build benchmark comparing Qwen3 MoE across BF16/FP8/nvFP4 precisions

## [2026-02-06] Created analyze_nsys.py Script

### Implementation Details
- **File**: `benchmarks/qwen3_moe/analyze_nsys.py`
- **Purpose**: Process Nsight Systems traces to categorize CUDA kernels and calculate Amdahl's law ceiling for MoE expert GEMM optimization

### Key Features
1. **CLI Interface**: Uses argparse with `--input` and `--output` flags
2. **Nsys Integration**: Calls `nsys stats --report gputrace --format csv` to extract kernel trace data
3. **Kernel Categorization**: 8 categories with pattern matching (case-insensitive):
   - `expert_gemm`: cutlass/fused_moe/nvfp4_gemm/fp8 kernels
   - `attention_gemm`: flash_attn/fmha/gpt_attention kernels
   - `gate_gemm`: cublas kernels (future: NVTX range filtering)
   - `routing_permute`: topk/sort/scatter/gather/permute/index kernels
   - `quant_dequant`: quantize/dequant/cast/fp4/fp8 quant kernels
   - `norm`: rms_norm/layer_norm kernels
   - `allreduce`: nccl/allreduce/reduce_scatter kernels
   - `other`: everything else
4. **Output Formats**:
   - **Stdout**: Human-readable table with time (ms), percentage, kernel count
   - **JSON**: Detailed breakdown with sample kernel names and Amdahl's law note

### Amdahl's Law Calculation
- Formula: `max_speedup = 1 / (1 - expert_pct/100)`
- Example: If expert_gemm is 52% of total time, max speedup from infinitely fast expert kernels is ~2.08x
- This tells users whether optimizing expert precision is worthwhile given overhead from routing, permute, quant/dequant, allreduce

### CSV Parsing Strategy
- Handles different nsys output formats by looking for header line with "Name" and "Time"/"Duration"
- Extracts kernel name and total time (ns) from CSV
- Robust to comma-separated numbers and quoted strings

### Testing
- CLI help verified: `python benchmarks/qwen3_moe/analyze_nsys.py --help` exits 0
- Script made executable with shebang `#!/usr/bin/env python3`

### Future Enhancements
- NVTX range filtering for gate_gemm (to distinguish from other cublas calls)
- Support for `nsys export --type sqlite` as alternative to CSV parsing
- Visualization/plotting capabilities (currently just tables and JSON)

### Dependencies
- Requires `nsys` CLI tool from Nsight Systems installation
- Standard library only: argparse, csv, json, re, subprocess, tempfile, pathlib, collections

## [2026-02-06] Created bench_utils.py and perf_bench.py

### bench_utils.py — Shared Utilities
- **SM validation**: Uses `_PRECISION_SM_CONSTRAINTS` dict mirroring `_QUANT_SUPPORT_TABLE` from `fused_moe_cutlass.py:68-114`
  - BF16: SM≥80, FP8: SM≥89, nvFP4: SM∈{100,103}
  - Constraint types: `min`, `exact`, `in` — matching the original table's semantics
  - SM version computed as `major*10 + minor` (e.g. (9,0)→90)
- **Model loading**: `load_model()` uses `from tensorrt_llm import LLM` with `backend="pytorch"`, no QuantConfig needed for pre-quantized checkpoints
- **MoE backend detection**: Inspects `llm.model.model.layers[0].mlp.experts.__class__.__name__` — can return CutlassFusedMoE or TRTLLMGenFusedMoE
- **Result serialization**: JSON with UTC timestamp, SM version, GPU name metadata
- **Cleanup**: `del llm` + `gc.collect()` + `torch.cuda.empty_cache()` + `synchronize()`

### perf_bench.py — Performance Benchmark
- **Timing**: `torch.cuda.Event(enable_timing=True)` pattern from `auto_deploy/utils/benchmark.py:87-118`
  - `synchronize()` before `start.record()`, `end.record()`, `synchronize()` after
- **Nsys mode**: Detects `NSYS_PROFILING_SESSION_ID` env var, runs exactly 1 iteration between `cudaProfilerStart/Stop`
  - Uses `enable_layerwise_nvtx_marker=True` via LLM constructor (not manual NVTX markers)
  - Prints usage instructions if not under nsys
- **Deterministic prompts**: 16 fixed seed prompts, repeated/truncated to target input_len (4 chars ≈ 1 token heuristic)
- **Comparison table**: Printed at end in perf mode with precision, backend, avg/min/max latency, tok/s, ms/tok

### Key Findings
- `benchmarks/` had no `__init__.py` at top level — created one (precedent: `benchmarks/cpp/__init__.py` existed)
- `perf_bench.py` includes `sys.path` manipulation for direct execution since `benchmarks/` isn't always a package
- TRT-LLM lazy imports (`from tensorrt_llm import LLM`) deferred to function scope to keep module-level import clean

## [2026-02-06] Created ppl_bench.py — Perplexity Benchmark

### Implementation Details
- **File**: `benchmarks/qwen3_moe/ppl_bench.py`
- **Purpose**: Measure perplexity on WikiText-2 test set across BF16/FP8/nvFP4 precisions

### Architecture
1. **WikiText-2 Loading**: Uses `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")`
   - Concatenates all non-empty lines into one text blob
   - Tokenizes entire corpus at once via `AutoTokenizer` from the first available model path
   - Chunks into non-overlapping windows of `--input_len` tokens
   - Same chunks used for all precisions (deterministic, fair comparison)
2. **Perplexity Formula**: Exact match of `tensorrt_llm/tools/ppl.py`:
   - `nlls = -logits.log_softmax(dim=-1)`
   - `ppls = nlls.gather(-1, output_ids.long().unsqueeze(-1))`
   - `ppl = ppls.mean().exp().item()`
3. **Context Logits**: Uses `SamplingParams(max_tokens=1, return_context_logits=True)` to get logits for all input positions without generating tokens
   - Logits at position i predict token at position i+1, so we use `logits[:-1]` vs `tokens[1:]`
4. **Overall PPL**: Geometric mean of per-chunk PPLs = `exp(mean(log(ppl_i)))`

### Key Design Decisions
- **Tokenizer loaded once from first available model path** — all Qwen3 precisions share the same tokenizer
- **`max_tokens=1`** — we only need context-phase logits, no generation needed
- **Token ID lists passed directly to `llm.generate()`** — avoids double-tokenization
- **Chunks capped via `--num_chunks`** (default 100) to keep runtime bounded
- **JSON output** includes `per_chunk_ppls` list for detailed analysis

### CLI Arguments (PPL-specific)
- `--input_len` (default 512): tokens per chunk
- `--num_chunks` (default 100): max chunks to evaluate
- `--dataset` (default "wikitext"): dataset choice (extensible)

### Testing
- CLI help verified: `python benchmarks/qwen3_moe/ppl_bench.py --help` exits 0
- Follows same sys.path and import pattern as `perf_bench.py`

### Key Insight: context_logits in High-Level LLM API
- `RequestOutput.context_logits` returns `torch.Tensor` of shape `(input_len, vocab_size)`
- Enabled via `SamplingParams(return_context_logits=True)`
- This is different from the old runner API used in `examples/summarize.py` which returns both context_logits and generation_logits separately via `output_generation_logits=True`

