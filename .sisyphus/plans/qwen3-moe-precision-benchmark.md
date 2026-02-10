# Qwen3 MoE Precision Benchmark: BF16 vs FP8 vs nvFP4

## TL;DR

> **Quick Summary**: Build a benchmark script that compares Qwen3-30B-A3B MoE inference across 3 precision levels (BF16, FP8, nvFP4) — measuring both performance (latency, throughput, kernel breakdown) and accuracy (perplexity on WikiText). The goal is to determine whether lower-precision expert GEMMs deliver real end-to-end speedup once routing, permutation, quant/dequant, and allreduce overhead are accounted for.
>
> **Deliverables**:
> - `benchmarks/qwen3_moe/bench_utils.py` — shared utilities (SM validation, model loading helpers, CLI arg builders, JSON result writer)
> - `benchmarks/qwen3_moe/perf_bench.py` — performance benchmark (latency, throughput, nsys profiling mode)
> - `benchmarks/qwen3_moe/ppl_bench.py` — accuracy benchmark (WikiText-2 perplexity measurement)
> - `benchmarks/qwen3_moe/analyze_nsys.py` — helper to parse Nsight Systems traces into kernel category breakdown
> - Structured JSON results per precision config
> - Nsight Systems `.nsys-rep` trace files for kernel-level analysis
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves (4 tasks)
> **Critical Path**: Task 1 → Task 2 → Task 4

---

## Context

### Original Request
User wants to evaluate whether mixed precision quantization for MoE experts is viable for Qwen3 MoE. Individual GEMM microbenchmarks show speedup at lower precisions, but the concern is that surrounding operations (routing, permutation, quant/dequant kernels, allreduce) may eat into the gains. Need a fair end-to-end comparison with both performance and accuracy metrics.

### Interview Summary
**Key Discussions**:
- Model: Qwen3-30B-A3B (30B total, 3B active) in `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`
- 3 precisions: BF16 (baseline), FP8, nvFP4. Marlin INT4 is future scope.
- Target: Ampere, Hopper, Blackwell (but nvFP4 only works on Blackwell SM100/103)
- User already has GEMM microbenchmark results — this is about the full picture
- Benchmark should be realistic — same inputs, natural routing (no replay), enough iterations to average variance
- Two measurement levels: (1) e2e latency/throughput + perplexity, (2) kernel-level breakdown via Nsight Systems
- Accuracy metric: perplexity on WikiText to quantify precision degradation

### Metis Review
**Identified Gaps** (addressed):
- Hardware × precision compatibility: nvFP4 is Blackwell-only, FP8 requires SM≥89. Script must validate and skip unsupported combos.
- MoE backend can silently change with precision (CutlassFusedMoE vs TRTLLMGenFusedMoE). Must log which backend is actually used.
- POST_MOE_FUSION only triggers for nvFP4+TRTLLM backend, giving it a structural advantage. Must be controlled or documented.
- FP8 has two variants (per-tensor QDQ vs block scales). Must be explicit.
- TRT-LLM has built-in `enable_layerwise_nvtx_marker=True` — no need for manual NVTX instrumentation in model code.

---

## Work Objectives

### Core Objective
Create a standalone benchmark that fairly compares Qwen3-30B-A3B inference across BF16, FP8, and nvFP4 on both performance and accuracy axes.

### Concrete Deliverables
- `benchmarks/qwen3_moe/bench_utils.py` — shared utilities module (SM validation, model loading, CLI arg builders, JSON writer, MoE backend detection)
- `benchmarks/qwen3_moe/perf_bench.py` — performance benchmark (latency, throughput, nsys profiling mode)
- `benchmarks/qwen3_moe/ppl_bench.py` — accuracy benchmark (WikiText-2 perplexity)
- `benchmarks/qwen3_moe/analyze_nsys.py` — post-hoc script to categorize kernels from nsys sqlite export
- JSON results files per config: `perf_{precision}_{sm_version}.json`, `ppl_{precision}_{sm_version}.json`
- Nsight Systems traces: `profile_{precision}_{sm_version}.nsys-rep`

### Definition of Done
- [ ] Script runs BF16 on SM≥80, FP8 on SM≥89, nvFP4 on SM∈{100,103}
- [ ] Gracefully skips unsupported precision/hardware combos with clear message
- [ ] Produces structured JSON with latency, throughput, perplexity, and metadata
- [ ] Nsight profiling mode produces valid `.nsys-rep` files

### Must Have
- Fair comparison: same inputs, same prompt set, same batch config across all 3 precisions
- GPU-side timing via `torch.cuda.Event(enable_timing=True)` for latency measurement
- Perplexity measurement on WikiText-2 (or similar) for accuracy comparison
- Hardware validation before each precision run
- Warmup iterations (≥5) before timed runs
- Logs which MoE backend (CutlassFusedMoE/TRTLLMGenFusedMoE) is actually instantiated
- Nsight-compatible profiling mode (detect `NSYS_PROFILING_SESSION_ID`)

### Must NOT Have (Guardrails)
- Must NOT modify any files under `tensorrt_llm/_torch/models/` or `tensorrt_llm/_torch/modules/`
- Must NOT add manual NVTX markers inside TRT-LLM model code — use `enable_layerwise_nvtx_marker=True`
- Must NOT use `time.time()` for GPU kernel timing (CPU-side timing only for wall clock)
- Must NOT hardcode model paths — accept as CLI argument
- Must NOT assume all 3 precisions work on all hardware
- Must NOT include Marlin INT4 (future scope)
- Must NOT over-engineer the Nsight analysis script — just kernel categorization, not visualization

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL verification is agent-executable.

### Test Decision
- **Infrastructure exists**: NO (standalone benchmark, no test framework needed)
- **Automated tests**: NO (benchmark script, not library code)
- **Agent-Executed QA**: ALWAYS

### Agent-Executed QA Scenarios (per task, detailed below in TODOs)

The primary verification is:
1. Script runs without error for each supported precision on current hardware
2. JSON output has correct structure
3. Perplexity values are in sane ranges (BF16 < 20, quantized < 2x BF16)
4. Nsight mode produces non-empty `.nsys-rep` file

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Shared utils (bench_utils.py) + perf benchmark (qwen3_moe_perf_bench.py) — includes nsys mode
└── Task 3: Nsight analysis helper script (analyze_nsys.py)

Wave 2 (After Task 1):
└── Task 2: Perplexity benchmark (qwen3_moe_ppl_bench.py) — imports from bench_utils

Wave 3 (After Wave 2):
└── Task 4: Integration testing — run both scripts on available hardware
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 4 | 3 |
| 2 | 1 | 4 | 3 |
| 3 | None | 4 | 1, 2 |
| 4 | 1, 2, 3 | None | None (final) |

---

## TODOs

- [x] 1. Create shared utilities module and performance benchmark script

  **What to do**:

  **Part A: Create `benchmarks/qwen3_moe/` package**:
  - Create `benchmarks/qwen3_moe/__init__.py` (empty, makes directory importable as package)
  - Create `benchmarks/qwen3_moe/bench_utils.py` — shared utilities for both perf and ppl scripts:
  - Hardware validation function:
    - `get_sm_version()` → tuple via `torch.cuda.get_device_capability()`
    - `validate_precision(precision, sm_version)` → bool, checks against `_QUANT_SUPPORT_TABLE` logic from `fused_moe_cutlass.py:68-114`:
      - BF16 (None): SM≥80
      - FP8 (`QuantAlgo.FP8`): SM≥89
      - nvFP4 (`QuantAlgo.NVFP4`): SM∈{100, 103}
    - `get_supported_precisions(requested, sm_version)` → filters list, logs skipped with reason
  - Model loading helper:
    - `load_model(model_path, enable_nvtx=False)` → returns `LLM` instance
    - Uses `LLM(model=model_path, backend="pytorch")` — no `QuantConfig` needed
    - Pre-quantized checkpoints from NVIDIA on HuggingFace (e.g., `nvidia/Qwen3-30B-A3B-FP8`) include `hf_quant_config.json` which TRT-LLM reads automatically
    - If `enable_nvtx=True`, sets `enable_layerwise_nvtx_marker=True` in LLM constructor
  - MoE backend detection:
    - `detect_moe_backend(llm)` → string, inspects `model.model.layers[0].mlp.experts.__class__.__name__`
  - Common CLI arg builder:
    - `add_common_args(parser)` → adds shared args to an argparse parser:
      - `--model_path`: path or HF repo ID for BF16 checkpoint (e.g., `Qwen/Qwen3-30B-A3B`)
      - `--fp8_model_path`: path or HF repo ID for FP8 checkpoint (e.g., `nvidia/Qwen3-30B-A3B-FP8`). If not set, skips FP8.
      - `--nvfp4_model_path`: path or HF repo ID for nvFP4 checkpoint. If not set, skips nvFP4.
      - `--precisions`: comma-separated list, default `bf16,fp8,nvfp4` (only runs precisions that have a model path)
      - `--output_dir`: where to write JSON results, default `./benchmark_results`
      - `--disable_fusion`: flag to set `TRTLLM_QWEN3_EAGER_FUSION_DISABLED=1` for controlled comparison
      - `--moe_backend`: `auto`, `CUTLASS`, `TRTLLM` — default `auto`
    - `get_model_path_for_precision(args, precision)` → returns correct path or None
  - JSON result writer:
    - `write_result(output_dir, prefix, precision, sm_version, data_dict)` → writes `{prefix}_{precision}_sm{major}{minor}.json`
    - Common metadata auto-populated: `precision`, `sm_version`, `gpu_name`, `moe_backend`, `fusion_disabled`
  - Model cleanup helper:
    - `cleanup_model(llm)` → `del llm; torch.cuda.empty_cache(); gc.collect()`

  **Part B: Create `benchmarks/qwen3_moe/perf_bench.py`** — performance benchmark:
  - Imports from `bench_utils`
  - CLI interface (argparse) using `add_common_args(parser)` plus perf-specific args:
    - `--batch_size`: default 1
    - `--input_len`: default 512
    - `--output_len`: default 256
    - `--num_iterations`: default 20 (timed runs)
    - `--warmup_iterations`: default 5
    - `--mode`: `perf` (default) or `nsys`
  - For each supported precision:
    - Load model via `bench_utils.load_model(path, enable_nvtx=(mode=='nsys'))`
    - Detect and log MoE backend via `bench_utils.detect_moe_backend(llm)`
    - Generate a fixed set of prompts (same across all precisions) — use a deterministic prompt list
    - Run warmup iterations (call `llm.generate()`, discard results)
    - Run timed iterations using `torch.cuda.Event(enable_timing=True)`:
      ```python
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start.record()
      outputs = llm.generate(prompts, sampling_params)
      end.record()
      torch.cuda.synchronize()
      latency_ms = start.elapsed_time(end)
      ```
    - Compute metrics: mean latency, std latency, tokens/sec (output tokens / latency), prefill tokens/sec
    - Write JSON via `bench_utils.write_result(...)`:
      ```json
      {
        "precision": "fp8",
        "quant_algo": "FP8",
        "sm_version": [9, 0],
        "gpu_name": "NVIDIA H100",
        "moe_backend": "CutlassFusedMoE",
        "fusion_disabled": false,
        "batch_size": 1,
        "input_len": 512,
        "output_len": 256,
        "num_iterations": 20,
        "warmup_iterations": 5,
        "mean_latency_ms": 123.4,
        "std_latency_ms": 2.1,
        "output_tokens_per_sec": 456.7,
        "prefill_tokens_per_sec": 1234.5
      }
      ```
    - Clean up model via `bench_utils.cleanup_model(llm)` between precision runs
  - `--mode nsys` support:
    - Detect Nsight profiling session via `NSYS_PROFILING_SESSION_ID` env var (pattern from `auto_deploy/utils/benchmark.py:98`)
    - Enable NVTX via `load_model(path, enable_nvtx=True)`
    - Run exactly 1 timed iteration within `cudaProfilerStart()`/`cudaProfilerStop()` bracket:
      ```python
      torch.cuda.cudart().cudaProfilerStart()
      outputs = llm.generate(prompts, sampling_params)
      torch.cuda.cudart().cudaProfilerStop()
      ```
    - If not running under nsys, print usage instructions:
      ```
      To capture Nsight trace, run:
        nsys profile --trace=cuda,nvtx --cuda-graph-trace=node -o profile_bf16 \
          python benchmarks/qwen3_moe/perf_bench.py --model_path <path> --precisions bf16 --mode nsys
      ```
  - Print comparison table at end (perf mode):
    ```
    Precision | Latency (ms) | Tokens/sec
    ----------|-------------|----------
    bf16      |       123.4 |      456.7
    fp8       |        89.1 |      632.4
    ```

  **Must NOT do**:
  - Do NOT modify TRT-LLM source code
  - Do NOT use `time.time()` for GPU timing
  - Do NOT hardcode model paths
  - Do NOT add NVTX markers inside `tensorrt_llm/_torch/models/` or `tensorrt_llm/_torch/modules/` — use `enable_layerwise_nvtx_marker=True`
  - Do NOT run multiple iterations in nsys mode (one iteration is sufficient for kernel analysis)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding TRT-LLM LLM API, quantization config, and CUDA profiling conventions. Creates two files with shared module design.
  - **Skills**: []
    - No special skills needed — pure Python scripting with TRT-LLM API

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 3)
  - **Blocks**: Tasks 2, 4
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `tensorrt_llm/_torch/auto_deploy/utils/benchmark.py:87-118` — Existing benchmark utility pattern: uses `torch.cuda.Event`, warmup, Nsight detection via `NSYS_PROFILING_SESSION_ID`, JSON result storage
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py:85-163` — `Qwen3MoE` class showing gate→experts→allreduce flow
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py:202-204` — `TRTLLM_QWEN3_EAGER_FUSION_DISABLED` env var for controlling fusion
  - `tensorrt_llm/_torch/auto_deploy/utils/benchmark.py:98-102` — Nsight detection and `cudaProfilerStart/Stop` pattern

  **API/Type References**:
  - `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:68-114` — `_QUANT_SUPPORT_TABLE` with exact SM constraints per precision
  - `tensorrt_llm/quantization/mode.py:52-60` — `QuantAlgo` enum values (FP8, FP8_BLOCK_SCALES, NVFP4)
  - `tensorrt_llm/llmapi/llm_args.py:2954-2957` — `enable_layerwise_nvtx_marker` field definition
  - `tensorrt_llm/_torch/pyexecutor/model_engine.py:222` — Where layerwise NVTX marker is applied

  **Documentation References**:
  - `docs/source/torch/arch_overview.md` — PyTorch backend architecture overview

  **WHY Each Reference Matters**:
  - `benchmark.py` — Follow its Event timing + nsys detection pattern verbatim; it's the codebase convention
  - `_QUANT_SUPPORT_TABLE` — Use this exact logic for hardware validation in `bench_utils.py`; don't roll your own
  - `modeling_qwen3_moe.py` — Understand what fusion env var to expose as CLI flag
  - `enable_layerwise_nvtx_marker` — This is the correct way to get per-module profiling; no custom hooks needed

  **Acceptance Criteria**:

  - [ ] `python -c "from benchmarks.qwen3_moe.bench_utils import validate_precision, load_model, write_result; print('PASS')"` → exits 0 (module importable)
  - [ ] `python benchmarks/qwen3_moe/perf_bench.py --help` exits 0 and shows all CLI args
  - [ ] `python benchmarks/qwen3_moe/perf_bench.py --model_path <path> --precisions bf16 --mode perf --num_iterations 2 --warmup_iterations 1` → exits 0, writes `benchmark_results/perf_bf16_smXX.json`
  - [ ] JSON file contains all required keys: `precision`, `sm_version`, `moe_backend`, `mean_latency_ms`, `output_tokens_per_sec`
  - [ ] Running with `--precisions nvfp4` on Hopper (SM90) prints skip message and exits cleanly (no crash)
  - [ ] Running with `--mode nsys` outside nsys prints usage instructions

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Shared utils module is importable
    Tool: Bash
    Preconditions: benchmarks/qwen3_moe/bench_utils.py exists
    Steps:
      1. python -c "from benchmarks.qwen3_moe.bench_utils import validate_precision, get_sm_version, load_model, detect_moe_backend, write_result, cleanup_model, add_common_args; print('PASS')"
      2. Assert: exit code 0
      3. Assert: stdout contains "PASS"
    Expected Result: All public functions importable
    Evidence: stdout captured

  Scenario: BF16 perf benchmark produces valid results
    Tool: Bash
    Preconditions: Qwen3-30B-A3B checkpoint available, GPU with SM≥80
    Steps:
      1. python benchmarks/qwen3_moe/perf_bench.py --model_path <path> --precisions bf16 --mode perf --num_iterations 3 --warmup_iterations 2 --output_dir /tmp/bench_test
      2. Assert: exit code 0
      3. python -c "import json; d=json.load(open('/tmp/bench_test/perf_bf16_smXX.json')); assert d['precision']=='bf16'; assert d['mean_latency_ms']>0; assert d['output_tokens_per_sec']>0; print('PASS')"
      4. Assert: stdout contains "PASS"
    Expected Result: JSON with valid performance metrics
    Evidence: /tmp/bench_test/perf_bf16_smXX.json

  Scenario: Unsupported precision gracefully skipped
    Tool: Bash
    Preconditions: Running on Ampere (SM80)
    Steps:
      1. python benchmarks/qwen3_moe/perf_bench.py --model_path <path> --precisions nvfp4 --mode perf 2>&1
      2. Assert: stdout or stderr contains "Skipping nvfp4" or "not supported on SM"
      3. Assert: exit code 0 (not a crash)
    Expected Result: Clean skip message, no traceback
    Evidence: stdout captured

  Scenario: Nsys mode prints instructions when not under nsys
    Tool: Bash
    Preconditions: Not running under nsys
    Steps:
      1. python benchmarks/qwen3_moe/perf_bench.py --model_path <path> --precisions bf16 --mode nsys 2>&1
      2. Assert: output contains "nsys profile" (usage instructions)
    Expected Result: User sees how to invoke with nsys
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add shared utils and Qwen3 MoE perf benchmark`
  - Files: `benchmarks/qwen3_moe/bench_utils.py`, `benchmarks/qwen3_moe/perf_bench.py`

---

- [ ] 2. Create perplexity benchmark script

  **What to do**:
  - Create `benchmarks/qwen3_moe/ppl_bench.py` — standalone perplexity measurement script
  - Imports from `bench_utils` for shared functionality (hardware validation, model loading, CLI args, JSON writer)
  - CLI interface (argparse) using `bench_utils.add_common_args(parser)` plus ppl-specific args:
    - `--input_len`: context window size for chunks, default 512
    - `--num_chunks`: number of WikiText chunks to evaluate, default 100
    - `--dataset`: dataset name, default `wikitext` (future: could add others)
  - Implement perplexity measurement:
    - Load WikiText-2 test set via `datasets` library (`datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")`)
    - Concatenate all text, tokenize into chunks of `input_len` tokens
    - For each chunk, call `llm.generate()` with `max_new_tokens=1` and `return_log_probs=True` (or use `SamplingParams(max_tokens=1, return_generation_logits=True, logprobs=True)`)
    - Alternatively, use TRT-LLM's existing `tensorrt_llm.tools.ppl.ppl()` function which computes: `nlls = -logits.log_softmax(dim=-1); ppls = nlls.gather(-1, output_ids); return ppls.mean().exp().item()`
    - Reference: `tensorrt_llm/tools/ppl.py:1-7` — existing PPL calculation utility
    - Compute per-token perplexity across all chunks
    - If `prompt_logprobs` is available in the PyTorch backend (it is per release notes), use that for more efficient perplexity computation — no need to generate, just compute logprobs for the input
  - Use the same WikiText chunks for all 3 precision configs (same data = fair comparison)
  - Limit to first N chunks — configurable via `--num_chunks`
  - Write JSON via `bench_utils.write_result(...)`:
    ```json
    {
      "precision": "fp8",
      "sm_version": [9, 0],
      "gpu_name": "NVIDIA H100",
      "moe_backend": "CutlassFusedMoE",
      "fusion_disabled": false,
      "dataset": "wikitext-2-raw-v1",
      "input_len": 512,
      "num_chunks": 100,
      "perplexity": 8.52,
      "per_chunk_ppls": [7.8, 9.1, ...]
    }
    ```
  - Clean up model via `bench_utils.cleanup_model(llm)` between precision runs
  - Print comparison table at end:
    ```
    Precision | Perplexity | Chunks
    ----------|------------|-------
    bf16      |       8.2  |   100
    fp8       |       8.5  |   100
    nvfp4     |       9.1  |   100
    ```

  **Must NOT do**:
  - Do NOT compute perplexity on random data — must use a real dataset
  - Do NOT use the full WikiText dataset if it takes >10 min per precision — cap chunks
  - Do NOT duplicate code that already exists in `bench_utils.py` — import it

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding of perplexity computation and TRT-LLM's logprob API
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (only task in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 4
  - **Blocked By**: Task 1 (needs `benchmarks/qwen3_moe/bench_utils.py`)

  **References**:

  **Pattern References**:
  - `benchmarks/qwen3_moe/bench_utils.py` — Shared utils created in Task 1: use `add_common_args`, `load_model`, `validate_precision`, `write_result`, `cleanup_model`
  - `tensorrt_llm/tools/ppl.py:1-7` — Existing perplexity calculation: `nlls = -logits.log_softmax(dim=-1); ppls = nlls.gather(-1, output_ids); return ppls.mean().exp()`
  - `examples/summarize.py:853` — Shows how perplexity is reported in existing benchmarks (`Per-token perplexity: {np.mean(ppls)}`)

  **API/Type References**:
  - `tensorrt_llm/sampling_params.py` — `SamplingParams` fields for logprob/logit return
  - `tensorrt_llm/evaluate/lm_eval.py:50-60` — `LmEvalWrapper` pattern for using LLM API with evaluation harness

  **WHY Each Reference Matters**:
  - `bench_utils.py` — Import all shared logic from here; do NOT reinvent model loading or CLI args
  - `ppl.py` — Use this exact formula for consistency with TRT-LLM's own accuracy reporting
  - `sampling_params.py` — Need to know which params control logprob return in PyTorch backend

  **Acceptance Criteria**:

  - [ ] `python benchmarks/qwen3_moe/ppl_bench.py --help` exits 0 and shows all CLI args
  - [ ] `python benchmarks/qwen3_moe/ppl_bench.py --model_path <path> --precisions bf16 --num_chunks 10` → exits 0
  - [ ] JSON output at `benchmark_results/ppl_bf16_smXX.json` contains `"perplexity": <float>` where value is >1.0 and <100.0

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Perplexity measurement produces sane values for BF16
    Tool: Bash
    Preconditions: Qwen3-30B-A3B BF16 checkpoint, datasets library installed
    Steps:
      1. python benchmarks/qwen3_moe/ppl_bench.py --model_path <path> --precisions bf16 --num_chunks 10 --output_dir /tmp/bench_ppl
      2. Assert: exit code 0
      3. python -c "import json; d=json.load(open('/tmp/bench_ppl/ppl_bf16_smXX.json')); ppl=d['perplexity']; assert 1.0 < ppl < 50.0, f'PPL {ppl} out of range'; print(f'PPL={ppl} PASS')"
    Expected Result: Perplexity between 1-50 for a real model on WikiText
    Evidence: JSON file with perplexity value

  Scenario: Multiple precisions compared in one run
    Tool: Bash
    Preconditions: BF16 and FP8 checkpoints available, SM≥89
    Steps:
      1. python benchmarks/qwen3_moe/ppl_bench.py --model_path <path> --fp8_model_path <fp8_path> --precisions bf16,fp8 --num_chunks 10 --output_dir /tmp/bench_ppl_multi
      2. Assert: exit code 0
      3. ls /tmp/bench_ppl_multi/ppl_*.json | wc -l
      4. Assert: count ≥ 2
      5. python -c "import json; bf16=json.load(open('/tmp/bench_ppl_multi/ppl_bf16_smXX.json')); fp8=json.load(open('/tmp/bench_ppl_multi/ppl_fp8_smXX.json')); print(f'BF16={bf16[\"perplexity\"]:.2f} FP8={fp8[\"perplexity\"]:.2f} PASS')"
    Expected Result: Both JSON files with valid perplexity values
    Evidence: /tmp/bench_ppl_multi/ppl_*.json files
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add Qwen3 MoE perplexity benchmark script`
  - Files: `benchmarks/qwen3_moe/ppl_bench.py`

---

- [x] 3. Create Nsight trace analysis helper script (unchanged from prior plan)

  **What to do**:
  - Create `benchmarks/qwen3_moe/analyze_nsys.py` that:
    - Takes a `.nsys-rep` or exported `.sqlite` file as input
    - Uses `nsys stats --report gputrace --format csv` to extract kernel trace (or parse sqlite directly)
    - Categorizes each CUDA kernel into buckets by name pattern:
      - `expert_gemm`: kernels matching `*cutlass*moe*`, `*fused_moe*`, `*nvfp4_gemm*`, `*fp8*gemm*`
      - `attention_gemm`: kernels matching `*flash*attn*`, `*fmha*`, `*gpt_attention*`
      - `gate_gemm`: kernels matching `*cublas*` within gate NVTX range
      - `routing_permute`: kernels matching `*topk*`, `*sort*`, `*scatter*`, `*gather*`, `*permute*`, `*index*`
      - `quant_dequant`: kernels matching `*quantize*`, `*dequant*`, `*cast*`, `*fp4*quant*`, `*fp8*quant*`
      - `norm`: kernels matching `*rms_norm*`, `*layer_norm*`, `*rmsnorm*`
      - `allreduce`: kernels matching `*nccl*`, `*allreduce*`, `*reduce_scatter*`
      - `other`: everything else
    - Output a summary table:
      ```
      Category        | Time (ms) | % of Total | Kernel Count
      --------------- |-----------|------------|-------------
      expert_gemm     |     45.2  |     52.3%  |          24
      attention_gemm  |     18.1  |     20.9%  |           6
      quant_dequant   |      8.3  |      9.6%  |          12
      routing_permute |      5.2  |      6.0%  |          18
      allreduce       |      4.1  |      4.7%  |           2
      norm            |      2.8  |      3.2%  |           6
      gate_gemm       |      1.5  |      1.7%  |           1
      other           |      1.3  |      1.5%  |          14
      --------------- |-----------|------------|-------------
      TOTAL           |     86.5  |    100.0%  |          83
      ```
    - Also output JSON for programmatic consumption
    - **This tells the user their Amdahl's ceiling**: if `expert_gemm` is 52% of total, max speedup from any expert precision change is 1/(1-0.52) ≈ 2.08x

  **Must NOT do**:
  - Do NOT try to parse `.nsys-rep` binary directly — use `nsys stats` CLI or sqlite export
  - Do NOT create charts/plots — just text tables and JSON

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: String matching and CSV parsing — straightforward Python scripting
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 4
  - **Blocked By**: None

  **References**:

  **External References**:
  - Nsight Systems CLI: `nsys stats --report gputrace --format csv <file.nsys-rep>` produces CSV with kernel names, durations, etc.
  - Alternative: `nsys export --type sqlite <file.nsys-rep>` → query `CUPTI_ACTIVITY_KIND_KERNEL` table

  **Acceptance Criteria**:

  - [ ] `python benchmarks/qwen3_moe/analyze_nsys.py --help` exits 0
  - [ ] Given a valid `.nsys-rep` file, produces category breakdown table to stdout
  - [ ] JSON output written to `--output` path with all categories and percentages

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Analysis script processes nsys trace
    Tool: Bash
    Preconditions: A .nsys-rep file exists from Task 3 profiling run
    Steps:
      1. nsys stats --report gputrace --format csv profile_bf16.nsys-rep > /dev/null 2>&1
      2. Assert: exit code 0 (nsys can process the file)
      3. python benchmarks/qwen3_moe/analyze_nsys.py --input profile_bf16.nsys-rep --output /tmp/kernel_breakdown.json
      4. Assert: exit code 0
      5. python -c "import json; d=json.load(open('/tmp/kernel_breakdown.json')); assert 'categories' in d; assert len(d['categories'])>0; print('PASS')"
    Expected Result: Kernel breakdown JSON with categorized timings
    Evidence: /tmp/kernel_breakdown.json
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add Nsight trace kernel categorization script`
  - Files: `benchmarks/qwen3_moe/analyze_nsys.py`

---

- [ ] 4. Integration test — full run and comparison

  **What to do**:
  - Run the performance benchmark on available hardware:
    - `python benchmarks/qwen3_moe/perf_bench.py --model_path <path> --fp8_model_path <fp8_path> --precisions bf16,fp8,nvfp4 --mode perf --num_iterations 10`
    - (nvfp4 will auto-skip on non-Blackwell)
  - Run the perplexity benchmark on available hardware:
    - `python benchmarks/qwen3_moe/ppl_bench.py --model_path <path> --fp8_model_path <fp8_path> --precisions bf16,fp8,nvfp4 --num_chunks 50`
  - Run Nsight profiling for each supported precision:
    - `nsys profile ... python benchmarks/qwen3_moe/perf_bench.py ... --mode nsys` for each precision
  - Run analysis on each trace:
    - `python benchmarks/qwen3_moe/analyze_nsys.py --input profile_{precision}.nsys-rep`
  - Verify results make sense:
    - BF16 latency > FP8 latency (FP8 should be faster)
    - FP8 latency > nvFP4 latency on Blackwell (FP4 should be faster)
    - BF16 perplexity ≤ FP8 perplexity ≤ nvFP4 perplexity (lower precision = higher PPL)
    - Expert GEMM percentage should be the dominant category in kernel breakdown
  - Print final comparison summary combining perf + ppl results

  **Must NOT do**:
  - Do NOT consider this task failed if nvFP4 is unavailable (hardware-dependent)
  - Do NOT fail if ordering assumptions are violated — just report the numbers

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Just running scripts and verifying outputs
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential, final)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2, 3

  **References**:

  All references from Tasks 1-3 apply.

  **Acceptance Criteria**:

  - [ ] Perf benchmark produces valid JSON for all supported precisions
  - [ ] PPL benchmark produces valid JSON for all supported precisions
  - [ ] Comparison data available across both scripts' outputs
  - [ ] At least one Nsight trace successfully analyzed with kernel breakdown

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Full benchmark suite runs end-to-end
    Tool: Bash
    Preconditions: Qwen3-30B-A3B checkpoint, GPU with SM≥89 (Hopper minimum for 2 precisions)
    Steps:
      1. python benchmarks/qwen3_moe/perf_bench.py --model_path <path> --fp8_model_path <fp8_path> --precisions bf16,fp8 --mode perf --num_iterations 5 --output_dir /tmp/bench_full
      2. Assert: exit code 0
      3. python benchmarks/qwen3_moe/ppl_bench.py --model_path <path> --fp8_model_path <fp8_path> --precisions bf16,fp8 --num_chunks 10 --output_dir /tmp/bench_full
      4. Assert: exit code 0
      5. ls /tmp/bench_full/perf_*.json /tmp/bench_full/ppl_*.json | wc -l
      6. Assert: count ≥ 4 (2 perf + 2 ppl)
      7. python -c "
         import json, glob
         perf_files = sorted(glob.glob('/tmp/bench_full/perf_*.json'))
         ppl_files = sorted(glob.glob('/tmp/bench_full/ppl_*.json'))
         for pf in perf_files:
             d = json.load(open(pf))
             print(f\"PERF {d['precision']:8s} | {d['mean_latency_ms']:8.1f} ms | {d['output_tokens_per_sec']:8.1f} tok/s\")
         for pf in ppl_files:
             d = json.load(open(pf))
             print(f\"PPL  {d['precision']:8s} | {d['perplexity']:.2f}\")
         print('PASS')
         "
    Expected Result: Comparison data for 2+ precisions across both scripts
    Evidence: /tmp/bench_full/perf_*.json and /tmp/bench_full/ppl_*.json files
  ```

  **Commit**: NO (no code changes — just running the benchmark)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(benchmark): add shared utils and Qwen3 MoE perf benchmark` | `benchmarks/qwen3_moe/bench_utils.py`, `benchmarks/qwen3_moe/perf_bench.py` | `python benchmarks/qwen3_moe/perf_bench.py --help` |
| 2 | `feat(benchmark): add Qwen3 MoE perplexity benchmark script` | `benchmarks/qwen3_moe/ppl_bench.py` | `python benchmarks/qwen3_moe/ppl_bench.py --help` |
| 3 | `feat(benchmark): add Nsight trace kernel categorization script` | `benchmarks/qwen3_moe/analyze_nsys.py` | `python benchmarks/qwen3_moe/analyze_nsys.py --help` |

---

## Success Criteria

### Verification Commands
```bash
# Shared utils importable
python -c "from benchmarks.qwen3_moe.bench_utils import validate_precision, load_model; print('OK')"

# Perf benchmark is runnable
python benchmarks/qwen3_moe/perf_bench.py --help

# PPL benchmark is runnable
python benchmarks/qwen3_moe/ppl_bench.py --help

# BF16 perf benchmark works
python benchmarks/qwen3_moe/perf_bench.py --model_path <path> --precisions bf16 --mode perf --num_iterations 3

# BF16 ppl benchmark works
python benchmarks/qwen3_moe/ppl_bench.py --model_path <path> --precisions bf16 --num_chunks 10

# Analysis script works
python benchmarks/qwen3_moe/analyze_nsys.py --help
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] `bench_utils.py` is importable and provides all shared functions
- [ ] Both scripts handle all 3 precisions with correct hardware validation (via shared utils)
- [ ] Perf script measures latency/throughput and supports nsys mode
- [ ] PPL script computes perplexity on WikiText-2 using `tensorrt_llm.tools.ppl.ppl()` formula
- [ ] Nsight mode uses built-in `enable_layerwise_nvtx_marker=True`
- [ ] JSON results contain all metadata fields for reproducibility
- [ ] Both scripts print comparison tables at end of multi-precision runs
- [ ] No code duplication between perf and ppl scripts — shared logic lives in `bench_utils.py`
