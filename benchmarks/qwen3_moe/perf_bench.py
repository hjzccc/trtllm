#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Performance benchmark for Qwen3-30B-A3B MoE across BF16 / FP8 / nvFP4.

Measures end-to-end generation latency using ``torch.cuda.Event`` timing
(following the pattern in ``tensorrt_llm/_torch/auto_deploy/utils/benchmark.py``).

Supports two modes:
    * **perf** (default) — runs ``--num_iterations`` timed iterations after
      ``--warmup_iterations`` warm-up rounds and reports per-token latencies.
    * **nsys** — designed for use under ``nsys profile``.  Runs exactly one
      profiled iteration between ``cudaProfilerStart`` / ``cudaProfilerStop``
      and enables layerwise NVTX markers via TRT-LLM's built-in flag.

Usage examples::

    # Standard performance sweep
    python benchmarks/qwen3_moe/perf_bench.py \\
        --model_path_bf16 /models/Qwen3-30B-A3B \\
        --model_path_fp8  /models/Qwen3-30B-A3B-FP8 \\
        --batch_size 1 --input_len 128 --output_len 128

    # Nsys profiling (FP8 only)
    nsys profile -o qwen3_fp8 python benchmarks/qwen3_moe/perf_bench.py \\
        --model_path_fp8 /models/Qwen3-30B-A3B-FP8 \\
        --precisions fp8 --mode nsys
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

# Allow running as `python benchmarks/qwen3_moe/perf_bench.py` from repo root
# even though `benchmarks/` has no top-level __init__.py.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.qwen3_moe.bench_utils import (  # noqa: E402
    add_common_args,
    cleanup_model,
    detect_moe_backend,
    get_model_path_for_precision,
    get_sm_version,
    get_supported_precisions,
    load_model,
    write_result,
)

logger = logging.getLogger("qwen3_moe_bench")

# ---------------------------------------------------------------------------
# Deterministic prompt generation (fixed list for fairness across precisions)
# ---------------------------------------------------------------------------

_SEED_PROMPTS = [
    "Explain the architecture of Mixture-of-Experts models and how they differ from dense transformers.",
    "Write a Python function that computes the Fibonacci sequence using dynamic programming.",
    "Describe the key innovations in the NVIDIA Blackwell GPU architecture.",
    "Summarize the plot of Shakespeare's Hamlet in three paragraphs.",
    "What are the main challenges in deploying large language models at scale?",
    "Explain how quantization reduces memory usage without significant accuracy loss.",
    "Write a short essay on the history of artificial intelligence research.",
    "Describe the difference between FP8 and FP4 number formats in neural networks.",
    "What is the role of the router in a Mixture-of-Experts transformer layer?",
    "Explain the concept of KV-cache and why it matters for autoregressive generation.",
    "Write a recursive algorithm for binary search in C++.",
    "What are the trade-offs between model parallelism and data parallelism?",
    "Describe how NVIDIA TensorRT-LLM optimizes inference for large language models.",
    "Explain the mathematical foundation of attention mechanisms in transformers.",
    "What is speculative decoding and how does it improve inference throughput?",
    "Write a SQL query that finds the top 10 customers by total order value.",
]


def _build_prompts(batch_size: int, input_len: int) -> list[str]:
    """Build a deterministic list of *batch_size* prompts.

    Each prompt is truncated or repeated to approximate *input_len* tokens
    (rough heuristic: 1 token ≈ 4 characters).

    Args:
        batch_size: Number of prompts.
        input_len: Target input length in tokens.

    Returns:
        List of prompt strings.
    """
    target_chars = input_len * 4  # rough token-to-char ratio
    prompts: list[str] = []
    for i in range(batch_size):
        base = _SEED_PROMPTS[i % len(_SEED_PROMPTS)]
        # Repeat until we reach the target character length
        repeated = base
        while len(repeated) < target_chars:
            repeated += " " + base
        prompts.append(repeated[:target_chars])
    return prompts


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def _run_perf_benchmark(
    llm,
    prompts: list[str],
    output_len: int,
    num_iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    """Run timed iterations and return latency metrics.

    Follows the ``torch.cuda.Event`` timing pattern from
    ``tensorrt_llm/_torch/auto_deploy/utils/benchmark.py:87-118``.

    Args:
        llm: Loaded ``LLM`` instance.
        prompts: Input prompt list.
        output_len: Max new tokens per request.
        num_iterations: Number of timed iterations.
        warmup_iterations: Number of untimed warm-up iterations.

    Returns:
        Dict with latency statistics.
    """
    from tensorrt_llm import SamplingParams

    sampling_params = SamplingParams(max_new_tokens=output_len)

    # --- Warm-up (untimed) ---
    for _ in range(warmup_iterations):
        llm.generate(prompts, sampling_params=sampling_params)

    # --- Timed iterations ---
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies_ms: list[float] = []

    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start.record()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        end.record()
        torch.cuda.synchronize()
        latencies_ms.append(start.elapsed_time(end))

    # Compute total generated tokens from last iteration
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    batch_size = len(prompts)

    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    min_latency_ms = min(latencies_ms)
    max_latency_ms = max(latencies_ms)
    tokens_per_sec = (total_output_tokens / avg_latency_ms) * 1000.0
    time_per_token_ms = avg_latency_ms / max(total_output_tokens, 1)

    return {
        "mode": "perf",
        "batch_size": batch_size,
        "input_len": len(prompts[0]) // 4,  # approximate
        "output_len": output_len,
        "num_iterations": num_iterations,
        "warmup_iterations": warmup_iterations,
        "total_output_tokens_last_iter": total_output_tokens,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "min_latency_ms": round(min_latency_ms, 3),
        "max_latency_ms": round(max_latency_ms, 3),
        "tokens_per_second": round(tokens_per_sec, 2),
        "time_per_token_ms": round(time_per_token_ms, 3),
        "all_latencies_ms": [round(l, 3) for l in latencies_ms],
    }


def _run_nsys_benchmark(
    llm,
    prompts: list[str],
    output_len: int,
) -> dict[str, Any]:
    """Run a single iteration under nsys profiling.

    Follows the nsys detection pattern from
    ``tensorrt_llm/_torch/auto_deploy/utils/benchmark.py:98-102``.

    If not running under ``nsys``, prints usage instructions and exits.

    Args:
        llm: Loaded ``LLM`` instance (with NVTX markers enabled).
        prompts: Input prompt list.
        output_len: Max new tokens.

    Returns:
        Dict with nsys-mode metadata.
    """
    from tensorrt_llm import SamplingParams

    sampling_params = SamplingParams(max_new_tokens=output_len)

    use_nsys = bool(os.environ.get("NSYS_PROFILING_SESSION_ID", None))

    if not use_nsys:
        logger.warning(
            "nsys mode requested but NSYS_PROFILING_SESSION_ID not set.\n"
            "Run with:\n"
            "  nsys profile -o <output> python benchmarks/qwen3_moe/perf_bench.py "
            "--mode nsys ...\n"
            "Proceeding with a single untimed iteration for validation."
        )

    # Warm-up: 1 iteration
    llm.generate(prompts, sampling_params=sampling_params)

    if use_nsys:
        torch.cuda.cudart().cudaProfilerStart()

    # Single profiled iteration
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    if use_nsys:
        torch.cuda.cudart().cudaProfilerStop()

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    return {
        "mode": "nsys",
        "batch_size": len(prompts),
        "output_len": output_len,
        "total_output_tokens": total_output_tokens,
        "nsys_session_detected": use_nsys,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    """Print a formatted comparison table of perf results across precisions.

    Args:
        results: ``{precision: result_dict}`` mapping.
    """
    perf_results = {k: v for k, v in results.items() if v.get("mode") == "perf"}
    if not perf_results:
        return

    header = (
        f"{'Precision':<10} {'Backend':<25} {'Avg(ms)':<12} "
        f"{'Min(ms)':<12} {'Max(ms)':<12} {'Tok/s':<12} {'ms/tok':<10}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("Qwen3-30B-A3B MoE Performance Comparison")
    print(sep)
    print(header)
    print(sep)

    for prec, data in perf_results.items():
        backend = data.get("moe_backend", "N/A")
        print(
            f"{prec:<10} {backend:<25} "
            f"{data['avg_latency_ms']:<12.3f} "
            f"{data['min_latency_ms']:<12.3f} "
            f"{data['max_latency_ms']:<12.3f} "
            f"{data['tokens_per_second']:<12.2f} "
            f"{data['time_per_token_ms']:<10.3f}"
        )

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Performance benchmark for Qwen3-30B-A3B MoE across precisions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Standard run:\n"
            "  python benchmarks/qwen3_moe/perf_bench.py \\\n"
            "      --model_path_bf16 /models/Qwen3-30B-A3B \\\n"
            "      --model_path_fp8  /models/Qwen3-30B-A3B-FP8\n\n"
            "  # Nsys profiling:\n"
            "  nsys profile -o qwen3 python benchmarks/qwen3_moe/perf_bench.py \\\n"
            "      --model_path_fp8 /models/Qwen3-30B-A3B-FP8 \\\n"
            "      --precisions fp8 --mode nsys\n"
        ),
    )

    # Perf-specific args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of prompts per batch (default: 1).",
    )
    parser.add_argument(
        "--input_len",
        type=int,
        default=128,
        help="Approximate input length in tokens (default: 128).",
    )
    parser.add_argument(
        "--output_len",
        type=int,
        default=128,
        help="Max new tokens to generate (default: 128).",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Number of timed iterations in perf mode (default: 5).",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=2,
        help="Number of warm-up iterations in perf mode (default: 2).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["perf", "nsys"],
        default="perf",
        help="Benchmark mode: 'perf' for latency measurement, 'nsys' for profiling (default: perf).",
    )

    # Shared args from bench_utils
    add_common_args(parser)

    return parser.parse_args()


def main() -> None:
    """Entry point for the performance benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = _parse_args()

    # Validate GPU availability
    if not torch.cuda.is_available():
        logger.error("No CUDA device found. Exiting.")
        sys.exit(1)

    sm_version = get_sm_version()
    sm_num = sm_version[0] * 10 + sm_version[1]
    gpu_name = torch.cuda.get_device_name()
    logger.info("GPU: %s (SM %d)", gpu_name, sm_num)

    # Filter precisions to supported ones
    supported = get_supported_precisions(args.precisions, sm_version)
    if not supported:
        logger.error("No supported precisions for SM %d. Exiting.", sm_num)
        sys.exit(1)

    # Build deterministic prompts
    prompts = _build_prompts(args.batch_size, args.input_len)
    logger.info(
        "Benchmark config: batch_size=%d, input_len≈%d, output_len=%d, mode=%s",
        args.batch_size,
        args.input_len,
        args.output_len,
        args.mode,
    )

    all_results: dict[str, dict[str, Any]] = {}

    for precision in supported:
        model_path = get_model_path_for_precision(args, precision)
        if model_path is None:
            logger.warning(
                "No model path for precision '%s' — skipping. "
                "Use --model_path_%s to provide one.",
                precision,
                precision,
            )
            continue

        logger.info("=" * 60)
        logger.info("Benchmarking precision: %s", precision.upper())
        logger.info("Model path: %s", model_path)
        logger.info("=" * 60)

        enable_nvtx = args.mode == "nsys"
        llm = load_model(model_path, enable_nvtx=enable_nvtx)

        moe_backend = detect_moe_backend(llm)
        logger.info("MoE backend: %s", moe_backend)

        if args.mode == "perf":
            result = _run_perf_benchmark(
                llm,
                prompts,
                args.output_len,
                args.num_iterations,
                args.warmup_iterations,
            )
        else:
            result = _run_nsys_benchmark(llm, prompts, args.output_len)

        result["moe_backend"] = moe_backend
        all_results[precision] = result

        # Write result to JSON
        write_result(
            args.output_dir,
            prefix="perf",
            precision=precision,
            sm_version=sm_version,
            data_dict=result,
        )

        cleanup_model(llm)
        logger.info("Finished precision: %s\n", precision.upper())

    # Print comparison table (perf mode only)
    if args.mode == "perf":
        _print_comparison_table(all_results)

    logger.info("All benchmarks complete. Results in: %s/", args.output_dir)


if __name__ == "__main__":
    main()
