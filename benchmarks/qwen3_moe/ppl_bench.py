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
"""Perplexity benchmark for Qwen3-30B-A3B MoE across BF16 / FP8 / nvFP4.

Evaluates perplexity on the WikiText-2 test set using the same tokenized
chunks for every precision, ensuring a fair comparison.

The perplexity formula matches ``tensorrt_llm/tools/ppl.py``::

    nlls = -logits.log_softmax(dim=-1)
    ppls = nlls.gather(-1, output_ids.long().unsqueeze(-1))
    ppl  = ppls.mean().exp().item()

Usage examples::

    # Evaluate all precisions
    python benchmarks/qwen3_moe/ppl_bench.py \\
        --model_path_bf16 /models/Qwen3-30B-A3B \\
        --model_path_fp8  /models/Qwen3-30B-A3B-FP8 \\
        --input_len 512 --num_chunks 100

    # Single precision, shorter run
    python benchmarks/qwen3_moe/ppl_bench.py \\
        --model_path_fp8 /models/Qwen3-30B-A3B-FP8 \\
        --precisions fp8 --num_chunks 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

# Allow running as `python benchmarks/qwen3_moe/ppl_bench.py` from repo root
# even though `benchmarks/` has no top-level __init__.py.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.qwen3_moe.bench_utils import (  # noqa: E402
    add_common_args,
    cleanup_model,
    detect_moe_backend,
    get_moe_backend_for_precision,
    get_model_path_for_precision,
    get_sm_version,
    get_supported_precisions,
    load_model,
    write_result,
)

logger = logging.getLogger("qwen3_moe_bench")

# ---------------------------------------------------------------------------
# WikiText-2 dataset preparation
# ---------------------------------------------------------------------------


def _load_wikitext2_tokens(tokenizer, input_len: int, num_chunks: int) -> list[list[int]]:
    """Load WikiText-2 test set, tokenize, and split into fixed-length chunks.

    The entire test split is concatenated into a single token stream (empty
    lines are filtered out).  The stream is then divided into non-overlapping
    chunks of ``input_len`` tokens.  Only the first ``num_chunks`` chunks are
    returned, keeping runtime bounded.

    Args:
        tokenizer: A HuggingFace-compatible tokenizer.
        input_len: Number of tokens per chunk.
        num_chunks: Maximum number of chunks to return.

    Returns:
        List of token-id lists, each of length ``input_len``.
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # Concatenate all non-empty lines into one string
    text = "\n".join(line for line in dataset["text"] if line.strip())

    # Tokenize the entire corpus at once
    all_token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)

    # Chunk into non-overlapping windows of `input_len` tokens
    total_available = len(all_token_ids) // input_len
    n_chunks = min(num_chunks, total_available)
    if n_chunks < num_chunks:
        logger.warning(
            "WikiText-2 only provides %d full chunks of %d tokens (requested %d). Using %d chunks.",
            total_available,
            input_len,
            num_chunks,
            n_chunks,
        )

    chunks: list[list[int]] = []
    for i in range(n_chunks):
        start = i * input_len
        end = start + input_len
        chunks.append(all_token_ids[start:end])

    logger.info(
        "Prepared %d chunks of %d tokens from WikiText-2 (%d total tokens).",
        len(chunks),
        input_len,
        len(all_token_ids),
    )
    return chunks


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------


def _ppl(logits: torch.Tensor, output_ids: torch.Tensor) -> float:
    """Compute per-token perplexity — mirrors ``tensorrt_llm/tools/ppl.py``.

    Args:
        logits: Tensor of shape ``(seq_len, vocab_size)``.
        output_ids: Tensor of shape ``(seq_len,)`` — the target token ids.

    Returns:
        Scalar perplexity value.
    """
    nlls = -logits.log_softmax(dim=-1)
    ppls = nlls.gather(-1, output_ids.long().unsqueeze(-1))
    return ppls.mean().exp().item()


def _evaluate_perplexity(
    llm,
    chunks: list[list[int]],
) -> dict[str, Any]:
    """Run perplexity evaluation over pre-tokenized chunks.

    For each chunk the model is called with ``return_context_logits=True``
    and ``max_tokens=1`` (we only need the context-phase logits, not
    generation).  Perplexity is computed as the cross-entropy loss between
    the logits at position *i* and the ground-truth token at position *i+1*.

    Args:
        llm: A loaded ``LLM`` instance.
        chunks: Pre-tokenized chunks (lists of token ids).

    Returns:
        Dict with ``perplexity`` (geometric mean across chunks) and
        ``per_chunk_ppls`` list.
    """
    from tensorrt_llm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=1,
        return_context_logits=True,
    )

    per_chunk_ppls: list[float] = []

    for idx, token_ids in enumerate(chunks):
        # Build prompt from token ids using the tokenizer embedded in LLM
        # The LLM.generate() can accept token ids directly as a list of ints
        # wrapped in a list (batch of 1).
        outputs = llm.generate(
            [token_ids],
            sampling_params=sampling_params,
        )

        output = outputs[0]

        # context_logits shape: (input_len, vocab_size)
        context_logits = output.context_logits
        if context_logits is None:
            raise RuntimeError(
                "context_logits is None — ensure the model supports "
                "return_context_logits=True with the PyTorch backend."
            )

        if idx == 0:
            logger.info(
                "  context_logits shape: %s (input_len=%d)",
                list(context_logits.shape),
                len(token_ids),
            )

        # context_logits[i] predicts the token at position i+1
        # (standard causal-LM convention — unshifted).
        #
        # TRT-LLM *should* return shape (input_len, V), but context
        # chunking or KV-cache pressure can cause fewer logits to be
        # returned (e.g. the first chunk's logits may be discarded).
        # We handle any returned length L <= input_len by aligning
        # from the RIGHT: use logits[:-1] to predict the last L-1
        # target tokens.
        input_ids_tensor = torch.tensor(token_ids, device=context_logits.device)
        n_logits = context_logits.shape[0]
        n_input = len(token_ids)
        if n_logits > n_input or n_logits < 2:
            raise RuntimeError(
                f"Unexpected context_logits length {n_logits} for input length {n_input}."
            )
        # logits[:-1] → positions that have a known next-token target.
        # Targets are the last (n_logits - 1) tokens of the input.
        pred_logits = context_logits[:-1]  # (n_logits-1, V)
        target_ids = input_ids_tensor[n_input - n_logits + 1 :]  # (n_logits-1,)
        chunk_ppl = _ppl(pred_logits, target_ids)
        per_chunk_ppls.append(chunk_ppl)

        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info(
                "  Chunk %d/%d — PPL: %.4f",
                idx + 1,
                len(chunks),
                chunk_ppl,
            )

    # Overall perplexity: geometric mean of per-chunk PPLs
    # = exp(mean(log(ppl_i)))
    import math

    log_ppls = [math.log(p) for p in per_chunk_ppls]
    overall_ppl = math.exp(sum(log_ppls) / len(log_ppls))

    logger.info("Overall perplexity: %.4f (%d chunks)", overall_ppl, len(per_chunk_ppls))

    return {
        "perplexity": round(overall_ppl, 4),
        "per_chunk_ppls": [round(p, 4) for p in per_chunk_ppls],
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    """Print a formatted comparison table of perplexity across precisions.

    Args:
        results: ``{precision: result_dict}`` mapping.
    """
    if not results:
        return

    header = f"{'Precision':<10} {'Perplexity':<15} {'Chunks':<10} {'MoE Backend':<25}"
    sep = "-" * len(header)

    print("\n" + sep)
    print("Qwen3-30B-A3B MoE Perplexity Comparison (WikiText-2)")
    print(sep)
    print(header)
    print(sep)

    for prec, data in results.items():
        backend = data.get("moe_backend", "N/A")
        ppl_val = data.get("perplexity", float("nan"))
        n_chunks = data.get("num_chunks", "?")
        print(f"{prec:<10} {ppl_val:<15.4f} {n_chunks:<10} {backend:<25}")

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perplexity benchmark for Qwen3-30B-A3B MoE across precisions (WikiText-2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # All precisions:\n"
            "  python benchmarks/qwen3_moe/ppl_bench.py \\\n"
            "      --model_path_bf16 /models/Qwen3-30B-A3B \\\n"
            "      --model_path_fp8  /models/Qwen3-30B-A3B-FP8\n\n"
            "  # Single precision, quick run:\n"
            "  python benchmarks/qwen3_moe/ppl_bench.py \\\n"
            "      --model_path_fp8 /models/Qwen3-30B-A3B-FP8 \\\n"
            "      --precisions fp8 --num_chunks 20\n"
        ),
    )

    # PPL-specific args
    parser.add_argument(
        "--input_len",
        type=int,
        default=512,
        help="Number of tokens per evaluation chunk (default: 512).",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=100,
        help="Maximum number of chunks to evaluate (default: 100).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext"],
        help="Dataset for perplexity evaluation (default: wikitext).",
    )

    # Shared args from bench_utils
    add_common_args(parser)

    return parser.parse_args()


def main() -> None:
    """Entry point for the perplexity benchmark."""
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

    logger.info(
        "PPL config: input_len=%d, num_chunks=%d, dataset=%s",
        args.input_len,
        args.num_chunks,
        args.dataset,
    )

    # Tokenize WikiText-2 once (shared across all precisions for fairness).
    # We need a tokenizer — load it from the first available model path.
    first_model_path = None
    for prec in supported:
        p = get_model_path_for_precision(args, prec)
        if p is not None:
            first_model_path = p
            break

    if first_model_path is None:
        logger.error(
            "No model paths provided. Use --model_path_bf16 / "
            "--model_path_fp8 / --model_path_nvfp4."
        )
        sys.exit(1)

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer from: %s", first_model_path)
    tokenizer = AutoTokenizer.from_pretrained(first_model_path, trust_remote_code=True)

    chunks = _load_wikitext2_tokens(tokenizer, args.input_len, args.num_chunks)
    if not chunks:
        logger.error("No chunks generated from WikiText-2. Exiting.")
        sys.exit(1)

    all_results: dict[str, dict[str, Any]] = {}

    for precision in supported:
        model_path = get_model_path_for_precision(args, precision)
        if model_path is None:
            logger.warning(
                "No model path for precision '%s' — skipping. Use --model_path_%s to provide one.",
                precision,
                precision,
            )
            continue

        moe_backend = get_moe_backend_for_precision(precision, sm_version)

        logger.info("=" * 60)
        logger.info("Evaluating perplexity: %s", precision.upper())
        logger.info("Model path: %s", model_path)
        logger.info("MoE backend: %s", moe_backend)
        logger.info("=" * 60)

        llm = load_model(model_path, moe_backend=moe_backend)

        moe_backend = detect_moe_backend(llm)
        logger.info("MoE backend: %s", moe_backend)

        result = _evaluate_perplexity(llm, chunks)

        result["moe_backend"] = moe_backend
        result["dataset"] = args.dataset
        result["input_len"] = args.input_len
        result["num_chunks"] = len(chunks)
        all_results[precision] = result

        # Write result to JSON
        write_result(
            args.output_dir,
            prefix="ppl",
            precision=precision,
            sm_version=sm_version,
            data_dict=result,
        )

        cleanup_model(llm)
        logger.info("Finished precision: %s\n", precision.upper())

    # Print comparison table
    _print_comparison_table(all_results)

    logger.info("All evaluations complete. Results in: %s/", args.output_dir)


if __name__ == "__main__":
    main()
