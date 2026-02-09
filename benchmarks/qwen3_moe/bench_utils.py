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
"""Shared utilities for Qwen3-30B-A3B MoE precision benchmarks.

Provides SM validation, model loading/cleanup, MoE backend detection,
CLI argument helpers, and result serialization.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pathlib
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# When running from the TensorRT-LLM source tree the local ``tensorrt_llm/``
# directory shadows the *installed* package (which contains the compiled C++
# ``bindings`` extension).  We detect this situation and temporarily patch
# ``sys.path`` so that ``import tensorrt_llm`` resolves to the installed wheel.
# ---------------------------------------------------------------------------
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
    # Also remove '' or '.' which Python adds for CWD — it has the same effect
    # when CWD == repo root.
    for _cwd_marker in ("", "."):
        if _cwd_marker in sys.path and os.path.abspath(
                _cwd_marker) == _REPO_ROOT:
            sys.path.remove(_cwd_marker)
            break
from typing import Any

import torch

logger = logging.getLogger("qwen3_moe_bench")

# ---------------------------------------------------------------------------
# SM / precision validation
# ---------------------------------------------------------------------------
# Mirrors _QUANT_SUPPORT_TABLE from
# tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:68-114
_PRECISION_SM_CONSTRAINTS: dict[str, tuple[str, Any]] = {
    "bf16": ("min", 80),
    "fp8": ("min", 89),
    "nvfp4": ("min",100),
}


def get_sm_version() -> tuple[int, int]:
    """Return the (major, minor) SM version of the current CUDA device."""
    return torch.cuda.get_device_capability()


def validate_precision(precision: str, sm_version: tuple[int, int]) -> bool:
    """Check whether *precision* is supported on the given SM version.

    Uses the same constraint logic as ``_QUANT_SUPPORT_TABLE`` in
    ``fused_moe_cutlass.py``.

    Args:
        precision: One of ``"bf16"``, ``"fp8"``, ``"nvfp4"``.
        sm_version: ``(major, minor)`` tuple from
            :func:`torch.cuda.get_device_capability`.

    Returns:
        ``True`` if the precision is supported, ``False`` otherwise.
    """
    precision = precision.lower()
    if precision not in _PRECISION_SM_CONSTRAINTS:
        logger.warning("Unknown precision '%s' — cannot validate.", precision)
        return False

    constraint_type, constraint_val = _PRECISION_SM_CONSTRAINTS[precision]
    sm_major = sm_version[0] * 10 + sm_version[1]  # e.g. (9, 0) → 90

    if constraint_type == "min":
        return sm_major >= constraint_val
    if constraint_type == "exact":
        return sm_major == constraint_val
    if constraint_type == "in":
        return sm_major in constraint_val

    return False


def get_supported_precisions(
    requested: list[str],
    sm_version: tuple[int, int],
) -> list[str]:
    """Filter *requested* precisions to those supported on this GPU.

    Logs a skip message for each unsupported precision.

    Args:
        requested: Ordered list of precision strings.
        sm_version: ``(major, minor)`` tuple.

    Returns:
        Filtered list retaining original order.
    """
    supported: list[str] = []
    for prec in requested:
        if validate_precision(prec, sm_version):
            supported.append(prec)
        else:
            sm_num = sm_version[0] * 10 + sm_version[1]
            logger.warning(
                "Skipping precision '%s' — not supported on SM %d.",
                prec,
                sm_num,
            )
    return supported


# ---------------------------------------------------------------------------
# Model loading / cleanup
# ---------------------------------------------------------------------------

def load_model(model_path: str, enable_nvtx: bool = False):
    """Load a Qwen3-30B-A3B model via the TRT-LLM high-level ``LLM`` API.

    Pre-quantized FP8/nvFP4 checkpoints include ``hf_quant_config.json``
    which TRT-LLM reads automatically — no ``QuantConfig`` needed.

    Args:
        model_path: Path to the HuggingFace-format checkpoint.
        enable_nvtx: If ``True``, passes
            ``enable_layerwise_nvtx_marker=True`` for nsys profiling.

    Returns:
        A ``tensorrt_llm.LLM`` instance.
    """
    from tensorrt_llm import LLM

    kwargs: dict[str, Any] = {
        "model": model_path,
        "backend": "pytorch",
    }
    if enable_nvtx:
        kwargs["enable_layerwise_nvtx_marker"] = True

    llm = LLM(**kwargs)
    return llm


def detect_moe_backend(llm) -> str:
    """Inspect the loaded model to determine which MoE kernel is active.

    The MoE backend can silently switch between ``CutlassFusedMoE`` and
    ``TRTLLMGenFusedMoE`` depending on precision and SM version.

    Args:
        llm: A loaded ``LLM`` instance.

    Returns:
        Class name of the experts module (e.g. ``"CutlassFusedMoE"``) or
        ``"unknown"`` if introspection fails.
    """
    try:
        layer0 = llm.model.model.layers[0]
        experts = layer0.mlp.experts
        return experts.__class__.__name__
    except (AttributeError, IndexError):
        logger.warning("Could not introspect MoE backend — returning 'unknown'.")
        return "unknown"


def cleanup_model(llm) -> None:
    """Explicitly delete an LLM instance and reclaim GPU memory.

    Args:
        llm: The ``LLM`` object to destroy.
    """
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register shared CLI arguments for all benchmark scripts.

    Adds:
        ``--model_path_bf16``, ``--model_path_fp8``, ``--model_path_nvfp4``,
        ``--precisions``, ``--output_dir``.
    """
    parser.add_argument(
        "--model_path_bf16",
        type=str,
        default=None,
        help="Path to BF16 Qwen3-30B-A3B checkpoint.",
    )
    parser.add_argument(
        "--model_path_fp8",
        type=str,
        default=None,
        help="Path to FP8 Qwen3-30B-A3B checkpoint (e.g. nvidia/Qwen3-30B-A3B-FP8).",
    )
    parser.add_argument(
        "--model_path_nvfp4",
        type=str,
        default=None,
        help="Path to nvFP4 Qwen3-30B-A3B checkpoint.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["bf16", "fp8", "nvfp4"],
        choices=["bf16", "fp8", "nvfp4"],
        help="Precisions to benchmark (default: all three).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to write JSON result files.",
    )


def get_model_path_for_precision(
    args: argparse.Namespace,
    precision: str,
) -> str | None:
    """Look up the model-path argument corresponding to *precision*.

    Args:
        args: Parsed CLI namespace.
        precision: ``"bf16"``, ``"fp8"``, or ``"nvfp4"``.

    Returns:
        The path string or ``None`` if not provided.
    """
    mapping = {
        "bf16": "model_path_bf16",
        "fp8": "model_path_fp8",
        "nvfp4": "model_path_nvfp4",
    }
    attr = mapping.get(precision.lower())
    if attr is None:
        return None
    return getattr(args, attr, None)


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def write_result(
    output_dir: str,
    prefix: str,
    precision: str,
    sm_version: tuple[int, int],
    data_dict: dict[str, Any],
) -> pathlib.Path:
    """Write a benchmark result to a JSON file.

    The file is written to
    ``<output_dir>/<prefix>_<precision>_sm<SM>.json``.

    Args:
        output_dir: Target directory (created if needed).
        prefix: Filename prefix (e.g. ``"perf"``).
        precision: Precision label.
        sm_version: ``(major, minor)`` tuple.
        data_dict: Arbitrary dict of results to serialise.

    Returns:
        :class:`pathlib.Path` of the written file.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sm_str = f"sm{sm_version[0] * 10 + sm_version[1]}"
    filename = f"{prefix}_{precision}_{sm_str}.json"
    filepath = out / filename

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "precision": precision,
        "sm_version": list(sm_version),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        **data_dict,
    }

    with filepath.open("w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Result written to %s", filepath)
    return filepath
