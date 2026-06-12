#!/usr/bin/env python3
"""Recommend an OmniVoice online serving profile for a workload JSONL."""

from __future__ import annotations

import argparse
import json

from omnivoice.serving.profiles import (
    ServingProfileRecommendation,
    recommend_serving_profile,
    recommended_runtime_config,
)
from omnivoice.utils.common import str2bool
from omnivoice.utils.data_utils import read_test_list


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a representative OmniVoice request JSONL and recommend "
            "a measured online-batching serving profile."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_list", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--frame_rate", type=int, default=25)
    parser.add_argument("--latency_sensitive", type=str2bool, default=False)
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--warmup_test_list",
        default=None,
        help=(
            "Representative JSONL to include in the emitted serve command for "
            "startup warmup. Defaults to --test_list."
        ),
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser


def build_recommended_serve_command(
    recommendation: ServingProfileRecommendation,
    *,
    model: str,
    host: str,
    port: int,
    warmup_test_list: str | None = None,
) -> list[str]:
    runtime = recommended_runtime_config()
    command = [
        "omnivoice-serve-online-batch",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--scheduler_profile",
        recommendation.profile.name,
        "--compile_llm",
        _bool_cli(runtime["compile_llm"]),
        "--compile_mode",
        str(runtime["compile_mode"]),
        "--compile_audio_heads",
        _bool_cli(runtime["compile_audio_heads"]),
        "--allow_reduce_overhead_worker",
        _bool_cli(runtime["allow_reduce_overhead_worker"]),
        "--matmul_precision",
        str(runtime["matmul_precision"]),
        "--num_step",
        str(runtime["num_step"]),
        "--batched_decode",
        _bool_cli(runtime["batched_decode"]),
        "--reuse_static_input_embeds",
        _bool_cli(runtime["reuse_static_input_embeds"]),
        "--split_guidance_forward",
        str(runtime["split_guidance_forward"]),
        "--split_guidance_min_batch_size",
        str(runtime["split_guidance_min_batch_size"]),
        "--split_guidance_min_saved_context_ratio",
        str(runtime["split_guidance_min_saved_context_ratio"]),
    ]
    if warmup_test_list is not None:
        command.extend(
            [
                "--warmup_test_list",
                warmup_test_list,
                "--warmup_batches",
                str(runtime["warmup_batches"]),
                "--warmup_fill_batch",
                _bool_cli(runtime["warmup_fill_batch"]),
            ]
        )
    return command


def _bool_cli(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    args = get_parser().parse_args()
    samples = read_test_list(args.test_list)
    recommendation = recommend_serving_profile(
        samples,
        concurrency=args.concurrency,
        frame_rate=args.frame_rate,
        latency_sensitive=args.latency_sensitive,
    )
    output = recommendation.to_dict()
    output["serve_command"] = build_recommended_serve_command(
        recommendation,
        model=args.model,
        host=args.host,
        port=args.port,
        warmup_test_list=args.warmup_test_list or args.test_list,
    )
    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=args.indent,
        )
    )


if __name__ == "__main__":
    main()
