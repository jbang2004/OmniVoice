#!/usr/bin/env python3
"""Sweep client-side load profiles against an OmniVoice HTTP batch server."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from omnivoice.cli.benchmark_http_batch import _run as run_http_batch_benchmark
from omnivoice.cli.benchmark_utils import (
    build_http_sweep_profiles,
    parse_float_grid,
    parse_int_grid,
    rank_http_sweep_results,
    summarize_best_http_sweep_result,
)
from omnivoice.utils.common import str2bool


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated /v1/tts load profiles against one online batch server "
            "and rank them by success rate, wall RTF, queue latency, and batch fill."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/tts")
    parser.add_argument("--scheduler_url", default="http://127.0.0.1:8000/v1/scheduler")
    parser.add_argument("--test_list", required=True)
    parser.add_argument("--res_dir", required=True)
    parser.add_argument("--concurrency_values", default="1,2,4,8,16")
    parser.add_argument("--arrival_gap_ms_values", default="0")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--warmup_repeats",
        type=int,
        default=0,
        help=(
            "Run this many client load profiles before measured repeats for each "
            "concurrency/gap pair. Warmup results are saved but excluded from ranking."
        ),
    )
    parser.add_argument("--request_repeats", type=int, default=1)
    parser.add_argument("--timeout_s", type=float, default=120.0)
    parser.add_argument("--save_wavs", type=str2bool, default=False)
    parser.add_argument("--reset_scheduler_metrics", type=str2bool, default=True)
    parser.add_argument("--pre_register_voices", type=str2bool, default=False)
    parser.add_argument("--voice_register_url", default=None)
    return parser


def _reset_url_from_scheduler_url(scheduler_url: Optional[str]) -> Optional[str]:
    if not scheduler_url:
        return None
    suffix = "/v1/scheduler"
    if scheduler_url.endswith(suffix):
        return scheduler_url[: -len(suffix)] + "/v1/scheduler/reset_metrics"
    return scheduler_url.rstrip("/") + "/reset_metrics"


async def _reset_scheduler_metrics(
    *,
    scheduler_url: Optional[str],
    timeout_s: float,
) -> Optional[dict[str, Any]]:
    reset_url = _reset_url_from_scheduler_url(scheduler_url)
    if reset_url is None:
        return None

    import httpx

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(
            reset_url,
            json={"reset_prompt_cache_stats": False},
        )
    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "text": response.text,
        }
    return response.json()


async def _run_profile(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
) -> dict[str, Any]:
    profile_dir = Path(args.res_dir) / profile["name"]
    profile_dir.mkdir(parents=True, exist_ok=True)
    reset_result = None
    if args.reset_scheduler_metrics:
        reset_result = await _reset_scheduler_metrics(
            scheduler_url=args.scheduler_url,
            timeout_s=args.timeout_s,
        )

    batch_args = argparse.Namespace(
        url=args.url,
        scheduler_url=args.scheduler_url,
        test_list=args.test_list,
        res_dir=str(profile_dir),
        concurrency=profile["concurrency"],
        arrival_gap_ms=profile["arrival_gap_ms"],
        request_repeats=args.request_repeats,
        timeout_s=args.timeout_s,
        save_wavs=args.save_wavs,
        pre_register_voices=args.pre_register_voices,
        voice_register_url=args.voice_register_url,
    )
    summary = await run_http_batch_benchmark(batch_args)
    return {
        "profile": profile,
        "reset_result": reset_result,
        "summary": summary,
    }


def _compact_result(row: dict[str, Any]) -> dict[str, Any]:
    summary = row["summary"]
    profile = row["profile"]
    scheduler_after = summary.get("scheduler_after") or {}
    request_metrics = summary.get("request_metrics") or {}
    return {
        "profile": profile,
        "is_warmup": bool(profile.get("is_warmup")),
        "num_successful": summary.get("num_successful"),
        "num_requests": summary.get("num_requests"),
        "wall_s": summary.get("wall_s"),
        "audio_s": summary.get("audio_s"),
        "rtf_wall": summary.get("rtf_wall"),
        "avg_batch_size": scheduler_after.get("avg_batch_size"),
        "infer_rtf": scheduler_after.get("infer_rtf"),
        "split_retry_batches": scheduler_after.get("split_retry_batches"),
        "last_batch_split_retries": scheduler_after.get("last_batch_split_retries"),
        "last_batch_model_calls": scheduler_after.get("last_batch_model_calls"),
        "last_batch_max_execution_size": scheduler_after.get(
            "last_batch_max_execution_size"
        ),
        "last_batch_profile": scheduler_after.get("last_batch_profile"),
        "adaptive_batch_caps": scheduler_after.get("adaptive_batch_caps"),
        "adaptive_cap_success_streaks": scheduler_after.get(
            "adaptive_cap_success_streaks"
        ),
        "voice_registration": summary.get("voice_registration"),
        "batch_size_histogram": request_metrics.get("batch_size_histogram"),
        "queue_wait_ms": request_metrics.get("queue_wait_ms"),
        "batch_cost_tokens": request_metrics.get("batch_cost_tokens"),
        "batch_max_cost_tokens": request_metrics.get("batch_max_cost_tokens"),
        "batch_context_tokens": request_metrics.get("batch_context_tokens"),
        "batch_max_context_tokens": request_metrics.get(
            "batch_max_context_tokens"
        ),
        "batch_context_padding_ratio": request_metrics.get(
            "batch_context_padding_ratio"
        ),
        "batch_reason_histogram": request_metrics.get("batch_reason_histogram"),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    res_dir = Path(args.res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    profiles = build_http_sweep_profiles(
        concurrency_values=parse_int_grid(args.concurrency_values),
        arrival_gap_ms_values=parse_float_grid(args.arrival_gap_ms_values),
        repeats=args.repeats,
        warmup_repeats=args.warmup_repeats,
    )

    results = []
    for profile in profiles:
        results.append(await _run_profile(args=args, profile=profile))

    ranked = rank_http_sweep_results(results)
    warmup_results = [row for row in results if (row.get("profile") or {}).get("is_warmup")]
    best = summarize_best_http_sweep_result(results)
    summary = {
        "profiles": profiles,
        "best": _compact_result(best) if best else None,
        "ranked": [_compact_result(row) for row in ranked],
        "warmup_results": [_compact_result(row) for row in warmup_results],
        "results": results,
    }
    (res_dir / "http_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = get_parser().parse_args()
    summary = asyncio.run(_run(args))
    print(
        json.dumps(
            {
                "best": summary["best"],
                "ranked": summary["ranked"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
