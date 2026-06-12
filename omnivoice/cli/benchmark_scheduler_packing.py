#!/usr/bin/env python3
"""Offline benchmark for OmniVoice scheduler packing policies."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from omnivoice.serving import (
    BatchSchedulerConfig,
    OmniVoiceBatchRequest,
    OmniVoiceBatchScheduler,
)
from omnivoice.serving.batcher import _QueuedRequest
from omnivoice.utils.common import str2bool
from omnivoice.utils.data_utils import read_test_list


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OmniVoice online scheduler packing policies without "
            "loading the model or using GPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_list", required=True)
    parser.add_argument("--res_file", default=None)
    parser.add_argument("--policies", default="target,target_context")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--request_repeats", type=int, default=1)
    parser.add_argument("--frame_rate", type=int, default=25)
    parser.add_argument("--max_total_target_tokens", type=int, default=6144)
    parser.add_argument("--max_total_context_tokens", type=int, default=12288)
    parser.add_argument("--max_cost_ratio", type=float, default=2.0)
    parser.add_argument("--max_context_ratio", type=float, default=2.0)
    parser.add_argument("--max_context_padding_ratio", type=float, default=2.0)
    parser.add_argument("--max_seed_lookahead", type=int, default=128)
    parser.add_argument("--include_batches", type=str2bool, default=False)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def _policies(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("policies must contain at least one value")
    valid = {"target", "context", "target_context"}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"unknown candidate pack policies: {', '.join(unknown)}")
    return values


def _expand_samples(
    samples: list[dict[str, Any]],
    *,
    request_repeats: int,
) -> list[dict[str, Any]]:
    if request_repeats <= 0:
        raise ValueError("request_repeats must be positive")
    if request_repeats == 1:
        return samples

    expanded: list[dict[str, Any]] = []
    for repeat_index in range(request_repeats):
        for sample_index, sample in enumerate(samples):
            repeated = dict(sample)
            base_id = (
                sample.get("id")
                or sample.get("save_name")
                or f"sample_{sample_index + 1:04d}"
            )
            repeated["id"] = f"{base_id}_rep{repeat_index + 1}"
            expanded.append(repeated)
    return expanded


def _request_from_sample(sample: dict[str, Any], index: int) -> OmniVoiceBatchRequest:
    return OmniVoiceBatchRequest(
        request_id=str(sample.get("id") or sample.get("save_name") or index),
        text=str(sample.get("text") or ""),
        language=sample.get("language_id") or sample.get("language"),
        ref_audio=sample.get("ref_audio"),
        ref_text=sample.get("ref_text"),
        instruct=sample.get("instruct"),
        duration=sample.get("duration"),
        speed=sample.get("speed"),
        cost_tokens_hint=sample.get("cost_tokens_hint"),
        priority=sample.get("priority") or "normal",
    )


def _queued_from_sample(
    *,
    scheduler: OmniVoiceBatchScheduler,
    loop: asyncio.AbstractEventLoop,
    sample: dict[str, Any],
    index: int,
    enqueued_at: float,
) -> _QueuedRequest:
    request = _request_from_sample(sample, index)
    cost_tokens = scheduler._estimate_cost_tokens(request)
    context_tokens_hint = sample.get("context_tokens_hint")
    if context_tokens_hint is None:
        context_tokens = scheduler._estimate_context_tokens(
            request,
            cost_tokens=cost_tokens,
        )
    else:
        context_tokens = max(1, int(context_tokens_hint))
    return _QueuedRequest(
        request=request,
        future=loop.create_future(),
        loop=loop,
        enqueued_at=enqueued_at,
        cost_tokens=cost_tokens,
        context_tokens=context_tokens,
        mode=scheduler._mode_key(request),
    )


def simulate_packing(
    samples: list[dict[str, Any]],
    *,
    config: BatchSchedulerConfig,
    include_batches: bool = False,
) -> dict[str, Any]:
    loop = asyncio.new_event_loop()
    scheduler = OmniVoiceBatchScheduler(model=object(), config=config)
    now = time.monotonic()
    try:
        for index, sample in enumerate(samples):
            queued = _queued_from_sample(
                scheduler=scheduler,
                loop=loop,
                sample=sample,
                index=index,
                enqueued_at=now + index * 0.000001,
            )
            if queued.request.priority == "high":
                scheduler._ready_high.append(queued)
            else:
                scheduler._ready_normal.append(queued)

        started = time.perf_counter()
        batches: list[dict[str, Any]] = []
        while scheduler._pending_total_locked() > 0:
            seed = scheduler._peek_seed_locked()
            if seed is None:
                break
            batch = scheduler._select_batch_locked(seed)
            if not batch:
                batch = [seed]
            scheduler._pop_selected_locked(batch)
            batches.append(_summarize_batch(batch))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        loop.close()

    return _summarize_packing_result(
        policy=config.candidate_pack_policy,
        batches=batches,
        elapsed_ms=elapsed_ms,
        max_batch_size=config.max_batch_size,
        include_batches=include_batches,
    )


def _summarize_batch(batch: list[_QueuedRequest]) -> dict[str, Any]:
    cost_tokens = [item.cost_tokens for item in batch]
    context_tokens = [item.context_tokens for item in batch]
    max_cost_tokens = max(cost_tokens) if cost_tokens else 0
    max_context_tokens = max(context_tokens) if context_tokens else 0
    total_cost_tokens = sum(cost_tokens)
    total_context_tokens = sum(context_tokens)
    return {
        "request_ids": [item.request.request_id for item in batch],
        "batch_size": len(batch),
        "mode": batch[0].mode if batch else None,
        "cost_tokens": total_cost_tokens,
        "max_cost_tokens": max_cost_tokens,
        "target_padding_work": len(batch) * max_cost_tokens,
        "target_padding_ratio": (
            len(batch) * max_cost_tokens / total_cost_tokens
            if total_cost_tokens > 0
            else 0.0
        ),
        "context_tokens": total_context_tokens,
        "max_context_tokens": max_context_tokens,
        "context_padding_work": len(batch) * max_context_tokens,
        "context_padding_ratio": (
            len(batch) * max_context_tokens / total_context_tokens
            if total_context_tokens > 0
            else 0.0
        ),
    }


def _summarize_packing_result(
    *,
    policy: str,
    batches: list[dict[str, Any]],
    elapsed_ms: float,
    max_batch_size: int,
    include_batches: bool,
) -> dict[str, Any]:
    num_requests = sum(int(batch["batch_size"]) for batch in batches)
    num_batches = len(batches)
    total_cost = sum(int(batch["cost_tokens"]) for batch in batches)
    total_target_work = sum(int(batch["target_padding_work"]) for batch in batches)
    total_context = sum(int(batch["context_tokens"]) for batch in batches)
    total_context_work = sum(int(batch["context_padding_work"]) for batch in batches)
    batch_sizes = [int(batch["batch_size"]) for batch in batches]
    context_ratios = [float(batch["context_padding_ratio"]) for batch in batches]
    target_ratios = [float(batch["target_padding_ratio"]) for batch in batches]
    result = {
        "policy": policy,
        "num_requests": num_requests,
        "num_batches": num_batches,
        "avg_batch_size": num_requests / num_batches if num_batches else 0.0,
        "batch_size_histogram": _histogram(batch_sizes),
        "full_batch_count": sum(1 for size in batch_sizes if size == max_batch_size),
        "select_cpu_ms": elapsed_ms,
        "target_tokens": total_cost,
        "target_padding_work": total_target_work,
        "target_padding_work_ratio": (
            total_target_work / total_cost if total_cost > 0 else 0.0
        ),
        "target_padding_ratio": _distribution(target_ratios),
        "context_tokens": total_context,
        "context_padding_work": total_context_work,
        "context_padding_work_ratio": (
            total_context_work / total_context if total_context > 0 else 0.0
        ),
        "context_padding_ratio": _distribution(context_ratios),
    }
    if include_batches:
        result["batches"] = batches
    return result


def _compare_to_baseline(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    baseline = results[0]
    comparisons = []
    for row in results:
        comparisons.append(
            {
                "policy": row["policy"],
                "baseline_policy": baseline["policy"],
                "num_batches_delta": row["num_batches"] - baseline["num_batches"],
                "num_batches_pct_delta": _pct_delta(
                    row["num_batches"],
                    baseline["num_batches"],
                ),
                "avg_batch_size_pct_delta": _pct_delta(
                    row["avg_batch_size"],
                    baseline["avg_batch_size"],
                ),
                "target_padding_work_pct_delta": _pct_delta(
                    row["target_padding_work"],
                    baseline["target_padding_work"],
                ),
                "context_padding_work_pct_delta": _pct_delta(
                    row["context_padding_work"],
                    baseline["context_padding_work"],
                ),
                "select_cpu_ms_delta": row["select_cpu_ms"]
                - baseline["select_cpu_ms"],
            }
        )
    return comparisons


def _pct_delta(value: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (value - baseline) / baseline


def _histogram(values: list[Any]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts, key=str)}


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": max(ordered),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * quantile
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _run(args: argparse.Namespace) -> dict[str, Any]:
    samples = _expand_samples(
        read_test_list(args.test_list),
        request_repeats=args.request_repeats,
    )
    results = []
    for policy in _policies(args.policies):
        config = BatchSchedulerConfig(
            max_batch_size=args.batch_size,
            max_wait_ms=10_000.0,
            partial_batch_floor=1,
            max_total_target_tokens=args.max_total_target_tokens,
            max_total_context_tokens=args.max_total_context_tokens,
            max_cost_ratio=args.max_cost_ratio,
            max_context_ratio=args.max_context_ratio,
            max_context_padding_ratio=args.max_context_padding_ratio,
            ready_queue_capacity=max(len(samples), args.batch_size),
            frame_rate=args.frame_rate,
            use_model_duration_estimator=False,
            lookahead_for_full_batch=True,
            lookahead_for_partial_batch=True,
            max_seed_lookahead=args.max_seed_lookahead,
            candidate_pack_policy=policy,
            adaptive_memory_batch_cap=False,
        )
        results.append(
            simulate_packing(
                samples,
                config=config,
                include_batches=args.include_batches,
            )
        )

    summary = {
        "config": {
            "test_list": args.test_list,
            "num_samples": len(samples),
            "policies": _policies(args.policies),
            "batch_size": args.batch_size,
            "max_total_target_tokens": args.max_total_target_tokens,
            "max_total_context_tokens": args.max_total_context_tokens,
            "max_cost_ratio": args.max_cost_ratio,
            "max_context_ratio": args.max_context_ratio,
            "max_context_padding_ratio": args.max_context_padding_ratio,
            "max_seed_lookahead": args.max_seed_lookahead,
        },
        "results": results,
        "comparisons": _compare_to_baseline(results),
    }
    if args.res_file:
        path = Path(args.res_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=args.indent),
            encoding="utf-8",
        )
    return summary


def main() -> None:
    args = get_parser().parse_args()
    print(json.dumps(_run(args), ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
