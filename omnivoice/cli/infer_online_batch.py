#!/usr/bin/env python3
"""Online micro-batch benchmark CLI for OmniVoice.

This runner simulates concurrent single-request traffic and lets
``OmniVoiceBatchScheduler`` aggregate requests into real ``generate(list)``
calls. It is intended for throughput tuning and serving integration work; the
official offline batch runner remains ``omnivoice-infer-batch``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from omnivoice.cli.benchmark_utils import (
    request_exception_result,
    successful_request_results,
    summarize_request_results,
)
from omnivoice.models.omnivoice import OmniVoice
from omnivoice.serving import (
    BatchSchedulerConfig,
    OmniVoiceBatchRequest,
    OmniVoiceBatchScheduler,
)
from omnivoice.utils.common import get_best_device, str2bool
from omnivoice.utils.data_utils import read_test_list, require_sample_id


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online micro-batched OmniVoice inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--test_list", required=True)
    parser.add_argument("--res_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--compile_llm", type=str2bool, default=False)
    parser.add_argument("--compile_audio_heads", type=str2bool, default=False)
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument(
        "--allow_reduce_overhead_worker",
        type=str2bool,
        default=False,
        help=(
            "Allow torch.compile(mode='reduce-overhead') in the scheduler worker "
            "thread. This can be faster for one static shape, but CUDA graph "
            "capture is unsafe for heterogeneous online traffic on some PyTorch "
            "builds."
        ),
    )
    parser.add_argument("--matmul_precision", default=None)
    parser.add_argument(
        "--representative_warmup",
        type=str2bool,
        default=False,
        help="Warm up with the first real batch from test_list to precompile the steady-state shape.",
    )
    parser.add_argument(
        "--representative_warmup_batches",
        type=int,
        default=1,
        help="Number of scheduler-selected batches to warm up when representative_warmup is enabled.",
    )
    parser.add_argument(
        "--representative_warmup_fill_batch",
        type=str2bool,
        default=False,
        help=(
            "When representative warmup selects fewer samples than the "
            "scheduler max batch size, repeat compatible samples so torch.compile "
            "sees the maximum batch graph before live traffic."
        ),
    )

    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--arrival_gap_ms", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_wait_ms", type=float, default=120.0)
    parser.add_argument("--partial_batch_floor", type=int, default=2)
    parser.add_argument("--max_total_target_tokens", type=int, default=4096)
    parser.add_argument("--max_total_context_tokens", type=int, default=8192)
    parser.add_argument("--max_cost_ratio", type=float, default=1.8)
    parser.add_argument("--max_context_ratio", type=float, default=2.0)
    parser.add_argument("--max_context_padding_ratio", type=float, default=2.0)
    parser.add_argument("--ready_queue_capacity", type=int, default=64)
    parser.add_argument("--control_queue_capacity", type=int, default=16)
    parser.add_argument("--prompt_cache_entries", type=int, default=128)
    parser.add_argument("--use_model_duration_estimator", type=str2bool, default=True)
    parser.add_argument("--lookahead_for_full_batch", type=str2bool, default=True)
    parser.add_argument("--lookahead_for_partial_batch", type=str2bool, default=True)
    parser.add_argument(
        "--partial_lookahead_max_wait_multiplier",
        type=float,
        default=2.0,
    )
    parser.add_argument("--max_seed_lookahead", type=int, default=32)
    parser.add_argument(
        "--candidate_pack_policy",
        choices=["target", "context", "target_context"],
        default="target_context",
        help=(
            "Candidate ordering policy for lookahead packing. Use the same "
            "value as omnivoice-serve-online-batch when comparing benchmark "
            "and HTTP serving behavior."
        ),
    )
    parser.add_argument("--split_retry_on_memory_error", type=str2bool, default=True)
    parser.add_argument("--adaptive_memory_batch_cap", type=str2bool, default=True)
    parser.add_argument("--adaptive_memory_cap_recovery_successes", type=int, default=64)
    parser.add_argument("--max_generation_batches_before_control", type=int, default=8)

    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument(
        "--generation_mode",
        choices=["custom", "official_compatible", "optimized"],
        default="custom",
        help=(
            "High-level generation preset. custom honors the low-level flags; "
            "official_compatible pins decode/embedding behavior to the original "
            "path; optimized enables the recommended throughput settings."
        ),
    )
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument("--preprocess_prompt", type=str2bool, default=True)
    parser.add_argument("--postprocess_output", type=str2bool, default=True)
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument("--audio_chunk_duration", type=float, default=15.0)
    parser.add_argument("--audio_chunk_threshold", type=float, default=30.0)
    parser.add_argument("--batched_decode", type=str2bool, default=True)
    parser.add_argument(
        "--enforce_output_duration",
        type=str2bool,
        default=False,
        help=(
            "When samples provide duration, crop or right-pad the final waveform "
            "after post-processing so saved WAVs match the requested duration."
        ),
    )
    parser.add_argument("--batch_size_pad", type=int, default=None)
    parser.add_argument("--seq_len_bucket_multiple", type=int, default=1)
    parser.add_argument("--target_len_bucket_multiple", type=int, default=1)
    parser.add_argument("--reuse_static_input_embeds", type=str2bool, default=True)
    parser.add_argument("--split_guidance_forward", default="auto")
    parser.add_argument("--split_guidance_min_batch_size", type=int, default=8)
    parser.add_argument(
        "--split_guidance_min_saved_context_ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument("--warmup", type=int, default=0)
    return parser


def _dtype(name: str):
    if name == "float32":
        return torch.float32
    return torch.float16


def _effective_compile_mode(args) -> str:
    if (
        args.compile_mode == "reduce-overhead"
        and not args.allow_reduce_overhead_worker
    ):
        logging.warning(
            "torch.compile(mode='reduce-overhead') uses CUDA graphs and can "
            "crash when new shapes execute inside the online scheduler worker "
            "thread. Downgrading to compile_mode='default'. Set "
            "--allow_reduce_overhead_worker true only for controlled static-shape "
            "benchmarks."
        )
        return "default"
    return args.compile_mode


def _scheduler_config_from_args(args) -> BatchSchedulerConfig:
    return BatchSchedulerConfig(
        max_batch_size=args.batch_size,
        max_wait_ms=args.max_wait_ms,
        partial_batch_floor=args.partial_batch_floor,
        max_total_target_tokens=args.max_total_target_tokens,
        max_total_context_tokens=args.max_total_context_tokens,
        max_cost_ratio=args.max_cost_ratio,
        max_context_ratio=args.max_context_ratio,
        max_context_padding_ratio=args.max_context_padding_ratio,
        ready_queue_capacity=args.ready_queue_capacity,
        control_queue_capacity=args.control_queue_capacity,
        prompt_cache_entries=args.prompt_cache_entries,
        use_model_duration_estimator=args.use_model_duration_estimator,
        lookahead_for_full_batch=args.lookahead_for_full_batch,
        lookahead_for_partial_batch=args.lookahead_for_partial_batch,
        partial_lookahead_max_wait_multiplier=(
            args.partial_lookahead_max_wait_multiplier
        ),
        max_seed_lookahead=args.max_seed_lookahead,
        candidate_pack_policy=args.candidate_pack_policy,
        split_retry_on_memory_error=args.split_retry_on_memory_error,
        adaptive_memory_batch_cap=args.adaptive_memory_batch_cap,
        adaptive_memory_cap_recovery_successes=(
            args.adaptive_memory_cap_recovery_successes
        ),
        max_generation_batches_before_control=(
            args.max_generation_batches_before_control
        ),
    )


def _request_from_sample(sample: dict[str, Any]) -> OmniVoiceBatchRequest:
    return OmniVoiceBatchRequest(
        request_id=require_sample_id(sample),
        text=sample["text"],
        language=sample.get("language_id") or sample.get("language"),
        ref_audio=sample.get("ref_audio"),
        ref_text=sample.get("ref_text"),
        instruct=sample.get("instruct"),
        duration=sample.get("duration"),
        speed=sample.get("speed"),
        enforce_output_duration=sample.get("enforce_output_duration"),
        cost_tokens_hint=sample.get("cost_tokens_hint"),
        priority=sample.get("priority", "normal"),
    )


def _first_compatible_warmup_samples(samples: list[dict[str, Any]], limit: int):
    selected = samples[:limit]
    if not selected:
        return []
    first_is_clone = selected[0].get("ref_audio") is not None
    first_is_design = selected[0].get("instruct") is not None
    if first_is_clone:
        return [s for s in selected if s.get("ref_audio") is not None]
    if first_is_design:
        return [
            s
            for s in selected
            if s.get("ref_audio") is None and s.get("instruct") is not None
        ]
    return [
        s
        for s in selected
        if s.get("ref_audio") is None and s.get("instruct") is None
    ]


def _select_representative_warmup_samples(
    samples: list[dict[str, Any]],
    config: BatchSchedulerConfig,
) -> list[dict[str, Any]]:
    """Select the warmup batch that the online scheduler is likely to run.

    Static-shape compilation only pays off when warmup sees the same kind of
    batch as steady-state serving.  The naive "first N samples" warmup can miss
    the cost-bucket scheduler's first full batch when a young outlier is at the
    queue head, so this mirrors the scheduler's seed/bucket scoring without
    starting the background serving thread.
    """
    if not samples:
        return []

    scheduler = OmniVoiceBatchScheduler(model=None, config=config)
    candidates = []
    for index, sample in enumerate(samples):
        request = _request_from_sample(sample)
        cost = scheduler._estimate_cost_tokens(request)
        candidates.append(
            {
                "index": index,
                "sample": sample,
                "request": request,
                "mode": scheduler._mode_key(request),
                "cost": cost,
                "context": scheduler._estimate_context_tokens(
                    request,
                    cost_tokens=cost,
                ),
            }
        )

    high = [item for item in candidates if item["request"].priority == "high"]
    normal = [item for item in candidates if item["request"].priority != "high"]
    search_queue = high if high else normal
    ordered = high + normal
    fifo_seed = search_queue[0]

    def is_compatible(seed, item) -> bool:
        if item["mode"] != seed["mode"]:
            return False
        low = max(1, min(seed["cost"], item["cost"]))
        high_cost = max(seed["cost"], item["cost"])
        if high_cost / low > config.max_cost_ratio:
            return False
        low_context = max(1, min(seed["context"], item["context"]))
        high_context = max(seed["context"], item["context"])
        return high_context / low_context <= config.max_context_ratio

    def candidate_batch(seed, *, sort_by_cost: bool):
        batch = [seed]
        total_cost = seed["cost"]
        total_context = seed["context"]
        max_context = seed["context"]
        others = [
            item for item in ordered if item is not seed and is_compatible(seed, item)
        ]
        if sort_by_cost:
            others.sort(
                key=lambda item: (
                    abs(item["cost"] - seed["cost"]),
                    abs(item["context"] - seed["context"]),
                    item["index"],
                )
            )
        for item in others:
            if len(batch) >= config.max_batch_size:
                break
            if total_cost + item["cost"] > config.max_total_target_tokens:
                continue
            if total_context + item["context"] > config.max_total_context_tokens:
                continue
            next_total_context = total_context + item["context"]
            next_max_context = max(max_context, item["context"])
            if (
                config.max_context_padding_ratio > 0.0
                and (len(batch) + 1) * next_max_context / next_total_context
                > config.max_context_padding_ratio
            ):
                continue
            batch.append(item)
            total_cost += item["cost"]
            total_context = next_total_context
            max_context = next_max_context
        return batch

    def score(batch):
        costs = [item["cost"] for item in batch]
        contexts = [item["context"] for item in batch]
        cost_span = max(costs) - min(costs) if costs else 0
        context_span = max(contexts) - min(contexts) if contexts else 0
        context_padding_work = len(contexts) * max(contexts) if contexts else 0
        oldest_index = min((item["index"] for item in batch), default=0)
        return (
            len(batch),
            -cost_span,
            -context_span,
            -context_padding_work,
            -oldest_index,
        )

    fifo_batch = candidate_batch(fifo_seed, sort_by_cost=False)
    if not config.lookahead_for_full_batch:
        return [item["sample"] for item in fifo_batch]

    best_batch = fifo_batch
    best_score = score(best_batch)
    for candidate in search_queue[: config.max_seed_lookahead]:
        batch = candidate_batch(candidate, sort_by_cost=True)
        batch_score = score(batch)
        if batch_score > best_score:
            best_batch = batch
            best_score = batch_score

    if len(best_batch) >= config.max_batch_size:
        return [item["sample"] for item in best_batch]
    if (
        config.lookahead_for_partial_batch
        and len(best_batch) >= config.partial_batch_floor
        and best_score > score(fifo_batch)
    ):
        return [item["sample"] for item in best_batch]
    return [item["sample"] for item in fifo_batch]


def _select_representative_warmup_batches(
    samples: list[dict[str, Any]],
    config: BatchSchedulerConfig,
    max_batches: int,
    *,
    fill_batch: bool = False,
) -> list[list[dict[str, Any]]]:
    remaining = list(samples)
    batches = []
    for _ in range(max(1, max_batches)):
        batch = _select_representative_warmup_samples(remaining, config)
        if not batch:
            break
        batches.append(_fill_warmup_batch(batch, config) if fill_batch else batch)
        selected_ids = {id(sample) for sample in batch}
        remaining = [sample for sample in remaining if id(sample) not in selected_ids]
        if not remaining:
            break
    return batches


def _fill_warmup_batch(
    batch: list[dict[str, Any]],
    config: BatchSchedulerConfig,
) -> list[dict[str, Any]]:
    if not batch or len(batch) >= config.max_batch_size:
        return batch
    scheduler = OmniVoiceBatchScheduler(model=None, config=config)
    filled = list(batch)

    def item_cost_context(sample: dict[str, Any]) -> tuple[int, int]:
        request = _request_from_sample(sample)
        cost = scheduler._estimate_cost_tokens(request)
        context = scheduler._estimate_context_tokens(request, cost_tokens=cost)
        return cost, context

    costs_contexts = [item_cost_context(sample) for sample in filled]
    total_cost = sum(cost for cost, _ in costs_contexts)
    total_context = sum(context for _, context in costs_contexts)
    max_context = max((context for _, context in costs_contexts), default=0)

    def can_add(sample: dict[str, Any]) -> bool:
        cost, context = item_cost_context(sample)
        if total_cost + cost > config.max_total_target_tokens:
            return False
        if total_context + context > config.max_total_context_tokens:
            return False
        next_total_context = total_context + context
        next_max_context = max(max_context, context)
        if (
            config.max_context_padding_ratio > 0.0
            and (len(filled) + 1) * next_max_context / next_total_context
            > config.max_context_padding_ratio
        ):
            return False
        return True

    index = 0
    consecutive_rejections = 0
    while len(filled) < config.max_batch_size:
        source = batch[index % len(batch)]
        clone = dict(source)
        base_id = clone.get("id") or clone.get("save_name") or "warmup"
        clone["id"] = f"{base_id}__warmup_fill_{len(filled) + 1}"
        if not can_add(clone):
            consecutive_rejections += 1
            if consecutive_rejections >= len(batch):
                break
            index += 1
            continue
        cost, context = item_cost_context(clone)
        filled.append(clone)
        total_cost += cost
        total_context += context
        max_context = max(max_context, context)
        consecutive_rejections = 0
        index += 1
    return filled


async def _run_scheduler_warmup(
    scheduler: OmniVoiceBatchScheduler,
    warmup_batches: list[list[dict[str, Any]]],
) -> None:
    for batch_index, warmup_samples in enumerate(warmup_batches, start=1):
        logging.info(
            "Running scheduler-path warmup batch %d/%d with %d sample(s)",
            batch_index,
            len(warmup_batches),
            len(warmup_samples),
        )
        await asyncio.gather(
            *[
                scheduler.submit(_request_from_sample(sample))
                for sample in warmup_samples
            ]
        )
    scheduler.reset_metrics()


async def _run(args) -> dict[str, Any]:
    os.makedirs(args.res_dir, exist_ok=True)
    device = args.device or get_best_device()
    logging.info("Loading model from %s on %s", args.model, device)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=_dtype(args.dtype),
    )
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    compile_mode = _effective_compile_mode(args)
    if args.compile_llm:
        logging.info("Compiling LLM with torch.compile(mode=%s)", compile_mode)
        model.llm = torch.compile(
            model.llm,
            mode=compile_mode,
            fullgraph=False,
        )
    if args.compile_audio_heads:
        logging.info(
            "Compiling audio heads with torch.compile(mode=%s)",
            compile_mode,
        )
        model.audio_heads = torch.compile(
            model.audio_heads,
            mode=compile_mode,
            fullgraph=False,
        )

    generation_kwargs = {
        "generation_mode": args.generation_mode,
        "num_step": args.num_step,
        "guidance_scale": args.guidance_scale,
        "t_shift": args.t_shift,
        "denoise": args.denoise,
        "preprocess_prompt": args.preprocess_prompt,
        "postprocess_output": args.postprocess_output,
        "layer_penalty_factor": args.layer_penalty_factor,
        "position_temperature": args.position_temperature,
        "class_temperature": args.class_temperature,
        "audio_chunk_duration": args.audio_chunk_duration,
        "audio_chunk_threshold": args.audio_chunk_threshold,
        "batched_decode": args.batched_decode,
        "enforce_output_duration": args.enforce_output_duration,
        "batch_size_pad": args.batch_size_pad,
        "seq_len_bucket_multiple": args.seq_len_bucket_multiple,
        "target_len_bucket_multiple": args.target_len_bucket_multiple,
        "reuse_static_input_embeds": args.reuse_static_input_embeds,
        "split_guidance_forward": args.split_guidance_forward,
        "split_guidance_min_batch_size": args.split_guidance_min_batch_size,
        "split_guidance_min_saved_context_ratio": (
            args.split_guidance_min_saved_context_ratio
        ),
    }
    samples = read_test_list(args.test_list)
    scheduler_config = _scheduler_config_from_args(args)
    scheduler = OmniVoiceBatchScheduler(
        model,
        config=scheduler_config,
        generation_kwargs=generation_kwargs,
    )
    scheduler_started = False

    async def ensure_scheduler_started() -> None:
        nonlocal scheduler_started
        if scheduler_started:
            return
        await scheduler.start()
        scheduler_started = True

    try:
        if args.warmup > 0:
            logging.info("Running %d warmup iterations", args.warmup)
            dummy_ref_audio = (torch.randn(1, 24000), 24000)
            for _ in range(args.warmup):
                model.generate(
                    text=["hello"],
                    language=["en"],
                    ref_audio=[dummy_ref_audio],
                    ref_text=["hello"],
                    **generation_kwargs,
                )
        if args.representative_warmup and samples:
            warmup_batches = _select_representative_warmup_batches(
                samples,
                scheduler_config,
                args.representative_warmup_batches,
                fill_batch=args.representative_warmup_fill_batch,
            )
            logging.info(
                "Running scheduler-aware representative warmup with %d batch(es), %d sample(s)",
                len(warmup_batches),
                sum(len(batch) for batch in warmup_batches),
            )
            if compile_mode == "reduce-overhead":
                logging.info(
                    "Using scheduler worker thread for representative warmup "
                    "because torch.compile(mode='reduce-overhead') captures "
                    "thread-local CUDA graphs."
                )
            else:
                logging.info(
                    "Using scheduler worker thread for representative warmup "
                    "to exercise the same prompt-cache and batching path as serving."
                )
            await ensure_scheduler_started()
            await _run_scheduler_warmup(scheduler, warmup_batches)

        await ensure_scheduler_started()

        sem = asyncio.Semaphore(args.concurrency)
        started = time.monotonic()
        results = []

        async def submit_one(index: int, sample: dict[str, Any]):
            if args.arrival_gap_ms > 0:
                await asyncio.sleep(index * args.arrival_gap_ms / 1000.0)
            async with sem:
                request_id = require_sample_id(sample)
                request_started = time.monotonic()
                try:
                    req = _request_from_sample(sample)
                    result = await scheduler.submit(req)
                    audio_s = len(result.audio) / result.sample_rate
                    if audio_s <= 0.0:
                        raise ValueError("invalid_audio_result")
                    out_path = Path(args.res_dir) / f"{req.request_id}.wav"
                    sf.write(out_path, result.audio, result.sample_rate)
                    return {
                        "id": req.request_id,
                        "success": True,
                        "wav": str(out_path),
                        "audio_s": audio_s,
                        "batch_size": result.batch_size,
                        "queue_wait_ms": result.queue_wait_ms,
                        "batch_infer_s": result.batch_infer_s,
                        "batch_reason": result.batch_reason,
                        "batch_cost_tokens": result.batch_cost_tokens,
                        "batch_max_cost_tokens": result.batch_max_cost_tokens,
                        "batch_context_tokens": result.batch_context_tokens,
                        "batch_max_context_tokens": (
                            result.batch_max_context_tokens
                        ),
                        "batch_context_padding_ratio": (
                            result.batch_context_padding_ratio
                        ),
                    }
                except Exception as exc:
                    return request_exception_result(
                        request_id=request_id,
                        exc=exc,
                        request_wall_s=time.monotonic() - request_started,
                    )

        tasks = [asyncio.create_task(submit_one(i, s)) for i, s in enumerate(samples)]
        for item in await asyncio.gather(*tasks):
            results.append(item)
    finally:
        if scheduler_started:
            await scheduler.stop()

    wall_s = time.monotonic() - started
    snapshot = scheduler.snapshot()
    successful = successful_request_results(results)
    audio_s = sum(item["audio_s"] for item in successful)
    summary = {
        "num_requests": len(results),
        "num_successful": len(successful),
        "num_failed": len(results) - len(successful),
        "wall_s": wall_s,
        "audio_s": audio_s,
        "rtf_wall": wall_s / audio_s if audio_s > 0 else None,
        "scheduler": snapshot.__dict__,
        "request_metrics": summarize_request_results(
            successful,
            batch_size_key="batch_size",
            infer_s_key="batch_infer_s",
        ),
        "results": results,
    }
    summary_path = Path(args.res_dir) / "online_batch_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    args = get_parser().parse_args()
    summary = asyncio.run(_run(args))
    logging.info(
        (
            "Done: requests=%d successful=%d failed=%d wall=%.3fs "
            "audio=%.3fs avg_batch=%.2f summary=%s"
        ),
        summary["num_requests"],
        summary["num_successful"],
        summary["num_failed"],
        summary["wall_s"],
        summary["audio_s"],
        summary["scheduler"]["avg_batch_size"],
        Path(args.res_dir) / "online_batch_summary.json",
    )


if __name__ == "__main__":
    main()
