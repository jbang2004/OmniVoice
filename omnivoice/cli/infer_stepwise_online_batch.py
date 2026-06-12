#!/usr/bin/env python3
"""Experimental step-level online batching benchmark for OmniVoice."""

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

from omnivoice.cli.benchmark_utils import summarize_request_results
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.serving.batcher import OmniVoiceBatchRequest
from omnivoice.serving.stepwise import create_stepwise_states, run_generation_step
from omnivoice.serving.stepwise_batcher import (
    StepwiseOmniVoiceScheduler,
    StepwiseSchedulerConfig,
    build_static_step_shape,
)
from omnivoice.utils.common import get_best_device, str2bool
from omnivoice.utils.data_utils import read_test_list


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run experimental step-level online OmniVoice batching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--test_list", required=True)
    parser.add_argument("--res_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--compile_llm", type=str2bool, default=False)
    parser.add_argument("--compile_audio_heads", type=str2bool, default=False)
    parser.add_argument(
        "--compile_mode",
        default="default",
        help="Use default for the threaded stepwise scheduler; reduce-overhead uses CUDA graphs and can fail across threads.",
    )
    parser.add_argument(
        "--allow_reduce_overhead_worker",
        type=str2bool,
        default=False,
        help=(
            "Allow torch.compile(mode='reduce-overhead') in the stepwise worker "
            "thread. This is experimental and can fail with CUDA graph TLS "
            "assertions on heterogeneous traffic."
        ),
    )
    parser.add_argument("--matmul_precision", default=None)
    parser.add_argument(
        "--representative_warmup",
        type=str2bool,
        default=False,
        help="Warm up with the first real max-running batch from test_list.",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--arrival_gap_ms", type=float, default=0.0)
    parser.add_argument("--max_running_requests", type=int, default=8)
    parser.add_argument("--max_wait_ms", type=float, default=120.0)
    parser.add_argument("--partial_batch_floor", type=int, default=2)
    parser.add_argument("--max_total_target_tokens", type=int, default=4096)
    parser.add_argument("--max_total_context_tokens", type=int, default=8192)
    parser.add_argument("--max_cost_ratio", type=float, default=2.0)
    parser.add_argument("--max_context_ratio", type=float, default=2.0)
    parser.add_argument("--max_context_padding_ratio", type=float, default=2.0)
    parser.add_argument("--ready_queue_capacity", type=int, default=64)
    parser.add_argument("--control_queue_capacity", type=int, default=16)
    parser.add_argument("--prompt_cache_entries", type=int, default=128)
    parser.add_argument(
        "--pre_register_voices",
        type=str2bool,
        default=False,
        help=(
            "Create one voice_clone_prompt per unique ref_audio/ref_text before "
            "timed submissions, then pass prompts directly to stepwise requests."
        ),
    )
    parser.add_argument("--lookahead_for_full_batch", type=str2bool, default=True)
    parser.add_argument("--max_seed_lookahead", type=int, default=32)
    parser.add_argument(
        "--profile_cuda",
        type=str2bool,
        default=False,
        help="Synchronize CUDA around pack/forward/update timing probes.",
    )
    parser.add_argument(
        "--compile_static_shape",
        type=str2bool,
        default=False,
        help="Pad stepwise execution to fixed request slots and bucketed lengths to reduce torch.compile graph churn.",
    )
    parser.add_argument(
        "--seq_len_bucket_multiple",
        type=int,
        default=64,
        help="Context length bucket size used when --compile_static_shape is true.",
    )
    parser.add_argument(
        "--target_len_bucket_multiple",
        type=int,
        default=64,
        help="Target audio length bucket size used when --compile_static_shape is true.",
    )

    parser.add_argument("--num_step", type=int, default=32)
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
    parser.add_argument("--warmup", type=int, default=0)
    return parser


def _dtype(name: str):
    return torch.float32 if name == "float32" else torch.float16


def _effective_compile_mode(args) -> str:
    if (
        args.compile_mode == "reduce-overhead"
        and not args.allow_reduce_overhead_worker
    ):
        logging.warning(
            "torch.compile(mode='reduce-overhead') uses CUDA graphs and can "
            "crash inside the stepwise scheduler worker thread. Downgrading to "
            "compile_mode='default'. Set --allow_reduce_overhead_worker true "
            "only for controlled experiments."
        )
        return "default"
    return args.compile_mode


def _request_from_sample(sample: dict[str, Any]) -> OmniVoiceBatchRequest:
    return OmniVoiceBatchRequest(
        request_id=str(sample.get("id") or sample.get("save_name")),
        text=sample["text"],
        language=sample.get("language_id") or sample.get("language"),
        ref_audio=sample.get("ref_audio"),
        ref_text=sample.get("ref_text"),
        voice_clone_prompt=sample.get("voice_clone_prompt"),
        instruct=sample.get("instruct"),
        duration=sample.get("duration"),
        speed=sample.get("speed"),
        priority=sample.get("priority", "normal"),
    )


def _local_voice_registration_key(
    sample: dict[str, Any],
    *,
    preprocess_prompt: bool,
) -> tuple[Any, ...] | None:
    if sample.get("voice_clone_prompt") is not None:
        return None
    ref_audio = sample.get("ref_audio")
    if ref_audio is None:
        return None
    return (ref_audio, sample.get("ref_text"), preprocess_prompt)


def _sample_with_local_voice_prompt(
    sample: dict[str, Any],
    *,
    voice_id: str,
    voice_clone_prompt: Any,
) -> dict[str, Any]:
    rewritten = dict(sample)
    rewritten.pop("ref_audio", None)
    rewritten.pop("ref_text", None)
    rewritten["voice_id"] = voice_id
    rewritten["voice_clone_prompt"] = voice_clone_prompt
    return rewritten


async def _pre_register_local_voice_prompts(
    *,
    scheduler: StepwiseOmniVoiceScheduler,
    samples: list[dict[str, Any]],
    preprocess_prompt: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    registrations: dict[tuple[Any, ...], tuple[str, Any]] = {}
    rewritten_samples: list[dict[str, Any]] = []
    num_rewritten = 0

    for sample in samples:
        key = _local_voice_registration_key(
            sample,
            preprocess_prompt=preprocess_prompt,
        )
        if key is None:
            rewritten_samples.append(dict(sample))
            continue

        if key not in registrations:
            voice_id = f"local_voice_{len(registrations):04d}"
            prompt = await scheduler.create_voice_clone_prompt(
                ref_audio=sample["ref_audio"],
                ref_text=sample.get("ref_text"),
                preprocess_prompt=preprocess_prompt,
            )
            registrations[key] = (voice_id, prompt)

        voice_id, prompt = registrations[key]
        rewritten_samples.append(
            _sample_with_local_voice_prompt(
                sample,
                voice_id=voice_id,
                voice_clone_prompt=prompt,
            )
        )
        num_rewritten += 1

    return rewritten_samples, {
        "enabled": True,
        "num_registered_voices": len(registrations),
        "num_rewritten_samples": num_rewritten,
        "registration_s": time.monotonic() - started,
    }


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


def _generate_warmup_kwargs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "text": [s["text"] for s in samples],
        "language": [s.get("language_id") or s.get("language") for s in samples],
    }
    if any(s.get("ref_audio") is not None for s in samples):
        kwargs["ref_audio"] = [s.get("ref_audio") for s in samples]
        kwargs["ref_text"] = [s.get("ref_text") for s in samples]
    elif any(s.get("instruct") is not None for s in samples):
        kwargs["instruct"] = [s.get("instruct") for s in samples]
    if any(s.get("duration") is not None for s in samples):
        kwargs["duration"] = [s.get("duration") for s in samples]
    if any(s.get("speed") is not None for s in samples):
        kwargs["speed"] = [s.get("speed") for s in samples]
    return kwargs


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
        logging.warning(
            "compile_llm is experimental for stepwise scheduling. Dynamic "
            "active-batch shapes can trigger recompilation. Use "
            "--compile_static_shape true to pad request slots and bucket "
            "lengths, or prefer compiled online micro-batch for production "
            "throughput."
        )
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
    generation_config = OmniVoiceGenerationConfig(
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        denoise=args.denoise,
        preprocess_prompt=args.preprocess_prompt,
        postprocess_output=args.postprocess_output,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
        audio_chunk_duration=args.audio_chunk_duration,
        audio_chunk_threshold=args.audio_chunk_threshold,
    )
    samples = read_test_list(args.test_list)
    scheduler_config = StepwiseSchedulerConfig(
        max_running_requests=args.max_running_requests,
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
        profile_cuda=args.profile_cuda,
        compile_static_shape=args.compile_static_shape,
        seq_len_bucket_multiple=args.seq_len_bucket_multiple,
        target_len_bucket_multiple=args.target_len_bucket_multiple,
        lookahead_for_full_batch=args.lookahead_for_full_batch,
        max_seed_lookahead=args.max_seed_lookahead,
    )

    if args.warmup > 0:
        logging.info("Running %d warmup iterations", args.warmup)
        dummy_ref_audio = (torch.randn(1, 24000), 24000)
        for _ in range(args.warmup):
            model.generate(
                text=["hello"],
                language=["en"],
                ref_audio=[dummy_ref_audio],
                ref_text=["hello"],
                num_step=args.num_step,
            )
    if args.representative_warmup and samples:
        warmup_samples = _first_compatible_warmup_samples(
            samples,
            args.max_running_requests,
        )
        logging.info(
            "Running representative stepwise warmup with %d samples",
            len(warmup_samples),
        )
        warmup_kwargs = _generate_warmup_kwargs(warmup_samples)
        warmup_kwargs["preprocess_prompt"] = generation_config.preprocess_prompt
        with torch.inference_mode():
            warmup_task = model._preprocess_all(**warmup_kwargs)
            warmup_states = create_stepwise_states(
                model,
                warmup_task,
                generation_config,
            )
            run_generation_step(
                model,
                warmup_states,
                generation_config,
                profile_cuda=args.profile_cuda,
                **build_static_step_shape(warmup_states, scheduler_config),
            )

    scheduler = StepwiseOmniVoiceScheduler(
        model,
        scheduler_config=scheduler_config,
        generation_config=generation_config,
    )
    await scheduler.start()
    voice_registration = {"enabled": False}
    if args.pre_register_voices:
        samples, voice_registration = await _pre_register_local_voice_prompts(
            scheduler=scheduler,
            samples=samples,
            preprocess_prompt=generation_config.preprocess_prompt,
        )

    sem = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()

    async def submit_one(index: int, sample: dict[str, Any]):
        if args.arrival_gap_ms > 0:
            await asyncio.sleep(index * args.arrival_gap_ms / 1000.0)
        async with sem:
            req = _request_from_sample(sample)
            result = await scheduler.submit(req)
            out_path = Path(args.res_dir) / f"{req.request_id}.wav"
            sf.write(out_path, result.audio, result.sample_rate)
            return {
                "id": req.request_id,
                "wav": str(out_path),
                "audio_s": len(result.audio) / result.sample_rate,
                "max_step_batch_size": result.batch_size,
                "queue_wait_ms": result.queue_wait_ms,
                "infer_s": result.batch_infer_s,
            }

    try:
        results = await asyncio.gather(
            *[asyncio.create_task(submit_one(i, s)) for i, s in enumerate(samples)]
        )
    finally:
        await scheduler.stop()

    wall_s = time.monotonic() - started
    audio_s = sum(item["audio_s"] for item in results)
    summary = {
        "num_requests": len(results),
        "wall_s": wall_s,
        "audio_s": audio_s,
        "rtf_wall": wall_s / audio_s if audio_s > 0 else None,
        "voice_registration": voice_registration,
        "scheduler": scheduler.snapshot().__dict__,
        "request_metrics": summarize_request_results(
            results,
            batch_size_key="max_step_batch_size",
            infer_s_key="infer_s",
            reason_key=None,
        ),
        "results": results,
    }
    summary_path = Path(args.res_dir) / "stepwise_online_summary.json"
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
        "Done: requests=%d wall=%.3fs audio=%.3fs avg_step_batch=%.2f summary=%s",
        summary["num_requests"],
        summary["wall_s"],
        summary["audio_s"],
        summary["scheduler"]["avg_step_batch_size"],
        Path(args.res_dir) / "stepwise_online_summary.json",
    )


if __name__ == "__main__":
    main()
