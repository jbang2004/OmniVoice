#!/usr/bin/env python3
"""HTTP server for online micro-batched OmniVoice inference."""

from __future__ import annotations

import argparse
import logging

import torch

from omnivoice.cli.infer_online_batch import (
    _effective_compile_mode,
    _run_scheduler_warmup,
    _select_representative_warmup_batches,
)
from omnivoice.models.omnivoice import OmniVoice
from omnivoice.serving import (
    SERVING_PROFILES,
    BatchSchedulerConfig,
    OmniVoiceBatchScheduler,
    get_serving_profile,
)
from omnivoice.serving.http_server import (
    OnlineBatchServerState,
    create_online_batch_app,
)
from omnivoice.utils.common import get_best_device, str2bool
from omnivoice.utils.data_utils import read_test_list


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve OmniVoice with a single-GPU online batch scheduler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--compile_llm", type=str2bool, default=True)
    parser.add_argument("--compile_audio_heads", type=str2bool, default=False)
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--allow_reduce_overhead_worker", type=str2bool, default=False)
    parser.add_argument("--matmul_precision", default="high")

    parser.add_argument(
        "--scheduler_profile",
        choices=sorted(SERVING_PROFILES),
        default="balanced12",
        help="Named scheduler preset. Explicit scheduler flags override the preset.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_wait_ms", type=float, default=None)
    parser.add_argument("--partial_batch_floor", type=int, default=None)
    parser.add_argument("--max_total_target_tokens", type=int, default=None)
    parser.add_argument("--max_total_context_tokens", type=int, default=None)
    parser.add_argument("--max_cost_ratio", type=float, default=None)
    parser.add_argument("--max_context_ratio", type=float, default=None)
    parser.add_argument("--max_context_padding_ratio", type=float, default=None)
    parser.add_argument("--ready_queue_capacity", type=int, default=None)
    parser.add_argument("--control_queue_capacity", type=int, default=None)
    parser.add_argument("--prompt_cache_entries", type=int, default=None)
    parser.add_argument("--use_model_duration_estimator", type=str2bool, default=None)
    parser.add_argument("--lookahead_for_full_batch", type=str2bool, default=None)
    parser.add_argument("--lookahead_for_partial_batch", type=str2bool, default=None)
    parser.add_argument(
        "--partial_lookahead_max_wait_multiplier",
        type=float,
        default=None,
    )
    parser.add_argument("--max_seed_lookahead", type=int, default=None)
    parser.add_argument(
        "--candidate_pack_policy",
        choices=["target", "context", "target_context"],
        default=None,
        help=(
            "Candidate ordering policy for lookahead packing. 'target' is the "
            "original target-length-first policy; 'target_context' also tries "
            "context-length-first packing and keeps the better batch."
        ),
    )
    parser.add_argument("--split_retry_on_memory_error", type=str2bool, default=None)
    parser.add_argument("--adaptive_memory_batch_cap", type=str2bool, default=None)
    parser.add_argument("--adaptive_memory_cap_recovery_successes", type=int, default=None)
    parser.add_argument("--max_generation_batches_before_control", type=int, default=None)

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
    parser.add_argument("--batched_decode", type=str2bool, default=True)
    parser.add_argument("--batch_size_pad", type=int, default=None)
    parser.add_argument("--seq_len_bucket_multiple", type=int, default=1)
    parser.add_argument("--target_len_bucket_multiple", type=int, default=1)
    parser.add_argument("--collect_profile", type=str2bool, default=False)
    parser.add_argument("--reuse_static_input_embeds", type=str2bool, default=True)
    parser.add_argument("--split_guidance_forward", default="auto")
    parser.add_argument("--split_guidance_min_batch_size", type=int, default=8)
    parser.add_argument(
        "--split_guidance_min_saved_context_ratio",
        type=float,
        default=0.25,
    )

    parser.add_argument("--warmup_test_list", default=None)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument(
        "--warmup_fill_batch",
        type=str2bool,
        default=False,
        help=(
            "Repeat compatible warmup samples until each selected warmup batch "
            "reaches the scheduler max batch size. Useful with torch.compile "
            "when the representative JSONL is smaller than the target serving "
            "profile."
        ),
    )
    parser.add_argument("--max_request_text_chars", type=int, default=2000)
    parser.add_argument("--log_level", default="info")
    return parser


def _dtype(name: str):
    if name == "float32":
        return torch.float32
    return torch.float16


def _generation_kwargs(args) -> dict:
    return {
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
        "batch_size_pad": args.batch_size_pad,
        "seq_len_bucket_multiple": args.seq_len_bucket_multiple,
        "target_len_bucket_multiple": args.target_len_bucket_multiple,
        "collect_profile": args.collect_profile,
        "reuse_static_input_embeds": args.reuse_static_input_embeds,
        "split_guidance_forward": args.split_guidance_forward,
        "split_guidance_min_batch_size": args.split_guidance_min_batch_size,
        "split_guidance_min_saved_context_ratio": (
            args.split_guidance_min_saved_context_ratio
        ),
    }


def _scheduler_value(args, profile_defaults: dict, attr: str, field: str):
    value = getattr(args, attr, None)
    if value is not None:
        return value
    if field in profile_defaults:
        return profile_defaults[field]
    return getattr(BatchSchedulerConfig(), field)


def _scheduler_config(args) -> BatchSchedulerConfig:
    profile = get_serving_profile(getattr(args, "scheduler_profile", "balanced12"))
    profile_defaults = profile.scheduler
    return BatchSchedulerConfig(
        max_batch_size=_scheduler_value(
            args, profile_defaults, "batch_size", "max_batch_size"
        ),
        max_wait_ms=_scheduler_value(args, profile_defaults, "max_wait_ms", "max_wait_ms"),
        partial_batch_floor=_scheduler_value(
            args, profile_defaults, "partial_batch_floor", "partial_batch_floor"
        ),
        max_total_target_tokens=_scheduler_value(
            args,
            profile_defaults,
            "max_total_target_tokens",
            "max_total_target_tokens",
        ),
        max_total_context_tokens=_scheduler_value(
            args,
            profile_defaults,
            "max_total_context_tokens",
            "max_total_context_tokens",
        ),
        max_cost_ratio=_scheduler_value(
            args, profile_defaults, "max_cost_ratio", "max_cost_ratio"
        ),
        max_context_ratio=_scheduler_value(
            args, profile_defaults, "max_context_ratio", "max_context_ratio"
        ),
        max_context_padding_ratio=_scheduler_value(
            args,
            profile_defaults,
            "max_context_padding_ratio",
            "max_context_padding_ratio",
        ),
        ready_queue_capacity=_scheduler_value(
            args, profile_defaults, "ready_queue_capacity", "ready_queue_capacity"
        ),
        control_queue_capacity=_scheduler_value(
            args,
            profile_defaults,
            "control_queue_capacity",
            "control_queue_capacity",
        ),
        prompt_cache_entries=_scheduler_value(
            args, profile_defaults, "prompt_cache_entries", "prompt_cache_entries"
        ),
        use_model_duration_estimator=_scheduler_value(
            args,
            profile_defaults,
            "use_model_duration_estimator",
            "use_model_duration_estimator",
        ),
        lookahead_for_full_batch=_scheduler_value(
            args,
            profile_defaults,
            "lookahead_for_full_batch",
            "lookahead_for_full_batch",
        ),
        lookahead_for_partial_batch=_scheduler_value(
            args,
            profile_defaults,
            "lookahead_for_partial_batch",
            "lookahead_for_partial_batch",
        ),
        partial_lookahead_max_wait_multiplier=_scheduler_value(
            args,
            profile_defaults,
            "partial_lookahead_max_wait_multiplier",
            "partial_lookahead_max_wait_multiplier",
        ),
        max_seed_lookahead=_scheduler_value(
            args, profile_defaults, "max_seed_lookahead", "max_seed_lookahead"
        ),
        candidate_pack_policy=_scheduler_value(
            args,
            profile_defaults,
            "candidate_pack_policy",
            "candidate_pack_policy",
        ),
        split_retry_on_memory_error=_scheduler_value(
            args,
            profile_defaults,
            "split_retry_on_memory_error",
            "split_retry_on_memory_error",
        ),
        adaptive_memory_batch_cap=_scheduler_value(
            args,
            profile_defaults,
            "adaptive_memory_batch_cap",
            "adaptive_memory_batch_cap",
        ),
        adaptive_memory_cap_recovery_successes=(
            _scheduler_value(
                args,
                profile_defaults,
                "adaptive_memory_cap_recovery_successes",
                "adaptive_memory_cap_recovery_successes",
            )
        ),
        max_generation_batches_before_control=(
            _scheduler_value(
                args,
                profile_defaults,
                "max_generation_batches_before_control",
                "max_generation_batches_before_control",
            )
        ),
    )


def build_app(args):
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
        model.llm = torch.compile(model.llm, mode=compile_mode, fullgraph=False)
    if args.compile_audio_heads:
        logging.info("Compiling audio heads with torch.compile(mode=%s)", compile_mode)
        model.audio_heads = torch.compile(
            model.audio_heads,
            mode=compile_mode,
            fullgraph=False,
        )

    profile = get_serving_profile(getattr(args, "scheduler_profile", "balanced12"))
    scheduler_config = _scheduler_config(args)
    logging.info(
        "Using scheduler profile %s: %s; effective config=%s",
        profile.name,
        profile.description,
        scheduler_config,
    )
    scheduler = OmniVoiceBatchScheduler(
        model,
        config=scheduler_config,
        generation_kwargs=_generation_kwargs(args),
    )
    startup_warmup = None
    if args.warmup_test_list:
        samples = read_test_list(args.warmup_test_list)
        warmup_batches = _select_representative_warmup_batches(
            samples,
            scheduler_config,
            args.warmup_batches,
            fill_batch=args.warmup_fill_batch,
        )

        async def _warmup() -> None:
            logging.info(
                "Running startup scheduler warmup with %d batch(es), %d sample(s)",
                len(warmup_batches),
                sum(len(batch) for batch in warmup_batches),
            )
            await _run_scheduler_warmup(scheduler, warmup_batches)

        startup_warmup = _warmup

    state = OnlineBatchServerState(
        scheduler=scheduler,
        sample_rate=model.sampling_rate,
        max_request_text_chars=args.max_request_text_chars,
    )
    return create_online_batch_app(state, startup_warmup=startup_warmup)


def main() -> None:
    args = get_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        force=True,
    )
    app = build_app(args)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
