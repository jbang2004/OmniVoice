#!/usr/bin/env python3
"""Sweep OmniVoice online server configs and client load profiles."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

from omnivoice.cli.benchmark_http_sweep import _run as run_http_sweep
from omnivoice.cli.benchmark_utils import (
    build_server_sweep_profiles,
    parse_bool_grid,
    parse_float_grid,
    parse_int_grid,
    rank_http_sweep_results,
    rank_server_sweep_results,
    recommend_server_sweep_config,
    summarize_best_server_sweep_result,
)
from omnivoice.utils.common import str2bool


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start OmniVoice online batch servers with different scheduler/model "
            "configs, run the same HTTP load sweep against each, and rank the "
            "configs by success rate, wall RTF, queue latency, and batch fill."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--compile_llm", type=str2bool, default=True)
    parser.add_argument("--compile_audio_heads", type=str2bool, default=False)
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--allow_reduce_overhead_worker", type=str2bool, default=False)
    parser.add_argument("--matmul_precision", default="high")

    parser.add_argument("--batch_size_values", default="4,8,16")
    parser.add_argument("--max_wait_ms_values", default="20,40")
    parser.add_argument("--max_cost_ratio_values", default="1.2,1.4,1.8")
    parser.add_argument("--max_total_target_tokens_values", default="4096")
    parser.add_argument("--max_total_context_tokens", type=int, default=8192)
    parser.add_argument(
        "--max_total_context_tokens_values",
        default=None,
        help=(
            "Comma-separated max total context-token caps to sweep. When omitted, "
            "--max_total_context_tokens is used as a single fixed value."
        ),
    )
    parser.add_argument("--num_step_values", default="32")
    parser.add_argument("--partial_batch_floor", type=int, default=2)
    parser.add_argument("--ready_queue_capacity", type=int, default=128)
    parser.add_argument("--control_queue_capacity", type=int, default=16)
    parser.add_argument("--prompt_cache_entries", type=int, default=256)
    parser.add_argument("--use_model_duration_estimator", type=str2bool, default=True)
    parser.add_argument("--lookahead_for_full_batch", type=str2bool, default=True)
    parser.add_argument("--lookahead_for_partial_batch", type=str2bool, default=True)
    parser.add_argument(
        "--lookahead_for_partial_batch_values",
        default=None,
        help=(
            "Comma-separated booleans to sweep for timed-out partial-batch "
            "lookahead. When omitted, --lookahead_for_partial_batch is used."
        ),
    )
    parser.add_argument(
        "--partial_lookahead_max_wait_multiplier",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--partial_lookahead_max_wait_multiplier_values",
        default=None,
        help=(
            "Comma-separated hard FIFO wait multipliers to sweep for partial "
            "lookahead. When omitted, --partial_lookahead_max_wait_multiplier "
            "is used."
        ),
    )
    parser.add_argument("--max_seed_lookahead", type=int, default=64)
    parser.add_argument(
        "--candidate_pack_policy",
        choices=["target", "context", "target_context"],
        default="target_context",
        help=(
            "Candidate ordering policy for lookahead packing. Run separate "
            "sweeps with 'target' and 'target_context' for old/new packing A/B."
        ),
    )
    parser.add_argument("--split_retry_on_memory_error", type=str2bool, default=True)
    parser.add_argument("--adaptive_memory_batch_cap", type=str2bool, default=True)
    parser.add_argument("--adaptive_memory_cap_recovery_successes", type=int, default=64)
    parser.add_argument("--max_generation_batches_before_control", type=int, default=8)
    parser.add_argument("--max_context_ratio", type=float, default=2.0)
    parser.add_argument(
        "--max_context_ratio_values",
        default=None,
        help=(
            "Comma-separated context length ratios to sweep. When omitted, "
            "--max_context_ratio is used as a single fixed value."
        ),
    )
    parser.add_argument("--max_context_padding_ratio", type=float, default=2.0)
    parser.add_argument(
        "--max_context_padding_ratio_values",
        default=None,
        help=(
            "Comma-separated context padding-work ratios to sweep. This limits "
            "batch_size * max_context_tokens / sum_context_tokens. When "
            "omitted, --max_context_padding_ratio is used as a single fixed value."
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
    parser.add_argument("--warmup_fill_batch", type=str2bool, default=False)
    parser.add_argument("--max_request_text_chars", type=int, default=2000)
    parser.add_argument("--server_log_level", default="warning")
    parser.add_argument("--startup_timeout_s", type=float, default=180.0)
    parser.add_argument("--shutdown_timeout_s", type=float, default=20.0)

    parser.add_argument("--test_list", required=True)
    parser.add_argument("--res_dir", required=True)
    parser.add_argument("--client_concurrency_values", default="1,2,4,8,16")
    parser.add_argument("--client_arrival_gap_ms_values", default="0")
    parser.add_argument("--client_repeats", type=int, default=1)
    parser.add_argument(
        "--client_warmup_repeats",
        type=int,
        default=0,
        help=(
            "Run this many HTTP sweep warmup repeats per client profile before "
            "measured repeats. Warmup results are excluded from ranking."
        ),
    )
    parser.add_argument("--client_request_repeats", type=int, default=1)
    parser.add_argument("--client_timeout_s", type=float, default=120.0)
    parser.add_argument("--save_wavs", type=str2bool, default=False)
    parser.add_argument("--client_pre_register_voices", type=str2bool, default=False)
    parser.add_argument("--gpu_monitor", type=str2bool, default=True)
    parser.add_argument("--gpu_index", type=int, default=None)
    parser.add_argument("--gpu_sample_interval_ms", type=float, default=200.0)
    parser.add_argument("--target_gpu_util_percent", type=float, default=70.0)
    parser.add_argument("--high_queue_wait_ms", type=float, default=100.0)
    parser.add_argument("--balanced_rtf_tolerance", type=float, default=0.05)
    parser.add_argument("--quality_min_num_step", type=int, default=32)
    return parser


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _bool_cli(value: bool) -> str:
    return "true" if value else "false"


def _value_cli(value: Any) -> str:
    if isinstance(value, bool):
        return _bool_cli(value)
    return str(value)


def _maybe_extend(command: list[str], flag: str, value: Optional[Any]) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _float_value(raw: str) -> Optional[float]:
    raw = raw.strip()
    if raw in {"", "[Not Supported]", "N/A", "nan"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_nvidia_smi_gpu_rows(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        index = _float_value(parts[0])
        util = _float_value(parts[1])
        memory_used = _float_value(parts[2])
        memory_total = _float_value(parts[3])
        if index is None:
            continue
        rows.append(
            {
                "index": int(index),
                "gpu_util_percent": util,
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
            }
        )
    return rows


async def _query_gpu_samples_once(gpu_index: Optional[int]) -> list[dict[str, Any]]:
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())
    rows = _parse_nvidia_smi_gpu_rows(stdout.decode("utf-8", errors="replace"))
    if gpu_index is not None:
        rows = [row for row in rows if row["index"] == gpu_index]
    now = time.monotonic()
    for row in rows:
        row["sample_time_s"] = now
    return rows


def _distribution(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": max(ordered),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * quantile
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_gpu_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    util_values = [
        float(sample["gpu_util_percent"])
        for sample in samples
        if sample.get("gpu_util_percent") is not None
    ]
    memory_values = [
        float(sample["memory_used_mib"])
        for sample in samples
        if sample.get("memory_used_mib") is not None
    ]
    memory_total_values = [
        float(sample["memory_total_mib"])
        for sample in samples
        if sample.get("memory_total_mib") is not None
    ]
    indexes = sorted({sample["index"] for sample in samples if "index" in sample})
    return {
        "num_samples": len(samples),
        "gpu_indexes": indexes,
        "gpu_util_percent": _distribution(util_values),
        "memory_used_mib": _distribution(memory_values),
        "memory_total_mib": max(memory_total_values) if memory_total_values else None,
    }


class _GpuSampler:
    def __init__(
        self,
        *,
        enabled: bool,
        gpu_index: Optional[int],
        interval_s: float,
    ):
        self.enabled = enabled
        self.gpu_index = gpu_index
        self.interval_s = max(0.05, interval_s)
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        if not self.enabled:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        self._stop.set()
        if self._task is not None:
            await self._task
        return {
            "enabled": True,
            "summary": _summarize_gpu_samples(self.samples),
            "samples": self.samples,
            "errors": self.errors,
        }

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.extend(await _query_gpu_samples_once(self.gpu_index))
            except Exception as exc:
                self.errors.append(str(exc))
                if len(self.errors) >= 3:
                    return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass


def _build_server_command(args: argparse.Namespace, profile: dict[str, Any]) -> list[str]:
    command = [
        args.python,
        "-m",
        "omnivoice.cli.serve_online_batch",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--compile_llm",
        _bool_cli(args.compile_llm),
        "--compile_audio_heads",
        _bool_cli(args.compile_audio_heads),
        "--compile_mode",
        args.compile_mode,
        "--allow_reduce_overhead_worker",
        _bool_cli(args.allow_reduce_overhead_worker),
        "--matmul_precision",
        args.matmul_precision,
        "--batch_size",
        str(profile["batch_size"]),
        "--max_wait_ms",
        str(profile["max_wait_ms"]),
        "--partial_batch_floor",
        str(args.partial_batch_floor),
        "--max_total_target_tokens",
        str(profile["max_total_target_tokens"]),
        "--max_total_context_tokens",
        str(profile.get("max_total_context_tokens", args.max_total_context_tokens)),
        "--max_cost_ratio",
        str(profile["max_cost_ratio"]),
        "--max_context_ratio",
        str(profile.get("max_context_ratio", args.max_context_ratio)),
        "--max_context_padding_ratio",
        str(
            profile.get(
                "max_context_padding_ratio",
                args.max_context_padding_ratio,
            )
        ),
        "--ready_queue_capacity",
        str(args.ready_queue_capacity),
        "--control_queue_capacity",
        str(args.control_queue_capacity),
        "--prompt_cache_entries",
        str(args.prompt_cache_entries),
        "--use_model_duration_estimator",
        _bool_cli(args.use_model_duration_estimator),
        "--lookahead_for_full_batch",
        _bool_cli(args.lookahead_for_full_batch),
        "--lookahead_for_partial_batch",
        _bool_cli(
            profile.get(
                "lookahead_for_partial_batch",
                args.lookahead_for_partial_batch,
            )
        ),
        "--partial_lookahead_max_wait_multiplier",
        str(
            profile.get(
                "partial_lookahead_max_wait_multiplier",
                args.partial_lookahead_max_wait_multiplier,
            )
        ),
        "--max_seed_lookahead",
        str(args.max_seed_lookahead),
        "--candidate_pack_policy",
        str(getattr(args, "candidate_pack_policy", "target_context")),
        "--split_retry_on_memory_error",
        _bool_cli(args.split_retry_on_memory_error),
        "--adaptive_memory_batch_cap",
        _bool_cli(args.adaptive_memory_batch_cap),
        "--adaptive_memory_cap_recovery_successes",
        str(args.adaptive_memory_cap_recovery_successes),
        "--max_generation_batches_before_control",
        str(args.max_generation_batches_before_control),
        "--num_step",
        str(profile["num_step"]),
        "--guidance_scale",
        str(args.guidance_scale),
        "--t_shift",
        str(args.t_shift),
        "--denoise",
        _bool_cli(args.denoise),
        "--preprocess_prompt",
        _bool_cli(args.preprocess_prompt),
        "--postprocess_output",
        _bool_cli(args.postprocess_output),
        "--layer_penalty_factor",
        str(args.layer_penalty_factor),
        "--position_temperature",
        str(args.position_temperature),
        "--class_temperature",
        str(args.class_temperature),
        "--audio_chunk_duration",
        str(args.audio_chunk_duration),
        "--audio_chunk_threshold",
        str(args.audio_chunk_threshold),
        "--batched_decode",
        _bool_cli(args.batched_decode),
        "--seq_len_bucket_multiple",
        str(args.seq_len_bucket_multiple),
        "--target_len_bucket_multiple",
        str(args.target_len_bucket_multiple),
        "--collect_profile",
        _bool_cli(args.collect_profile),
        "--reuse_static_input_embeds",
        _bool_cli(args.reuse_static_input_embeds),
        "--split_guidance_forward",
        _value_cli(args.split_guidance_forward),
        "--split_guidance_min_batch_size",
        str(args.split_guidance_min_batch_size),
        "--split_guidance_min_saved_context_ratio",
        str(args.split_guidance_min_saved_context_ratio),
        "--warmup_batches",
        str(args.warmup_batches),
        "--warmup_fill_batch",
        _bool_cli(args.warmup_fill_batch),
        "--max_request_text_chars",
        str(args.max_request_text_chars),
        "--log_level",
        args.server_log_level,
    ]
    _maybe_extend(command, "--device", args.device)
    _maybe_extend(command, "--warmup_test_list", args.warmup_test_list)
    _maybe_extend(command, "--batch_size_pad", args.batch_size_pad)
    return command


async def _wait_for_server(
    *,
    health_url: str,
    process: asyncio.subprocess.Process,
    timeout_s: float,
) -> dict[str, Any]:
    import httpx

    deadline = time.monotonic() + timeout_s
    last_error = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(
                    f"server exited before becoming healthy: {process.returncode}"
                )
            try:
                response = await client.get(health_url)
                if response.status_code == 200:
                    return response.json()
                last_error = f"status={response.status_code} body={response.text}"
            except Exception as exc:
                last_error = str(exc)
            await asyncio.sleep(0.5)
    raise TimeoutError(f"server did not become healthy: {last_error}")


async def _terminate_process(
    process: asyncio.subprocess.Process,
    *,
    timeout_s: float,
) -> None:
    if process.returncode is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def _http_sweep_args_for_server(
    args: argparse.Namespace,
    *,
    server_profile_dir: Path,
) -> argparse.Namespace:
    base_url = _base_url(args.host, args.port)
    return argparse.Namespace(
        url=f"{base_url}/v1/tts",
        scheduler_url=f"{base_url}/v1/scheduler",
        test_list=args.test_list,
        res_dir=str(server_profile_dir / "client_sweep"),
        concurrency_values=args.client_concurrency_values,
        arrival_gap_ms_values=args.client_arrival_gap_ms_values,
        repeats=args.client_repeats,
        warmup_repeats=args.client_warmup_repeats,
        request_repeats=args.client_request_repeats,
        timeout_s=args.client_timeout_s,
        save_wavs=args.save_wavs,
        reset_scheduler_metrics=True,
        pre_register_voices=args.client_pre_register_voices,
        voice_register_url=None,
    )


def _compact_server_result(row: dict[str, Any]) -> dict[str, Any]:
    best = row.get("best") or {}
    compact_best = best.get("compact")
    gpu = row.get("gpu") or {}
    return {
        "server_profile": row.get("server_profile"),
        "healthy": row.get("healthy"),
        "error": row.get("error"),
        "best": compact_best,
        "gpu": gpu.get("summary") if gpu.get("enabled") else gpu,
        "server_log": row.get("server_log"),
    }


async def _run_server_profile(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
) -> dict[str, Any]:
    server_profile_dir = Path(args.res_dir) / profile["name"]
    server_profile_dir.mkdir(parents=True, exist_ok=True)
    log_path = server_profile_dir / "server.log"
    command = _build_server_command(args, profile)

    result: dict[str, Any] = {
        "server_profile": profile,
        "server_command": command,
        "server_log": str(log_path),
        "healthy": None,
        "best": None,
        "summary": None,
        "gpu": {"enabled": bool(args.gpu_monitor)},
    }
    with log_path.open("wb") as log_file:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            result["healthy"] = await _wait_for_server(
                health_url=f"{_base_url(args.host, args.port)}/healthz",
                process=process,
                timeout_s=args.startup_timeout_s,
            )
            gpu_sampler = _GpuSampler(
                enabled=bool(args.gpu_monitor),
                gpu_index=args.gpu_index,
                interval_s=args.gpu_sample_interval_ms / 1000.0,
            )
            await gpu_sampler.start()
            try:
                sweep_summary = await run_http_sweep(
                    _http_sweep_args_for_server(
                        args,
                        server_profile_dir=server_profile_dir,
                    )
                )
            finally:
                result["gpu"] = await gpu_sampler.stop()
            result["client_sweep"] = sweep_summary
            ranked = rank_http_sweep_results(sweep_summary.get("results") or [])
            best = ranked[0] if ranked else None
            result["best"] = {
                "compact": sweep_summary.get("best"),
                "summary": (best or {}).get("summary"),
            }
            result["summary"] = result["best"]["summary"]
        except Exception as exc:
            result["error"] = str(exc) or exc.__class__.__name__
        finally:
            await _terminate_process(
                process,
                timeout_s=args.shutdown_timeout_s,
            )
            result["server_returncode"] = process.returncode
    return result


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    res_dir = Path(args.res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    profiles = build_server_sweep_profiles(
        batch_size_values=parse_int_grid(args.batch_size_values),
        max_wait_ms_values=parse_float_grid(args.max_wait_ms_values),
        max_cost_ratio_values=parse_float_grid(args.max_cost_ratio_values),
        max_total_target_tokens_values=parse_int_grid(
            args.max_total_target_tokens_values
        ),
        max_total_context_tokens_values=parse_int_grid(
            args.max_total_context_tokens_values
            if args.max_total_context_tokens_values is not None
            else str(args.max_total_context_tokens)
        ),
        max_context_ratio_values=parse_float_grid(
            args.max_context_ratio_values
            if args.max_context_ratio_values is not None
            else str(args.max_context_ratio)
        ),
        max_context_padding_ratio_values=parse_float_grid(
            args.max_context_padding_ratio_values
            if args.max_context_padding_ratio_values is not None
            else str(args.max_context_padding_ratio)
        ),
        lookahead_for_partial_batch_values=parse_bool_grid(
            args.lookahead_for_partial_batch_values
            if args.lookahead_for_partial_batch_values is not None
            else _bool_cli(args.lookahead_for_partial_batch)
        ),
        partial_lookahead_max_wait_multiplier_values=parse_float_grid(
            args.partial_lookahead_max_wait_multiplier_values
            if args.partial_lookahead_max_wait_multiplier_values is not None
            else str(args.partial_lookahead_max_wait_multiplier)
        ),
        num_step_values=parse_int_grid(args.num_step_values),
    )
    for profile in profiles:
        profile["candidate_pack_policy"] = args.candidate_pack_policy
        profile["name"] = f'{profile["name"]}_pack{args.candidate_pack_policy}'

    results = []
    for profile in profiles:
        results.append(await _run_server_profile(args=args, profile=profile))

    ranked = rank_server_sweep_results(results)
    best = summarize_best_server_sweep_result(results)
    recommendation = recommend_server_sweep_config(
        results,
        target_gpu_util_percent=args.target_gpu_util_percent,
        high_queue_wait_ms=args.high_queue_wait_ms,
        balanced_rtf_tolerance=args.balanced_rtf_tolerance,
        quality_min_num_step=args.quality_min_num_step,
    )
    summary = {
        "profiles": profiles,
        "best": _compact_server_result(best) if best else None,
        "recommendation": recommendation,
        "ranked": [_compact_server_result(row) for row in ranked],
        "results": results,
    }
    (res_dir / "server_sweep_summary.json").write_text(
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
                "recommendation": summary["recommendation"],
                "ranked": summary["ranked"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
