"""Named serving profiles derived from real online-batch benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import soundfile as sf


@dataclass(frozen=True)
class ServingProfile:
    """A documented online-serving configuration preset."""

    name: str
    description: str
    scheduler: dict[str, Any]


@dataclass(frozen=True)
class ServingProfileRecommendation:
    """Workload-derived serving profile recommendation."""

    profile: ServingProfile
    reasons: tuple[str, ...]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_profile": self.profile.name,
            "description": self.profile.description,
            "reasons": list(self.reasons),
            "stats": self.stats,
            "scheduler": self.profile.scheduler,
            "runtime": recommended_runtime_config(),
        }


SERVING_PROFILES: dict[str, ServingProfile] = {
    "custom": ServingProfile(
        name="custom",
        description="Use only explicit CLI values and code defaults.",
        scheduler={},
    ),
    "balanced12": ServingProfile(
        name="balanced12",
        description=(
            "Default production profile for low/medium concurrency. This was "
            "the fastest measured num_step=32 b12 profile on the local benchmark "
            "and matched the best 52-request context-outlier sweep shape."
        ),
        scheduler={
            "max_batch_size": 12,
            "max_wait_ms": 20.0,
            "partial_batch_floor": 2,
            "max_total_target_tokens": 2048,
            "max_total_context_tokens": 4096,
            "max_cost_ratio": 1.4,
            "max_context_ratio": 2.0,
            "max_context_padding_ratio": 2.0,
            "ready_queue_capacity": 128,
            "control_queue_capacity": 16,
            "prompt_cache_entries": 256,
            "use_model_duration_estimator": True,
            "lookahead_for_full_batch": True,
            "lookahead_for_partial_batch": True,
            "partial_lookahead_max_wait_multiplier": 2.0,
            "max_seed_lookahead": 64,
            "candidate_pack_policy": "target_context",
            "split_retry_on_memory_error": True,
            "adaptive_memory_batch_cap": True,
            "adaptive_memory_cap_recovery_successes": 64,
            "max_generation_batches_before_control": 8,
        },
    ),
    "context12": ServingProfile(
        name="context12",
        description=(
            "Measured throughput profile for request windows that mix short "
            "prompts with long prompt/ref_text outliers. Local RTX 3060 12GB "
            "52-request context-outlier sweep favored b12 (RTF 0.584, GPU p50 "
            "99%) over b8, b10, and b16 while preserving num_step=32."
        ),
        scheduler={
            "max_batch_size": 12,
            "max_wait_ms": 20.0,
            "partial_batch_floor": 2,
            "max_total_target_tokens": 2048,
            "max_total_context_tokens": 4096,
            "max_cost_ratio": 1.4,
            "max_context_ratio": 2.0,
            "max_context_padding_ratio": 2.0,
            "ready_queue_capacity": 128,
            "control_queue_capacity": 16,
            "prompt_cache_entries": 256,
            "use_model_duration_estimator": True,
            "lookahead_for_full_batch": True,
            "lookahead_for_partial_batch": True,
            "partial_lookahead_max_wait_multiplier": 2.0,
            "max_seed_lookahead": 64,
            "candidate_pack_policy": "target_context",
            "split_retry_on_memory_error": True,
            "adaptive_memory_batch_cap": True,
            "adaptive_memory_cap_recovery_successes": 64,
            "max_generation_batches_before_control": 8,
        },
    ),
    "burst24": ServingProfile(
        name="burst24",
        description=(
            "High-burst profile for about 24 same-window requests. It trades a "
            "small RTF regression for one larger batch and low queue p95."
        ),
        scheduler={
            "max_batch_size": 24,
            "max_wait_ms": 20.0,
            "partial_batch_floor": 2,
            "max_total_target_tokens": 4096,
            "max_total_context_tokens": 8192,
            "max_cost_ratio": 2.0,
            "max_context_ratio": 2.0,
            "max_context_padding_ratio": 2.0,
            "ready_queue_capacity": 192,
            "control_queue_capacity": 16,
            "prompt_cache_entries": 256,
            "use_model_duration_estimator": True,
            "lookahead_for_full_batch": True,
            "lookahead_for_partial_batch": True,
            "partial_lookahead_max_wait_multiplier": 2.0,
            "max_seed_lookahead": 96,
            "candidate_pack_policy": "target_context",
            "split_retry_on_memory_error": True,
            "adaptive_memory_batch_cap": True,
            "adaptive_memory_cap_recovery_successes": 64,
            "max_generation_batches_before_control": 8,
        },
    ),
    "context8": ServingProfile(
        name="context8",
        description=(
            "Context-heterogeneous A/B profile for smaller partial batches. "
            "It beat b16 in the coarse 52-request sweep, but the adjacent "
            "b4/b6/b10/b12 sweep found context12 faster on the same workload."
        ),
        scheduler={
            "max_batch_size": 8,
            "max_wait_ms": 20.0,
            "partial_batch_floor": 2,
            "max_total_target_tokens": 2048,
            "max_total_context_tokens": 4096,
            "max_cost_ratio": 1.4,
            "max_context_ratio": 2.0,
            "max_context_padding_ratio": 2.0,
            "ready_queue_capacity": 128,
            "control_queue_capacity": 16,
            "prompt_cache_entries": 256,
            "use_model_duration_estimator": True,
            "lookahead_for_full_batch": True,
            "lookahead_for_partial_batch": True,
            "partial_lookahead_max_wait_multiplier": 2.0,
            "max_seed_lookahead": 64,
            "candidate_pack_policy": "target_context",
            "split_retry_on_memory_error": True,
            "adaptive_memory_batch_cap": True,
            "adaptive_memory_cap_recovery_successes": 64,
            "max_generation_batches_before_control": 8,
        },
    ),
    "context16": ServingProfile(
        name="context16",
        description=(
            "Context-heterogeneous A/B profile for larger partial batches. "
            "It can reduce model-call count when traffic naturally fills b16, "
            "but the local 52-request context-outlier sweep was slower than "
            "context12 because context caps left b16 underfilled."
        ),
        scheduler={
            "max_batch_size": 16,
            "max_wait_ms": 20.0,
            "partial_batch_floor": 2,
            "max_total_target_tokens": 2048,
            "max_total_context_tokens": 4096,
            "max_cost_ratio": 1.4,
            "max_context_ratio": 2.0,
            "max_context_padding_ratio": 2.0,
            "ready_queue_capacity": 192,
            "control_queue_capacity": 16,
            "prompt_cache_entries": 256,
            "use_model_duration_estimator": True,
            "lookahead_for_full_batch": True,
            "lookahead_for_partial_batch": True,
            "partial_lookahead_max_wait_multiplier": 2.0,
            "max_seed_lookahead": 96,
            "candidate_pack_policy": "target_context",
            "split_retry_on_memory_error": True,
            "adaptive_memory_batch_cap": True,
            "adaptive_memory_cap_recovery_successes": 64,
            "max_generation_batches_before_control": 8,
        },
    ),
    "wide32": ServingProfile(
        name="wide32",
        description=(
            "Wide-batch profile for sustained 32-request bursts. Use only when "
            "the traffic pattern can fill b32 and the extra memory is acceptable."
        ),
        scheduler={
            "max_batch_size": 32,
            "max_wait_ms": 100.0,
            "partial_batch_floor": 2,
            "max_total_target_tokens": 6144,
            "max_total_context_tokens": 12288,
            "max_cost_ratio": 2.0,
            "max_context_ratio": 2.0,
            "max_context_padding_ratio": 2.0,
            "ready_queue_capacity": 256,
            "control_queue_capacity": 16,
            "prompt_cache_entries": 256,
            "use_model_duration_estimator": True,
            "lookahead_for_full_batch": True,
            "lookahead_for_partial_batch": True,
            "partial_lookahead_max_wait_multiplier": 2.0,
            "max_seed_lookahead": 128,
            "candidate_pack_policy": "target_context",
            "split_retry_on_memory_error": True,
            "adaptive_memory_batch_cap": True,
            "adaptive_memory_cap_recovery_successes": 64,
            "max_generation_batches_before_control": 8,
        },
    ),
    "wide48": ServingProfile(
        name="wide48",
        description=(
            "Optional high-burst profile for roughly 48 same-window clone or "
            "auto requests. Local RTX 3060 12GB b48 benchmarks matched wide32 "
            "RTF while reducing large-burst queueing, but used about 9GB VRAM."
        ),
        scheduler={
            "max_batch_size": 48,
            "max_wait_ms": 100.0,
            "partial_batch_floor": 2,
            "max_total_target_tokens": 8192,
            "max_total_context_tokens": 18432,
            "max_cost_ratio": 2.0,
            "max_context_ratio": 2.0,
            "max_context_padding_ratio": 2.0,
            "ready_queue_capacity": 384,
            "control_queue_capacity": 16,
            "prompt_cache_entries": 256,
            "use_model_duration_estimator": True,
            "lookahead_for_full_batch": True,
            "lookahead_for_partial_batch": True,
            "partial_lookahead_max_wait_multiplier": 2.0,
            "max_seed_lookahead": 192,
            "candidate_pack_policy": "target_context",
            "split_retry_on_memory_error": True,
            "adaptive_memory_batch_cap": True,
            "adaptive_memory_cap_recovery_successes": 64,
            "max_generation_batches_before_control": 8,
        },
    ),
}


def get_serving_profile(name: str) -> ServingProfile:
    try:
        return SERVING_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(SERVING_PROFILES))
        raise ValueError(f"unknown serving profile {name!r}; choose one of: {choices}") from exc


def recommended_runtime_config() -> dict[str, Any]:
    """Return the measured runtime defaults for online GPU serving.

    These values are intentionally separate from scheduler profiles: the
    profile controls how requests are grouped, while this runtime block
    controls the model execution path shared by all recommended profiles.
    """

    return {
        "compile_llm": True,
        "compile_mode": "default",
        "compile_audio_heads": False,
        "allow_reduce_overhead_worker": False,
        "matmul_precision": "high",
        "num_step": 32,
        "batched_decode": True,
        "reuse_static_input_embeds": True,
        "split_guidance_forward": "auto",
        "split_guidance_min_batch_size": 8,
        "split_guidance_min_saved_context_ratio": 0.25,
        "warmup_batches": 2,
        "warmup_fill_batch": True,
        "startup_warmup_required": True,
        "notes": [
            "Run startup warmup with a representative JSONL when compile_llm is enabled; otherwise the first user batch pays torch.compile graph build time.",
            "Recommended startup warmup repeats compatible samples to fill the selected scheduler profile batch size, so the largest expected compile graph is built before live traffic.",
            "Auto split-guidance is enabled from batch size 8 when unconditional context savings are at least 25%; this preserves num_step while avoiding padded unconditional forward work.",
            "torch.compile is not bit-exact; use compile_llm=false only when exact reproducibility is more important than throughput.",
            "Do not reduce num_step for throughput unless you explicitly accept a quality change.",
        ],
    }


def recommend_serving_profile(
    samples: list[dict[str, Any]],
    *,
    concurrency: int,
    frame_rate: int = 25,
    latency_sensitive: bool = False,
) -> ServingProfileRecommendation:
    """Recommend a serving profile for a representative request window.

    The rules intentionally encode only benchmark-backed boundaries from the
    local online-batching sweeps:
    - low/medium concurrency defaults to ``balanced12``;
    - context-heterogeneous bursts use ``context12`` because it was the fastest
      measured 52-request context-outlier shape;
    - homogeneous 24-ish bursts use ``burst24``;
    - sustained homogeneous 32-request bursts use ``wide32``;
    - optional homogeneous 48-request bursts can use ``wide48``.
    """

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    stats = summarize_workload(samples, concurrency=concurrency, frame_rate=frame_rate)
    reasons: list[str] = []

    profile_name = "balanced12"
    if stats["num_requests"] == 0:
        reasons.append("No samples were provided; using balanced12 defaults.")
    elif latency_sensitive and concurrency <= 16:
        reasons.append(
            "Latency-sensitive low/medium concurrency traffic favors balanced12."
        )
    elif stats["context_heterogeneous"]:
        profile_name = "context12"
        reasons.append(
            "Context p95/p50 ratio indicates long prompt/ref_text outliers; "
            "context12 was fastest on the local 52-request context-outlier benchmark."
        )
    elif concurrency >= 48 and stats["num_requests"] >= 48:
        profile_name = "wide48"
        reasons.append(
            "Sustained homogeneous 48-request bursts can fill the optional "
            "wide48 profile; use it when the extra VRAM is acceptable."
        )
    elif concurrency >= 32 and stats["num_requests"] >= 32:
        profile_name = "wide32"
        reasons.append(
            "Sustained homogeneous 32-request bursts can fill the wide32 profile."
        )
    elif concurrency >= 20 or stats["num_requests"] >= 24:
        profile_name = "burst24"
        reasons.append(
            "Homogeneous high-burst traffic can fill the burst24 profile."
        )
    else:
        reasons.append("Low/medium concurrency traffic uses balanced12.")

    if stats["clone_request_share"] > 0:
        reasons.append(
            "Clone requests are present; keep prompt_cache_entries enabled for "
            "cross-request speaker reuse."
        )
    if stats["target_p95_to_p50_ratio"] > 2.0:
        reasons.append(
            "Target-duration spread is high; keep max_cost_ratio conservative."
        )

    return ServingProfileRecommendation(
        profile=get_serving_profile(profile_name),
        reasons=tuple(reasons),
        stats=stats,
    )


def summarize_workload(
    samples: list[dict[str, Any]],
    *,
    concurrency: int,
    frame_rate: int = 25,
) -> dict[str, Any]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    targets = [_estimate_target_tokens(sample, frame_rate=frame_rate) for sample in samples]
    ref_audio_token_cache: dict[tuple[Any, ...], int] = {}
    contexts = [
        _estimate_context_tokens(
            sample,
            cost_tokens=target,
            frame_rate=frame_rate,
            ref_audio_token_cache=ref_audio_token_cache,
        )
        for sample, target in zip(samples, targets)
    ]
    modes = [_mode_for_sample(sample) for sample in samples]
    clone_count = sum(1 for mode in modes if mode == "clone")
    context_p50 = _percentile(contexts, 0.50)
    context_p95 = _percentile(contexts, 0.95)
    target_p50 = _percentile(targets, 0.50)
    target_p95 = _percentile(targets, 0.95)
    context_ratio = _safe_ratio(context_p95, context_p50)
    target_ratio = _safe_ratio(target_p95, target_p50)
    outlier_threshold = (context_p50 or 0.0) * 2.5
    context_outliers = (
        sum(1 for value in contexts if outlier_threshold > 0 and value >= outlier_threshold)
        if contexts
        else 0
    )
    context_outlier_share = context_outliers / len(contexts) if contexts else 0.0

    return {
        "num_requests": len(samples),
        "concurrency": concurrency,
        "target_tokens": _distribution(targets),
        "context_tokens": _distribution(contexts),
        "target_p95_to_p50_ratio": target_ratio,
        "context_p95_to_p50_ratio": context_ratio,
        "context_outlier_share": context_outlier_share,
        "context_heterogeneous": context_ratio >= 2.5
        or context_outlier_share >= 0.05,
        "clone_request_share": clone_count / len(samples) if samples else 0.0,
        "mode_counts": _counts(modes),
    }


def _mode_for_sample(sample: dict[str, Any]) -> str:
    if (
        sample.get("voice_id") is not None
        or sample.get("ref_audio") is not None
        or sample.get("ref_audio_base64") is not None
    ):
        return "clone"
    if sample.get("instruct") is not None:
        return "design"
    return "auto"


def _estimate_target_tokens(sample: dict[str, Any], *, frame_rate: int) -> int:
    duration = sample.get("duration")
    if duration is not None:
        return max(1, int(float(duration) * frame_rate))
    speed = sample.get("speed") or 1.0
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 1.0
    if speed <= 0:
        speed = 1.0
    text = sample.get("text") or ""
    return max(16, int(max(1, len(text)) * 3.0 / speed))


def _estimate_context_tokens(
    sample: dict[str, Any],
    *,
    cost_tokens: int,
    frame_rate: int,
    ref_audio_token_cache: dict[tuple[Any, ...], int] | None = None,
) -> int:
    text_tokens = max(1, len(sample.get("text") or "") // 2)
    ref_text_tokens = max(0, len(sample.get("ref_text") or "") // 2)
    ref_audio_tokens = _estimate_ref_audio_tokens(
        sample.get("ref_audio"),
        frame_rate=frame_rate,
        cache=ref_audio_token_cache,
    )
    return max(
        1,
        int(cost_tokens + text_tokens + ref_text_tokens + ref_audio_tokens + 16),
    )


def _estimate_ref_audio_tokens(
    ref_audio: Any,
    *,
    frame_rate: int = 25,
    cache: dict[tuple[Any, ...], int] | None = None,
) -> int:
    if not isinstance(ref_audio, str):
        return 0
    key = _ref_audio_file_cache_key(ref_audio, frame_rate=frame_rate)
    if key is None:
        return 0
    if cache is not None and key in cache:
        return cache[key]
    try:
        info = sf.info(ref_audio)
        sample_rate = int(info.samplerate)
        frames = int(info.frames)
    except Exception:
        return 0
    if sample_rate <= 0:
        return 0
    value = max(1, int(frames / sample_rate * frame_rate))
    if cache is not None:
        cache[key] = value
    return value


def _ref_audio_file_cache_key(
    ref_audio: str,
    *,
    frame_rate: int,
) -> tuple[Any, ...] | None:
    path = os.path.abspath(ref_audio)
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return ("path", path, stat.st_size, stat.st_mtime_ns, frame_rate)


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * quantile
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator <= 0:
        return 0.0
    return numerator / denominator


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
