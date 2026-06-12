"""Shared helpers for OmniVoice benchmark CLIs."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional


def request_exception_result(
    *,
    request_id: str,
    exc: BaseException,
    request_wall_s: float,
    **extra: Any,
) -> dict[str, Any]:
    message = str(exc) or exc.__class__.__name__
    return {
        "id": request_id,
        "success": False,
        "request_wall_s": request_wall_s,
        "error": f"{exc.__class__.__name__}: {message}",
        **extra,
    }


def successful_request_results(
    results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [row for row in results if row.get("success") is True]


def summarize_request_results(
    results: Iterable[dict[str, Any]],
    *,
    batch_size_key: str,
    infer_s_key: str,
    queue_wait_ms_key: str = "queue_wait_ms",
    reason_key: Optional[str] = "batch_reason",
    batch_cost_tokens_key: Optional[str] = "batch_cost_tokens",
    batch_max_cost_tokens_key: Optional[str] = "batch_max_cost_tokens",
    batch_context_tokens_key: Optional[str] = "batch_context_tokens",
    batch_max_context_tokens_key: Optional[str] = "batch_max_context_tokens",
    batch_context_padding_ratio_key: Optional[str] = (
        "batch_context_padding_ratio"
    ),
) -> dict[str, Any]:
    rows = list(results)
    batch_sizes = _numeric_values(rows, batch_size_key, int)
    wait_ms = _numeric_values(rows, queue_wait_ms_key, float)
    infer_s = _numeric_values(rows, infer_s_key, float)
    batch_cost_tokens = _optional_numeric_values(rows, batch_cost_tokens_key, int)
    batch_max_cost_tokens = _optional_numeric_values(
        rows,
        batch_max_cost_tokens_key,
        int,
    )
    batch_context_tokens = _optional_numeric_values(
        rows,
        batch_context_tokens_key,
        int,
    )
    batch_max_context_tokens = _optional_numeric_values(
        rows,
        batch_max_context_tokens_key,
        int,
    )
    batch_context_padding_ratio = _optional_numeric_values(
        rows,
        batch_context_padding_ratio_key,
        float,
    )

    summary: dict[str, Any] = {
        "batch_size_histogram": _histogram(batch_sizes),
        "queue_wait_ms": _distribution(wait_ms),
        "request_infer_s": _distribution(infer_s),
    }
    if batch_cost_tokens_key is not None:
        summary["batch_cost_tokens"] = _distribution(batch_cost_tokens)
    if batch_max_cost_tokens_key is not None:
        summary["batch_max_cost_tokens"] = _distribution(batch_max_cost_tokens)
    if batch_context_tokens_key is not None:
        summary["batch_context_tokens"] = _distribution(batch_context_tokens)
    if batch_max_context_tokens_key is not None:
        summary["batch_max_context_tokens"] = _distribution(batch_max_context_tokens)
    if batch_context_padding_ratio_key is not None:
        summary["batch_context_padding_ratio"] = _distribution(
            batch_context_padding_ratio
        )
    if reason_key is not None:
        summary["batch_reason_histogram"] = _histogram(
            str(row[reason_key])
            for row in rows
            if reason_key in row and row[reason_key] is not None
        )
    return summary


def _numeric_values(rows: list[dict[str, Any]], key: str, caster) -> list[float]:
    values: list[float] = []
    for row in rows:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(caster(value))
        except (TypeError, ValueError):
            continue
    return values


def _optional_numeric_values(
    rows: list[dict[str, Any]],
    key: Optional[str],
    caster,
) -> list[float]:
    if key is None:
        return []
    return _numeric_values(rows, key, caster)


def _histogram(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts, key=str)}


def _distribution(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * quantile
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_int_grid(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("integer grid values must be positive")
    return values


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one number")
    if any(value < 0.0 for value in values):
        raise ValueError("float grid values must be non-negative")
    return values


def parse_bool_grid(raw: str) -> list[bool]:
    values = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"1", "true", "yes", "y", "on"}:
            values.append(True)
        elif item in {"0", "false", "no", "n", "off"}:
            values.append(False)
        else:
            raise ValueError(f"invalid boolean grid value: {part!r}")
    if not values:
        raise ValueError("grid must contain at least one boolean")
    return values


def build_http_sweep_profiles(
    *,
    concurrency_values: list[int],
    arrival_gap_ms_values: list[float],
    repeats: int,
    warmup_repeats: int = 0,
) -> list[dict[str, Any]]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup_repeats < 0:
        raise ValueError("warmup_repeats must be non-negative")

    profiles: list[dict[str, Any]] = []
    for concurrency in concurrency_values:
        for arrival_gap_ms in arrival_gap_ms_values:
            for warmup_index in range(warmup_repeats):
                profiles.append(
                    {
                        "name": _http_sweep_warmup_profile_name(
                            concurrency,
                            arrival_gap_ms,
                            warmup_index + 1,
                        ),
                        "concurrency": concurrency,
                        "arrival_gap_ms": arrival_gap_ms,
                        "repeat": 0,
                        "warmup_repeat": warmup_index + 1,
                        "is_warmup": True,
                    }
                )
            for repeat_index in range(repeats):
                profiles.append(
                    {
                        "name": _http_sweep_profile_name(
                            concurrency,
                            arrival_gap_ms,
                            repeat_index + 1,
                        ),
                        "concurrency": concurrency,
                        "arrival_gap_ms": arrival_gap_ms,
                        "repeat": repeat_index + 1,
                        "is_warmup": False,
                    }
                )
    return profiles


def build_server_sweep_profiles(
    *,
    batch_size_values: list[int],
    max_wait_ms_values: list[float],
    max_cost_ratio_values: list[float],
    max_total_target_tokens_values: list[int],
    max_total_context_tokens_values: list[int],
    max_context_ratio_values: list[float],
    max_context_padding_ratio_values: list[float],
    lookahead_for_partial_batch_values: list[bool],
    partial_lookahead_max_wait_multiplier_values: list[float],
    num_step_values: list[int],
) -> list[dict[str, Any]]:
    _require_grid("batch_size_values", batch_size_values)
    _require_grid("max_wait_ms_values", max_wait_ms_values)
    _require_grid("max_cost_ratio_values", max_cost_ratio_values)
    _require_grid("max_total_target_tokens_values", max_total_target_tokens_values)
    _require_grid("max_total_context_tokens_values", max_total_context_tokens_values)
    _require_grid("max_context_ratio_values", max_context_ratio_values)
    _require_grid(
        "max_context_padding_ratio_values",
        max_context_padding_ratio_values,
    )
    _require_grid(
        "lookahead_for_partial_batch_values",
        lookahead_for_partial_batch_values,
    )
    _require_grid(
        "partial_lookahead_max_wait_multiplier_values",
        partial_lookahead_max_wait_multiplier_values,
    )
    _require_grid("num_step_values", num_step_values)

    profiles: list[dict[str, Any]] = []
    for batch_size in batch_size_values:
        for max_wait_ms in max_wait_ms_values:
            for max_cost_ratio in max_cost_ratio_values:
                for max_total_target_tokens in max_total_target_tokens_values:
                    for max_total_context_tokens in max_total_context_tokens_values:
                        for max_context_ratio in max_context_ratio_values:
                            for max_context_padding_ratio in (
                                max_context_padding_ratio_values
                            ):
                                for lookahead_for_partial_batch in (
                                    lookahead_for_partial_batch_values
                                ):
                                    for partial_lookahead_multiplier in (
                                        partial_lookahead_max_wait_multiplier_values
                                    ):
                                        for num_step in num_step_values:
                                            profiles.append(
                                                {
                                                    "name": (
                                                        _server_sweep_profile_name(
                                                            batch_size=batch_size,
                                                            max_wait_ms=max_wait_ms,
                                                            max_cost_ratio=(
                                                                max_cost_ratio
                                                            ),
                                                            max_total_target_tokens=(
                                                                max_total_target_tokens
                                                            ),
                                                            max_total_context_tokens=(
                                                                max_total_context_tokens
                                                            ),
                                                            max_context_ratio=(
                                                                max_context_ratio
                                                            ),
                                                            max_context_padding_ratio=(
                                                                max_context_padding_ratio
                                                            ),
                                                            lookahead_for_partial_batch=(
                                                                lookahead_for_partial_batch
                                                            ),
                                                            partial_lookahead_max_wait_multiplier=(
                                                                partial_lookahead_multiplier
                                                            ),
                                                            num_step=num_step,
                                                        )
                                                    ),
                                                    "batch_size": batch_size,
                                                    "max_wait_ms": max_wait_ms,
                                                    "max_cost_ratio": max_cost_ratio,
                                                    "max_total_target_tokens": (
                                                        max_total_target_tokens
                                                    ),
                                                    "max_total_context_tokens": (
                                                        max_total_context_tokens
                                                    ),
                                                    "max_context_ratio": (
                                                        max_context_ratio
                                                    ),
                                                    "max_context_padding_ratio": (
                                                        max_context_padding_ratio
                                                    ),
                                                    "lookahead_for_partial_batch": (
                                                        lookahead_for_partial_batch
                                                    ),
                                                    "partial_lookahead_max_wait_multiplier": (
                                                        partial_lookahead_multiplier
                                                    ),
                                                    "num_step": num_step,
                                                }
                                            )
    return profiles


def rank_http_sweep_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in results if not _is_warmup_result(row)]
    return sorted(rows, key=_http_sweep_sort_key)


def summarize_best_http_sweep_result(
    results: Iterable[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    ranked = rank_http_sweep_results(results)
    return ranked[0] if ranked else None


def rank_server_sweep_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(list(results), key=_server_sweep_sort_key)


def summarize_best_server_sweep_result(
    results: Iterable[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    ranked = rank_server_sweep_results(results)
    return ranked[0] if ranked else None


def diagnose_server_sweep_result(
    row: dict[str, Any],
    *,
    target_gpu_util_percent: float = 70.0,
    high_queue_wait_ms: float = 100.0,
    quality_min_num_step: int = 32,
) -> dict[str, Any]:
    """Explain why one server sweep result is or is not near optimal."""

    server_profile = row.get("server_profile") or row.get("profile") or {}
    best = row.get("best") or {}
    compact = dict(best.get("compact") or row.get("best") or row.get("summary") or {})
    summary = best.get("summary") or row.get("summary") or {}
    if summary and "scheduler_after" in summary:
        for key, value in _compact_from_summary(summary).items():
            if compact.get(key) is None and value is not None:
                compact[key] = value

    num_requests = int(compact.get("num_requests") or summary.get("num_requests") or 0)
    num_successful = int(
        compact.get("num_successful") or summary.get("num_successful") or 0
    )
    success_rate = num_successful / num_requests if num_requests > 0 else 0.0
    gpu_summary = (row.get("gpu") or {}).get("summary") or row.get("gpu") or {}
    gpu_util_max = _nested_float(gpu_summary, "gpu_util_percent", "max")
    queue_wait_p95 = _nested_float(compact, "queue_wait_ms", "p95")
    context_padding_p95 = _nested_float(
        compact,
        "batch_context_padding_ratio",
        "p95",
    )
    avg_batch_size = _none_to_zero(compact.get("avg_batch_size"))
    max_batch_size = int(server_profile.get("batch_size") or 0)
    underfilled = max_batch_size > 0 and avg_batch_size < max_batch_size * 0.75
    reason_histogram = compact.get("batch_reason_histogram") or {}
    full_batch_share = _histogram_share(reason_histogram, "full")
    gpu_saturated = (
        gpu_util_max is not None and gpu_util_max >= target_gpu_util_percent
    )
    serial_full_batch_backlog = (
        gpu_saturated
        and full_batch_share is not None
        and full_batch_share >= 0.5
    )
    split_retry_batches = int(compact.get("split_retry_batches") or 0)
    adaptive_caps = compact.get("adaptive_batch_caps") or {}
    num_step = int(server_profile.get("num_step") or 0)

    issues: list[dict[str, Any]] = []
    if row.get("error"):
        issues.append(
            {
                "code": "server_error",
                "severity": "error",
                "message": row["error"],
                "action": "Inspect server_log and exclude this profile from recommendations.",
            }
        )
    if num_requests > 0 and success_rate < 1.0:
        issues.append(
            {
                "code": "request_failures",
                "severity": "error",
                "message": f"{num_successful}/{num_requests} requests succeeded.",
                "action": "Lower batch/cost limits or inspect failed request payloads.",
            }
        )
    if split_retry_batches > 0 or adaptive_caps:
        issues.append(
            {
                "code": "memory_pressure",
                "severity": "warning",
                "message": (
                    "The profile required split retry or learned an adaptive "
                    f"batch cap: {adaptive_caps or '{}'}."
                ),
                "action": (
                    "Treat it as near the memory boundary; prefer the learned cap "
                    "or reduce batch_size/max_total_target_tokens."
                ),
            }
        )
    if quality_min_num_step > 0 and num_step > 0 and num_step < quality_min_num_step:
        issues.append(
            {
                "code": "quality_probe_num_step",
                "severity": "warning",
                "message": (
                    f"num_step={num_step} is below the production quality floor "
                    f"{quality_min_num_step}."
                ),
                "action": (
                    "Treat this result as a scheduler stress probe only; rerun "
                    "with production num_step before deployment."
                ),
            }
        )
    if queue_wait_p95 is not None and queue_wait_p95 > high_queue_wait_ms:
        if serial_full_batch_backlog:
            queue_code = "queue_latency_serial_full_batches"
            queue_action = (
                "GPU is already saturated and most requests ran in full batches; "
                "this p95 is queueing behind earlier model calls, not waiting "
                "for batch formation. Use a lower-latency profile for interactive "
                "traffic, split latency/throughput pools, add serving replicas, "
                "or optimize model-kernel time. Raising max_wait_ms will not help."
            )
        elif underfilled:
            queue_code = "queue_latency_high_underfilled"
            queue_action = (
                "This profile is waiting behind partial batches. Increase "
                "max_wait_ms or relax max_cost_ratio/max_context_ratio/"
                "max_total_context_tokens/max_context_padding_ratio if "
                "full-batch throughput matters, otherwise lower batch_size."
            )
        else:
            queue_code = "queue_latency_high"
            queue_action = (
                "Lower max_wait_ms or add serving capacity if latency matters "
                "more than throughput."
            )
        issues.append(
            {
                "code": queue_code,
                "severity": "warning",
                "message": f"Queue p95 is {queue_wait_p95:.3f} ms.",
                "action": queue_action,
            }
        )
    if context_padding_p95 is not None and context_padding_p95 > 2.5:
        issues.append(
            {
                "code": "context_padding_waste",
                "severity": "warning",
                "message": (
                    "Batch context padding p95 is "
                    f"{context_padding_p95:.3f}x."
                ),
                "action": (
                    "Lower max_context_ratio or set max_context_padding_ratio "
                    "near 2.0 so long prompt-context requests do not force "
                    "short requests through oversized LLM padding."
                ),
            }
        )
    if gpu_util_max is not None and gpu_util_max < target_gpu_util_percent:
        if underfilled:
            code = "underfilled_batches"
            action = (
                "Increase client concurrency, reduce arrival gaps, or release "
                "more requests concurrently from the upstream workload."
            )
        else:
            code = "gpu_underutilized"
            action = (
                "Increase batch_size/max_total_target_tokens or num_step workload "
                "until GPU util improves or memory split retries appear."
            )
        issues.append(
            {
                "code": code,
                "severity": "info",
                "message": (
                    f"GPU util max is {gpu_util_max:.1f}% below target "
                    f"{target_gpu_util_percent:.1f}%."
                ),
                "action": action,
            }
        )

    if not issues:
        issues.append(
            {
                "code": "balanced",
                "severity": "info",
                "message": "No obvious bottleneck detected from the collected metrics.",
                "action": "Use this profile as the baseline and test adjacent larger batches.",
            }
        )

    return {
        "server_profile": server_profile,
        "client_profile": compact.get("profile"),
        "success_rate": success_rate,
        "rtf_wall": compact.get("rtf_wall"),
        "avg_batch_size": avg_batch_size,
        "gpu_util_max": gpu_util_max,
        "queue_wait_p95_ms": queue_wait_p95,
        "context_padding_p95": context_padding_p95,
        "full_batch_share": full_batch_share,
        "issues": issues,
    }


def recommend_server_sweep_config(
    results: Iterable[dict[str, Any]],
    *,
    target_gpu_util_percent: float = 70.0,
    high_queue_wait_ms: float = 100.0,
    balanced_rtf_tolerance: float = 0.05,
    quality_min_num_step: int = 32,
) -> dict[str, Any]:
    rows = list(results)
    ranked = rank_server_sweep_results(rows)
    best = ranked[0] if ranked else None
    if best is None:
        return {
            "best": None,
            "throughput_best": None,
            "balanced_best": None,
            "ranked_diagnostics": [],
            "diagnostics": [],
            "next_actions": ["Run at least one successful server sweep profile."],
        }

    throughput_best = diagnose_server_sweep_result(
        best,
        target_gpu_util_percent=target_gpu_util_percent,
        high_queue_wait_ms=high_queue_wait_ms,
        quality_min_num_step=quality_min_num_step,
    )
    ranked_diagnostics = [
        diagnose_server_sweep_result(
            row,
            target_gpu_util_percent=target_gpu_util_percent,
            high_queue_wait_ms=high_queue_wait_ms,
            quality_min_num_step=quality_min_num_step,
        )
        for row in ranked
    ]
    rtf_limit = _balanced_rtf_limit(
        throughput_best.get("rtf_wall"),
        balanced_rtf_tolerance=balanced_rtf_tolerance,
    )
    balanced_best = next(
        (
            diagnostic
            for diagnostic in ranked_diagnostics
            if _is_balanced_sweep_diagnostic(diagnostic, rtf_limit=rtf_limit)
        ),
        None,
    )
    next_actions = _dedupe_preserve_order(
        issue["action"] for issue in throughput_best["issues"]
    )
    if balanced_best is not None and balanced_best != throughput_best:
        next_actions = _dedupe_preserve_order(
            [
                (
                    "Use balanced_best for production traffic when its RTF is "
                    "within tolerance; keep throughput_best for offline saturation tests."
                ),
                *next_actions,
            ]
        )
    return {
        "best": throughput_best,
        "throughput_best": throughput_best,
        "balanced_best": balanced_best,
        "ranked_diagnostics": ranked_diagnostics,
        "next_actions": next_actions,
    }


def _balanced_rtf_limit(
    rtf_wall: Any,
    *,
    balanced_rtf_tolerance: float,
) -> Optional[float]:
    try:
        base = float(rtf_wall)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(base):
        return None
    tolerance = max(0.0, float(balanced_rtf_tolerance))
    return base * (1.0 + tolerance)


def _is_balanced_sweep_diagnostic(
    diagnostic: dict[str, Any],
    *,
    rtf_limit: Optional[float],
) -> bool:
    if diagnostic.get("success_rate") != 1.0:
        return False
    if any(
        issue.get("severity") in {"error", "warning"}
        for issue in diagnostic.get("issues") or []
    ):
        return False
    if rtf_limit is None:
        return True
    try:
        rtf_wall = float(diagnostic.get("rtf_wall"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(rtf_wall) and rtf_wall <= rtf_limit


def _http_sweep_profile_name(
    concurrency: int,
    arrival_gap_ms: float,
    repeat: int,
) -> str:
    gap = ("%g" % arrival_gap_ms).replace(".", "p")
    return f"c{concurrency}_gap{gap}ms_r{repeat}"


def _http_sweep_warmup_profile_name(
    concurrency: int,
    arrival_gap_ms: float,
    warmup_repeat: int,
) -> str:
    gap = ("%g" % arrival_gap_ms).replace(".", "p")
    return f"c{concurrency}_gap{gap}ms_warmup{warmup_repeat}"


def _is_warmup_result(row: dict[str, Any]) -> bool:
    return bool((row.get("profile") or {}).get("is_warmup"))


def _server_sweep_profile_name(
    *,
    batch_size: int,
    max_wait_ms: float,
    max_cost_ratio: float,
    max_total_target_tokens: int,
    max_total_context_tokens: int,
    max_context_ratio: float,
    max_context_padding_ratio: float,
    lookahead_for_partial_batch: bool,
    partial_lookahead_max_wait_multiplier: float,
    num_step: int,
) -> str:
    wait = ("%g" % max_wait_ms).replace(".", "p")
    ratio = ("%g" % max_cost_ratio).replace(".", "p")
    context_ratio = ("%g" % max_context_ratio).replace(".", "p")
    padding_ratio = ("%g" % max_context_padding_ratio).replace(".", "p")
    partial = "pl1" if lookahead_for_partial_batch else "pl0"
    partial_wait = ("%g" % partial_lookahead_max_wait_multiplier).replace(".", "p")
    return (
        f"b{batch_size}_wait{wait}ms_ratio{ratio}_"
        f"tok{max_total_target_tokens}_ctx{max_total_context_tokens}_"
        f"ctxr{context_ratio}_pad{padding_ratio}_{partial}x{partial_wait}_"
        f"step{num_step}"
    )


def _http_sweep_sort_key(row: dict[str, Any]):
    summary = row.get("summary", row)
    num_requests = int(summary.get("num_requests") or 0)
    num_successful = int(summary.get("num_successful") or 0)
    success_rate = num_successful / num_requests if num_requests > 0 else 0.0
    metrics = summary.get("request_metrics") or {}
    queue_wait = metrics.get("queue_wait_ms") or {}
    queue_p95 = _none_to_inf(queue_wait.get("p95"))
    avg_batch_size = _none_to_zero(
        (summary.get("scheduler_after") or {}).get("avg_batch_size")
    )
    rtf_wall = _none_to_inf(summary.get("rtf_wall"))
    wall_s = _none_to_inf(summary.get("wall_s"))
    return (
        -success_rate,
        rtf_wall,
        queue_p95,
        -avg_batch_size,
        wall_s,
        str(row.get("profile", {}).get("name") or row.get("name") or ""),
    )


def _server_sweep_sort_key(row: dict[str, Any]):
    best = row.get("best") or {}
    summary = (
        best.get("summary")
        or row.get("summary")
        or _summary_from_compact(best.get("compact") or row.get("compact") or {})
    )
    proxy = {
        "profile": row.get("server_profile") or row.get("profile") or {},
        "summary": summary,
    }
    return _http_sweep_sort_key(proxy)


def _summary_from_compact(compact: dict[str, Any]) -> dict[str, Any]:
    if not compact:
        return {}
    return {
        "profile": compact.get("profile"),
        "num_successful": compact.get("num_successful"),
        "num_requests": compact.get("num_requests"),
        "rtf_wall": compact.get("rtf_wall"),
        "wall_s": compact.get("wall_s"),
        "scheduler_after": {
            "avg_batch_size": compact.get("avg_batch_size"),
        },
        "request_metrics": {
            "queue_wait_ms": compact.get("queue_wait_ms"),
            "batch_reason_histogram": compact.get("batch_reason_histogram"),
        },
    }


def _compact_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    scheduler_after = summary.get("scheduler_after") or {}
    request_metrics = summary.get("request_metrics") or {}
    return {
        "profile": summary.get("profile"),
        "num_successful": summary.get("num_successful"),
        "num_requests": summary.get("num_requests"),
        "rtf_wall": summary.get("rtf_wall"),
        "avg_batch_size": scheduler_after.get("avg_batch_size"),
        "split_retry_batches": scheduler_after.get("split_retry_batches"),
        "adaptive_batch_caps": scheduler_after.get("adaptive_batch_caps"),
        "queue_wait_ms": request_metrics.get("queue_wait_ms"),
        "batch_reason_histogram": request_metrics.get("batch_reason_histogram"),
    }


def _nested_float(row: dict[str, Any], key: str, subkey: str) -> Optional[float]:
    nested = row.get(key) or {}
    value = nested.get(subkey)
    if value is None:
        return None
    return float(value)


def _histogram_share(
    histogram: dict[str, Any],
    key: str,
) -> Optional[float]:
    if not histogram:
        return None
    total = 0
    selected = 0
    for raw_key, raw_value in histogram.items():
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        total += value
        if str(raw_key) == key:
            selected += value
    if total <= 0:
        return None
    return selected / total


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _require_grid(name: str, values: list[Any]) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value")


def _none_to_inf(value: Any) -> float:
    if value is None:
        return math.inf
    return float(value)


def _none_to_zero(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)
