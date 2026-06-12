"""Online micro-batching scheduler for OmniVoice.

The upstream ``infer_batch`` CLI proves that ``OmniVoice.generate`` can run a
real list batch, but it is an offline JSONL runner. This module turns the same
model capability into an online serving primitive: many callers submit single
requests, a resident scheduler groups compatible requests for one model
instance, then executes one ``generate(list)`` call per micro-batch.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional, Sequence

import numpy as np
import soundfile as sf

from omnivoice.models.generation import (
    ensure_bool,
    ensure_min_float,
    ensure_non_negative_float,
    ensure_non_negative_int,
    ensure_positive_int,
    ensure_positive_float,
    resolve_optional_bool_flags,
)
from omnivoice.models.omnivoice import VoiceClonePrompt, _ref_audio_tuple_cache_marker


def _ensure_non_empty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _ensure_optional_str(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")
    return value


@dataclass(frozen=True)
class BatchSchedulerConfig:
    """Latency/throughput policy for a single resident OmniVoice instance."""

    max_batch_size: int = 8
    max_wait_ms: float = 120.0
    partial_batch_floor: int = 2
    max_total_target_tokens: int = 4096
    max_total_context_tokens: int = 8192
    max_cost_ratio: float = 1.8
    max_context_ratio: float = 2.0
    max_context_padding_ratio: float = 2.0
    ready_queue_capacity: int = 64
    control_queue_capacity: int = 16
    prompt_cache_entries: int = 128
    frame_rate: int = 25
    use_model_duration_estimator: bool = True
    lookahead_for_full_batch: bool = True
    lookahead_for_partial_batch: bool = False
    partial_lookahead_max_wait_multiplier: float = 2.0
    max_seed_lookahead: int = 32
    candidate_pack_policy: str = "target_context"
    split_retry_on_memory_error: bool = True
    adaptive_memory_batch_cap: bool = True
    adaptive_memory_cap_recovery_successes: int = 64
    max_generation_batches_before_control: int = 8

    def __post_init__(self):
        for field_name in (
            "max_batch_size",
            "partial_batch_floor",
            "max_total_target_tokens",
            "max_total_context_tokens",
            "ready_queue_capacity",
            "control_queue_capacity",
            "frame_rate",
            "max_seed_lookahead",
            "adaptive_memory_cap_recovery_successes",
            "max_generation_batches_before_control",
        ):
            object.__setattr__(
                self,
                field_name,
                ensure_positive_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "prompt_cache_entries",
            ensure_non_negative_int(
                self.prompt_cache_entries,
                "prompt_cache_entries",
            ),
        )
        for field_name in ("max_wait_ms", "max_context_padding_ratio"):
            object.__setattr__(
                self,
                field_name,
                ensure_non_negative_float(getattr(self, field_name), field_name),
            )
        for field_name in ("max_cost_ratio", "max_context_ratio"):
            object.__setattr__(
                self,
                field_name,
                ensure_min_float(getattr(self, field_name), field_name, 1.0),
            )
        object.__setattr__(
            self,
            "partial_lookahead_max_wait_multiplier",
            ensure_positive_float(
                self.partial_lookahead_max_wait_multiplier,
                "partial_lookahead_max_wait_multiplier",
            ),
        )
        for field_name in (
            "use_model_duration_estimator",
            "lookahead_for_full_batch",
            "lookahead_for_partial_batch",
            "split_retry_on_memory_error",
            "adaptive_memory_batch_cap",
        ):
            object.__setattr__(
                self,
                field_name,
                ensure_bool(getattr(self, field_name), field_name),
            )
        valid_pack_policies = {"target", "context", "target_context"}
        if self.candidate_pack_policy not in valid_pack_policies:
            choices = ", ".join(sorted(valid_pack_policies))
            raise ValueError(
                "candidate_pack_policy must be one of "
                f"{choices}; got {self.candidate_pack_policy!r}"
            )


@dataclass(frozen=True)
class OmniVoiceBatchRequest:
    """One TTS request submitted to the batch scheduler."""

    request_id: str
    text: str
    language: Optional[str] = None
    ref_audio: Optional[Any] = None
    ref_text: Optional[str] = None
    ref_audio_cache_key: Optional[tuple[Any, ...]] = None
    voice_clone_prompt: Optional[VoiceClonePrompt] = None
    instruct: Optional[str] = None
    duration: Optional[float] = None
    speed: Optional[float] = None
    enforce_output_duration: Optional[bool] = None
    cost_tokens_hint: Optional[int] = None
    priority: str = "normal"

    def __post_init__(self):
        object.__setattr__(
            self,
            "request_id",
            _ensure_non_empty_str(self.request_id, "request_id"),
        )
        object.__setattr__(
            self,
            "text",
            _ensure_non_empty_str(self.text, "text"),
        )
        for field_name in ("language", "ref_text", "instruct"):
            object.__setattr__(
                self,
                field_name,
                _ensure_optional_str(getattr(self, field_name), field_name),
            )
        for field_name in ("duration", "speed"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    ensure_positive_float(value, field_name),
                )
        if self.enforce_output_duration is not None:
            object.__setattr__(
                self,
                "enforce_output_duration",
                ensure_bool(
                    self.enforce_output_duration,
                    "enforce_output_duration",
                ),
            )
        if self.cost_tokens_hint is not None:
            object.__setattr__(
                self,
                "cost_tokens_hint",
                ensure_positive_int(self.cost_tokens_hint, "cost_tokens_hint"),
            )
        if self.priority not in {"normal", "high"}:
            raise ValueError("priority must be 'normal' or 'high'")


@dataclass(frozen=True)
class OmniVoiceBatchResult:
    """Generated audio and scheduler timing metadata for one request."""

    request_id: str
    audio: np.ndarray
    sample_rate: int
    batch_size: int
    queue_wait_ms: float
    batch_infer_s: float
    batch_reason: str
    batch_cost_tokens: int = 0
    batch_max_cost_tokens: int = 0
    batch_context_tokens: int = 0
    batch_max_context_tokens: int = 0
    batch_context_padding_ratio: float = 0.0
    generation_profile: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulerSnapshot:
    pending_high: int
    pending_normal: int
    total_batches: int
    total_requests: int
    avg_batch_size: float
    prompt_cache_hits: int
    prompt_cache_misses: int
    last_dispatch_reason: str
    pending_total: int = 0
    pending_control: int = 0
    queue_capacity: int = 0
    control_queue_capacity: int = 0
    max_batch_size: int = 0
    running_batch_size: int = 0
    running_batch_reason: str = "idle"
    running_batch_elapsed_ms: float = 0.0
    running_batch_cost_tokens: int = 0
    running_batch_context_tokens: int = 0
    last_batch_size: int = 0
    last_batch_infer_s: float = 0.0
    last_batch_cost_tokens: int = 0
    last_batch_max_cost_tokens: int = 0
    last_batch_context_tokens: int = 0
    last_batch_max_context_tokens: int = 0
    last_batch_context_padding_ratio: float = 0.0
    total_infer_s: float = 0.0
    total_generated_audio_s: float = 0.0
    infer_rtf: float = 0.0
    failed_batches: int = 0
    split_retry_batches: int = 0
    last_batch_split_retries: int = 0
    last_batch_model_calls: int = 0
    last_batch_max_execution_size: int = 0
    last_batch_profile: dict[str, Any] = field(default_factory=dict)
    adaptive_batch_caps: dict[str, int] = field(default_factory=dict)
    adaptive_cap_success_streaks: dict[str, int] = field(default_factory=dict)


@dataclass
class _QueuedRequest:
    request: OmniVoiceBatchRequest
    future: asyncio.Future[OmniVoiceBatchResult]
    loop: asyncio.AbstractEventLoop
    enqueued_at: float
    cost_tokens: int
    context_tokens: int
    mode: str


@dataclass
class _ControlRequest:
    func: Callable[[], Any]
    future: asyncio.Future[Any]
    loop: asyncio.AbstractEventLoop


@dataclass(frozen=True)
class _BatchRunResult:
    audios: list[np.ndarray]
    split_retries: int
    model_calls: int
    max_execution_batch_size: int
    generation_profile: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ModelBatchCallResult:
    audios: list[np.ndarray]
    generation_profile: dict[str, Any] = field(default_factory=dict)


class _VoiceClonePromptCache:
    def __init__(self, max_entries: int):
        self.max_entries = max(0, max_entries)
        self._entries: OrderedDict[tuple[Any, ...], VoiceClonePrompt] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(
        ref_audio: Any,
        ref_text: Optional[str],
        preprocess_prompt: bool,
        cache_key: Optional[tuple[Any, ...]] = None,
    ) -> Optional[tuple[Any, ...]]:
        if cache_key is not None:
            return ("custom", cache_key, ref_text, preprocess_prompt)
        if isinstance(ref_audio, str):
            path = os.path.abspath(ref_audio)
            try:
                stat = os.stat(path)
            except OSError:
                return ("path", path, ref_text, preprocess_prompt, None, None)
            return (
                "path",
                path,
                ref_text,
                preprocess_prompt,
                stat.st_size,
                stat.st_mtime_ns,
            )
        tuple_marker = _ref_audio_tuple_cache_marker(ref_audio)
        if tuple_marker is not None:
            return ("memory", tuple_marker, ref_text, preprocess_prompt)
        return None

    def get_or_create(
        self,
        *,
        model: Any,
        ref_audio: Any,
        ref_text: Optional[str],
        preprocess_prompt: bool,
        cache_key: Optional[tuple[Any, ...]] = None,
    ) -> VoiceClonePrompt:
        key = self._key(ref_audio, ref_text, preprocess_prompt, cache_key)
        if key is not None and key in self._entries:
            self.hits += 1
            prompt = self._entries.pop(key)
            self._entries[key] = prompt
            return prompt

        self.misses += 1
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt,
        )
        if key is not None and self.max_entries > 0:
            self._entries[key] = prompt
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return prompt

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0


class OmniVoiceBatchScheduler:
    """Single-model online batch scheduler.

    Only one scheduler should own a given GPU-resident model instance. The
    scheduler accepts concurrent callers, but it serializes GPU execution into
    one model call at a time so requests are batched instead of competing via
    multiple CUDA contexts.
    """

    def __init__(
        self,
        model: Any,
        *,
        config: Optional[BatchSchedulerConfig] = None,
        sample_rate: int = 24000,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.model = model
        self.config = config or BatchSchedulerConfig()
        self._validate_config(self.config)
        self.sample_rate = sample_rate
        self.generation_kwargs = generation_kwargs or {}
        self._prompt_cache = _VoiceClonePromptCache(self.config.prompt_cache_entries)
        self._ref_audio_token_cache: OrderedDict[tuple[Any, ...], int] = OrderedDict()

        self._state = threading.Condition()
        self._control: Deque[_ControlRequest] = deque()
        self._ready_high: Deque[_QueuedRequest] = deque()
        self._ready_normal: Deque[_QueuedRequest] = deque()
        self._generation_batches_since_control = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._metrics_lock = threading.Lock()
        self._total_batches = 0
        self._total_requests = 0
        self._last_dispatch_reason = "idle"
        self._running_batch_size = 0
        self._running_batch_reason = "idle"
        self._running_batch_started_at: Optional[float] = None
        self._running_batch_cost_tokens = 0
        self._running_batch_context_tokens = 0
        self._last_batch_size = 0
        self._last_batch_infer_s = 0.0
        self._last_batch_cost_tokens = 0
        self._last_batch_max_cost_tokens = 0
        self._last_batch_context_tokens = 0
        self._last_batch_max_context_tokens = 0
        self._last_batch_context_padding_ratio = 0.0
        self._total_infer_s = 0.0
        self._total_generated_audio_s = 0.0
        self._failed_batches = 0
        self._split_retry_batches = 0
        self._last_batch_split_retries = 0
        self._last_batch_model_calls = 0
        self._last_batch_max_execution_size = 0
        self._last_batch_profile: dict[str, Any] = {}
        self._adaptive_batch_caps: dict[str, int] = {}
        self._adaptive_cap_success_streaks: dict[str, int] = {}

    @staticmethod
    def _validate_config(config: BatchSchedulerConfig) -> None:
        valid_pack_policies = {"target", "context", "target_context"}
        if config.candidate_pack_policy not in valid_pack_policies:
            choices = ", ".join(sorted(valid_pack_policies))
            raise ValueError(
                "candidate_pack_policy must be one of "
                f"{choices}; got {config.candidate_pack_policy!r}"
            )

    async def start(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                return
            self._thread = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._engine_loop,
            name="omnivoice-batch-scheduler",
            daemon=True,
        )
        self._thread.start()

    async def stop(self) -> None:
        self._stop.set()
        with self._state:
            self._state.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if not self._thread.is_alive():
                self._thread = None

        pending: list[_QueuedRequest] = []
        controls: list[_ControlRequest] = []
        with self._state:
            controls.extend(self._control)
            pending.extend(self._ready_high)
            pending.extend(self._ready_normal)
            self._control.clear()
            self._ready_high.clear()
            self._ready_normal.clear()
        for item in controls:
            self._set_control_exception(item, RuntimeError("scheduler stopped"))
        for item in pending:
            self._set_exception(item, RuntimeError("scheduler stopped"))

    async def submit(self, request: OmniVoiceBatchRequest) -> OmniVoiceBatchResult:
        loop = asyncio.get_running_loop()
        queued = self._build_queued_request(request, loop)
        with self._state:
            if self._stop.is_set():
                raise RuntimeError("scheduler stopped")
            if self._pending_total_locked() >= self.config.ready_queue_capacity:
                raise RuntimeError("OmniVoice scheduler queue is full")
            if request.priority == "high":
                self._ready_high.append(queued)
            else:
                self._ready_normal.append(queued)
            self._state.notify()
        try:
            return await queued.future
        except asyncio.CancelledError:
            with self._state:
                if self._remove_queued_locked(queued):
                    self._state.notify_all()
            raise

    async def submit_many(
        self,
        requests: Sequence[OmniVoiceBatchRequest],
    ) -> list[OmniVoiceBatchResult]:
        if not requests:
            return []

        loop = asyncio.get_running_loop()
        queued_requests = [
            self._build_queued_request(request, loop) for request in requests
        ]
        with self._state:
            if self._stop.is_set():
                raise RuntimeError("scheduler stopped")
            if (
                self._pending_total_locked() + len(queued_requests)
                > self.config.ready_queue_capacity
            ):
                raise RuntimeError("OmniVoice scheduler queue is full")
            for queued in queued_requests:
                if queued.request.priority == "high":
                    self._ready_high.append(queued)
                else:
                    self._ready_normal.append(queued)
            self._state.notify_all()
        try:
            return await asyncio.gather(
                *(queued.future for queued in queued_requests)
            )
        except asyncio.CancelledError:
            with self._state:
                removed = False
                for queued in queued_requests:
                    removed = self._remove_queued_locked(queued) or removed
                if removed:
                    self._state.notify_all()
            raise

    def _build_queued_request(
        self,
        request: OmniVoiceBatchRequest,
        loop: asyncio.AbstractEventLoop,
    ) -> _QueuedRequest:
        future: asyncio.Future[OmniVoiceBatchResult] = loop.create_future()
        cost_tokens = self._estimate_cost_tokens(request)
        return _QueuedRequest(
            request=request,
            future=future,
            loop=loop,
            enqueued_at=time.monotonic(),
            cost_tokens=cost_tokens,
            context_tokens=self._estimate_context_tokens(
                request,
                cost_tokens=cost_tokens,
            ),
            mode=self._mode_key(request),
        )

    async def create_voice_clone_prompt(
        self,
        *,
        ref_audio: Any,
        ref_text: Optional[str] = None,
        preprocess_prompt: Optional[bool] = None,
        cache_key: Optional[tuple[Any, ...]] = None,
    ) -> VoiceClonePrompt:
        if preprocess_prompt is None:
            preprocess_prompt = ensure_bool(
                self.generation_kwargs.get("preprocess_prompt", True),
                "preprocess_prompt",
            )
        else:
            preprocess_prompt = ensure_bool(preprocess_prompt, "preprocess_prompt")
        return await self._run_control(
            lambda: self._prompt_cache.get_or_create(
                model=self.model,
                ref_audio=ref_audio,
                ref_text=ref_text,
                preprocess_prompt=preprocess_prompt,
                cache_key=cache_key,
            )
        )

    async def _run_control(self, func: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        control = _ControlRequest(func=func, future=future, loop=loop)
        with self._state:
            if self._stop.is_set():
                raise RuntimeError("scheduler stopped")
            if len(self._control) >= self.config.control_queue_capacity:
                raise RuntimeError("OmniVoice scheduler control queue is full")
            self._control.append(control)
            self._state.notify()
        try:
            return await future
        except asyncio.CancelledError:
            with self._state:
                if self._remove_control_locked(control):
                    self._state.notify_all()
            raise

    def snapshot(self) -> SchedulerSnapshot:
        with self._state:
            pending_high = len(self._ready_high)
            pending_normal = len(self._ready_normal)
            pending_control = len(self._control)
            pending_total = pending_high + pending_normal
            adaptive_batch_caps = dict(self._adaptive_batch_caps)
            adaptive_cap_success_streaks = dict(self._adaptive_cap_success_streaks)
        with self._metrics_lock:
            avg_batch_size = (
                self._total_requests / self._total_batches
                if self._total_batches
                else 0.0
            )
            running_elapsed_ms = 0.0
            if self._running_batch_started_at is not None:
                running_elapsed_ms = (
                    time.monotonic() - self._running_batch_started_at
                ) * 1000.0
            infer_rtf = (
                self._total_infer_s / self._total_generated_audio_s
                if self._total_generated_audio_s > 0.0
                else 0.0
            )
            return SchedulerSnapshot(
                pending_high=pending_high,
                pending_normal=pending_normal,
                total_batches=self._total_batches,
                total_requests=self._total_requests,
                avg_batch_size=avg_batch_size,
                prompt_cache_hits=self._prompt_cache.hits,
                prompt_cache_misses=self._prompt_cache.misses,
                last_dispatch_reason=self._last_dispatch_reason,
                pending_total=pending_total,
                pending_control=pending_control,
                queue_capacity=self.config.ready_queue_capacity,
                control_queue_capacity=self.config.control_queue_capacity,
                max_batch_size=self.config.max_batch_size,
                running_batch_size=self._running_batch_size,
                running_batch_reason=self._running_batch_reason,
                running_batch_elapsed_ms=running_elapsed_ms,
                running_batch_cost_tokens=self._running_batch_cost_tokens,
                running_batch_context_tokens=self._running_batch_context_tokens,
                last_batch_size=self._last_batch_size,
                last_batch_infer_s=self._last_batch_infer_s,
                last_batch_cost_tokens=self._last_batch_cost_tokens,
                last_batch_max_cost_tokens=self._last_batch_max_cost_tokens,
                last_batch_context_tokens=self._last_batch_context_tokens,
                last_batch_max_context_tokens=self._last_batch_max_context_tokens,
                last_batch_context_padding_ratio=(
                    self._last_batch_context_padding_ratio
                ),
                total_infer_s=self._total_infer_s,
                total_generated_audio_s=self._total_generated_audio_s,
                infer_rtf=infer_rtf,
                failed_batches=self._failed_batches,
                split_retry_batches=self._split_retry_batches,
                last_batch_split_retries=self._last_batch_split_retries,
                last_batch_model_calls=self._last_batch_model_calls,
                last_batch_max_execution_size=self._last_batch_max_execution_size,
                last_batch_profile=dict(self._last_batch_profile),
                adaptive_batch_caps=adaptive_batch_caps,
                adaptive_cap_success_streaks=adaptive_cap_success_streaks,
            )

    def reset_metrics(self, *, reset_prompt_cache_stats: bool = True) -> None:
        with self._metrics_lock:
            self._total_batches = 0
            self._total_requests = 0
            self._last_dispatch_reason = "idle"
            self._running_batch_size = 0
            self._running_batch_reason = "idle"
            self._running_batch_started_at = None
            self._running_batch_cost_tokens = 0
            self._running_batch_context_tokens = 0
            self._last_batch_size = 0
            self._last_batch_infer_s = 0.0
            self._last_batch_cost_tokens = 0
            self._last_batch_max_cost_tokens = 0
            self._last_batch_context_tokens = 0
            self._last_batch_max_context_tokens = 0
            self._last_batch_context_padding_ratio = 0.0
            self._total_infer_s = 0.0
            self._total_generated_audio_s = 0.0
            self._failed_batches = 0
            self._split_retry_batches = 0
            self._last_batch_split_retries = 0
            self._last_batch_model_calls = 0
            self._last_batch_max_execution_size = 0
            self._last_batch_profile = {}
            if reset_prompt_cache_stats:
                self._prompt_cache.reset_stats()

    @staticmethod
    def _mode_key(request: OmniVoiceBatchRequest) -> str:
        if request.voice_clone_prompt is not None or request.ref_audio is not None:
            return "clone"
        if request.instruct is not None:
            return "design"
        return "auto"

    def _estimate_cost_tokens(self, request: OmniVoiceBatchRequest) -> int:
        if request.cost_tokens_hint is not None:
            return max(1, int(request.cost_tokens_hint))
        if request.duration is not None:
            return max(1, int(request.duration * self.config.frame_rate))

        model_estimate = self._estimate_cost_tokens_with_model(request)
        if model_estimate is not None:
            return model_estimate

        # Conservative text-only estimate. The model will compute the exact
        # target length later, but this is good enough for queue bucketing.
        char_count = max(1, len(request.text))
        speed = request.speed if request.speed and request.speed > 0 else 1.0
        return max(16, int(char_count * 3.0 / speed))

    def _estimate_context_tokens(
        self,
        request: OmniVoiceBatchRequest,
        *,
        cost_tokens: int,
    ) -> int:
        text_tokens = max(1, len(request.text) // 2)
        ref_text_tokens = max(0, len(request.ref_text or "") // 2)
        ref_audio_tokens = 0
        if request.voice_clone_prompt is not None:
            ref_audio_tokens = int(request.voice_clone_prompt.ref_audio_tokens.size(-1))
            ref_text_tokens = max(
                ref_text_tokens,
                len(request.voice_clone_prompt.ref_text) // 2,
            )
        else:
            ref_audio_tokens = self._cheap_ref_audio_token_count(request.ref_audio) or 0

        # Include a small fixed style-token allowance so auto/design/clone
        # requests with the same target length still get separated when one has
        # a much longer text or prompt context.
        return max(
            1,
            int(cost_tokens + text_tokens + ref_text_tokens + ref_audio_tokens + 16),
        )

    def _estimate_cost_tokens_with_model(
        self,
        request: OmniVoiceBatchRequest,
    ) -> Optional[int]:
        if not self.config.use_model_duration_estimator:
            return None
        estimator = getattr(self.model, "_estimate_target_tokens", None)
        if estimator is None or getattr(self.model, "duration_estimator", None) is None:
            return None

        ref_text: Optional[str] = None
        ref_audio_tokens: Optional[int] = None
        if request.voice_clone_prompt is not None:
            ref_text = request.voice_clone_prompt.ref_text
            ref_audio_tokens = int(request.voice_clone_prompt.ref_audio_tokens.size(-1))
        elif request.ref_text:
            ref_text = request.ref_text
            ref_audio_tokens = self._cheap_ref_audio_token_count(request.ref_audio)

        speed = request.speed if request.speed and request.speed > 0 else 1.0
        try:
            return max(
                1,
                int(estimator(request.text, ref_text, ref_audio_tokens, speed=speed)),
            )
        except Exception:
            return None

    def _cheap_ref_audio_token_count(self, ref_audio: Any) -> Optional[int]:
        if isinstance(ref_audio, str):
            key = self._ref_audio_file_cache_key(ref_audio)
            if key is None:
                return None
            if key in self._ref_audio_token_cache:
                value = self._ref_audio_token_cache.pop(key)
                self._ref_audio_token_cache[key] = value
                return value
            try:
                info = sf.info(ref_audio)
                samples = int(info.frames)
                sample_rate = int(info.samplerate)
            except Exception:
                return None
            value = self._samples_to_audio_tokens(samples, sample_rate)
            if value is not None and self.config.prompt_cache_entries > 0:
                self._ref_audio_token_cache[key] = value
                while len(self._ref_audio_token_cache) > self.config.prompt_cache_entries:
                    self._ref_audio_token_cache.popitem(last=False)
            return value
        elif isinstance(ref_audio, tuple) and len(ref_audio) >= 2:
            waveform, sample_rate = ref_audio[0], ref_audio[1]
            try:
                samples = int(waveform.shape[-1])
                sample_rate = int(sample_rate)
            except (AttributeError, TypeError, ValueError):
                return None
        else:
            return None
        return self._samples_to_audio_tokens(samples, sample_rate)

    def _ref_audio_file_cache_key(self, ref_audio: str) -> Optional[tuple[Any, ...]]:
        path = os.path.abspath(ref_audio)
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return (
            "path",
            path,
            stat.st_size,
            stat.st_mtime_ns,
            self.config.frame_rate,
        )

    def _samples_to_audio_tokens(
        self,
        samples: int,
        sample_rate: int,
    ) -> Optional[int]:
        if sample_rate <= 0:
            return None
        return max(1, int(samples / sample_rate * self.config.frame_rate))

    def _pending_total_locked(self) -> int:
        return len(self._ready_high) + len(self._ready_normal)

    def _has_work_locked(self) -> bool:
        return bool(self._control) or self._pending_total_locked() > 0

    def _engine_loop(self) -> None:
        while not self._stop.is_set():
            control: Optional[_ControlRequest] = None
            batch: list[_QueuedRequest] = []
            reason = "empty"
            with self._state:
                while not self._stop.is_set() and not self._has_work_locked():
                    self._state.wait()
                if self._stop.is_set():
                    break

                while not self._stop.is_set():
                    control, batch, reason = self._pop_next_work_locked()
                    if control is not None or batch:
                        break
                    timeout = self._next_wait_seconds_locked()
                    self._state.wait(timeout=timeout)
                if self._stop.is_set():
                    break

            if control is not None:
                self._execute_control(control)
                continue
            self._execute_batch(batch, reason)

    def _pop_next_work_locked(
        self,
    ) -> tuple[Optional[_ControlRequest], list[_QueuedRequest], str]:
        if self._control and self._control_should_run_before_batch_locked():
            self._generation_batches_since_control = 0
            return self._control.popleft(), [], "control"
        batch, reason = self._try_pop_batch_locked()
        if batch:
            if self._control:
                self._generation_batches_since_control += 1
            else:
                self._generation_batches_since_control = 0
            return None, batch, reason
        if self._control:
            self._generation_batches_since_control = 0
            return self._control.popleft(), [], "control"
        return None, [], "wait"

    def _control_should_run_before_batch_locked(self) -> bool:
        limit = self.config.max_generation_batches_before_control
        return limit > 0 and self._generation_batches_since_control >= limit

    def _try_pop_batch_locked(self) -> tuple[list[_QueuedRequest], str]:
        seeds = self._dispatch_seed_candidates_locked()
        if not seeds:
            return [], "empty"
        for seed in seeds:
            batch, reason = self._ready_batch_for_seed_locked(seed)
            if batch:
                return batch, reason
        return [], "wait"

    def _ready_batch_for_seed_locked(
        self,
        seed: _QueuedRequest,
    ) -> tuple[list[_QueuedRequest], str]:
        batch = self._select_batch_locked(seed)
        compatible_count = len(batch)
        effective_max_batch_size = self._effective_max_batch_size_locked(seed.mode)
        oldest_wait_ms = (time.monotonic() - seed.enqueued_at) * 1000.0
        if compatible_count >= effective_max_batch_size:
            return self._pop_selected_locked(batch), "full"
        if (
            batch
            and compatible_count >= self.config.partial_batch_floor
            and oldest_wait_ms >= self.config.max_wait_ms
        ):
            if batch[0] is seed:
                return self._pop_selected_locked(batch), "timeout"
            if self.config.lookahead_for_partial_batch:
                return self._pop_selected_locked(batch), "timeout_lookahead"
        if batch and batch[0] is seed and oldest_wait_ms >= self.config.max_wait_ms * 2:
            return self._pop_selected_locked(batch), "aging"
        return [], "wait"

    def _dispatch_seed_candidates_locked(self) -> list[_QueuedRequest]:
        seeds: list[_QueuedRequest] = []
        if self._ready_high:
            seeds.append(self._ready_high[0])
        if self._ready_normal:
            seeds.append(self._ready_normal[0])
        return seeds

    def _peek_seed_locked(self) -> Optional[_QueuedRequest]:
        if self._ready_high:
            return self._ready_high[0]
        if self._ready_normal:
            return self._ready_normal[0]
        return None

    def _select_batch_locked(self, fifo_seed: _QueuedRequest) -> list[_QueuedRequest]:
        fifo_batch = self._candidate_batch_for_seed_locked(
            fifo_seed,
            sort_by_cost=False,
        )
        if (
            not self.config.lookahead_for_full_batch
            and not self.config.lookahead_for_partial_batch
        ):
            return fifo_batch

        oldest_wait_ms = (time.monotonic() - fifo_seed.enqueued_at) * 1000.0
        hard_wait_ms = self.config.max_wait_ms * max(
            1.0,
            self.config.partial_lookahead_max_wait_multiplier,
        )
        if oldest_wait_ms >= hard_wait_ms:
            return fifo_batch

        search_queue = self._queue_for_seed_locked(fifo_seed)
        if search_queue is None:
            return fifo_batch
        effective_max_batch_size = self._effective_max_batch_size_locked(
            fifo_seed.mode
        )
        best_batch = fifo_batch
        best_score = self._batch_candidate_score(best_batch)

        for index, candidate in enumerate(search_queue):
            if index >= self.config.max_seed_lookahead:
                break
            batch = self._candidate_batch_for_seed_locked(
                candidate,
                sort_by_cost=True,
            )
            score = self._batch_candidate_score(batch)
            if score > best_score:
                best_batch = batch
                best_score = score

        if len(best_batch) >= effective_max_batch_size:
            return best_batch
        if (
            self.config.lookahead_for_partial_batch
            and oldest_wait_ms >= self.config.max_wait_ms
            and len(best_batch) >= self.config.partial_batch_floor
            and best_score > self._batch_candidate_score(fifo_batch)
        ):
            return best_batch
        return fifo_batch

    def _queue_for_seed_locked(
        self,
        seed: _QueuedRequest,
    ) -> Optional[Deque[_QueuedRequest]]:
        for queue in (self._ready_high, self._ready_normal):
            if any(item is seed for item in queue):
                return queue
        return None

    def _candidate_batch_for_seed_locked(
        self,
        seed: _QueuedRequest,
        *,
        sort_by_cost: bool,
    ) -> list[_QueuedRequest]:
        queued: list[_QueuedRequest] = []
        for queue in (self._ready_high, self._ready_normal):
            queued.extend(queue)

        effective_max_batch_size = self._effective_max_batch_size_locked(seed.mode)
        batch = [seed]
        total_cost = seed.cost_tokens
        total_context = seed.context_tokens
        max_context = seed.context_tokens
        candidates = [
            item
            for item in queued
            if item is not seed and self._is_compatible(seed, item)
        ]

        candidate_orders = [candidates]
        if sort_by_cost:
            candidate_orders = self._candidate_pack_orders(seed, candidates)

        best_batch: list[_QueuedRequest] = []
        best_score = self._batch_candidate_score(best_batch)
        for ordered_candidates in candidate_orders:
            batch = self._build_candidate_batch(
                seed,
                ordered_candidates,
                effective_max_batch_size=effective_max_batch_size,
                initial_total_cost=total_cost,
                initial_total_context=total_context,
                initial_max_context=max_context,
            )
            score = self._batch_candidate_score(batch)
            if score > best_score:
                best_batch = batch
                best_score = score
        return best_batch

    def _candidate_pack_orders(
        self,
        seed: _QueuedRequest,
        candidates: list[_QueuedRequest],
    ) -> list[list[_QueuedRequest]]:
        def target_order() -> list[_QueuedRequest]:
            return sorted(
                candidates,
                key=lambda item: (
                    abs(item.cost_tokens - seed.cost_tokens),
                    abs(item.context_tokens - seed.context_tokens),
                    item.enqueued_at,
                ),
            )

        def context_order() -> list[_QueuedRequest]:
            return sorted(
                candidates,
                key=lambda item: (
                    abs(item.context_tokens - seed.context_tokens),
                    abs(item.cost_tokens - seed.cost_tokens),
                    item.enqueued_at,
                ),
            )

        policy = self.config.candidate_pack_policy
        if policy == "target":
            return [target_order()]
        if policy == "context":
            return [context_order()]
        return [
            target_order(),
            context_order(),
        ]

    def _build_candidate_batch(
        self,
        seed: _QueuedRequest,
        candidates: list[_QueuedRequest],
        *,
        effective_max_batch_size: int,
        initial_total_cost: int,
        initial_total_context: int,
        initial_max_context: int,
    ) -> list[_QueuedRequest]:
        batch = [seed]
        total_cost = initial_total_cost
        total_context = initial_total_context
        max_context = initial_max_context
        for item in candidates:
            if len(batch) >= effective_max_batch_size:
                break
            if total_cost + item.cost_tokens > self.config.max_total_target_tokens:
                continue
            if (
                total_context + item.context_tokens
                > self.config.max_total_context_tokens
            ):
                continue
            next_total_context = total_context + item.context_tokens
            next_max_context = max(max_context, item.context_tokens)
            if self._context_padding_ratio_exceeds(
                batch_size=len(batch) + 1,
                total_context=next_total_context,
                max_context=next_max_context,
            ):
                continue
            batch.append(item)
            total_cost += item.cost_tokens
            total_context = next_total_context
            max_context = next_max_context
        return batch

    @staticmethod
    def _batch_candidate_score(
        batch: list[_QueuedRequest],
    ) -> tuple[int, int, int, int, float]:
        costs = [item.cost_tokens for item in batch]
        contexts = [item.context_tokens for item in batch]
        cost_span = max(costs) - min(costs) if costs else 0
        context_span = max(contexts) - min(contexts) if contexts else 0
        context_padding_work = len(contexts) * max(contexts) if contexts else 0
        oldest_enqueued_at = min((item.enqueued_at for item in batch), default=0.0)
        return (
            len(batch),
            -cost_span,
            -context_span,
            -context_padding_work,
            -oldest_enqueued_at,
        )

    def _compatible_count_locked(self, seed: _QueuedRequest) -> int:
        total = 0
        total_cost = 0
        total_context = 0
        max_context = 0
        effective_max_batch_size = self._effective_max_batch_size_locked(seed.mode)
        for queue in (self._ready_high, self._ready_normal):
            for item in queue:
                if item is seed:
                    total += 1
                    total_cost += item.cost_tokens
                    total_context += item.context_tokens
                    max_context = max(max_context, item.context_tokens)
                    if total >= effective_max_batch_size:
                        return total
                    continue
                if not self._is_compatible(seed, item):
                    continue
                if total_cost + item.cost_tokens > self.config.max_total_target_tokens:
                    continue
                if (
                    total_context + item.context_tokens
                    > self.config.max_total_context_tokens
                ):
                    continue
                next_total_context = total_context + item.context_tokens
                next_max_context = max(max_context, item.context_tokens)
                if self._context_padding_ratio_exceeds(
                    batch_size=total + 1,
                    total_context=next_total_context,
                    max_context=next_max_context,
                ):
                    continue
                total += 1
                total_cost += item.cost_tokens
                total_context = next_total_context
                max_context = next_max_context
                if total >= effective_max_batch_size:
                    return total
        return total

    def _pop_compatible_locked(
        self,
        seed: _QueuedRequest,
        reason: str,
    ) -> list[_QueuedRequest]:
        del reason
        batch: list[_QueuedRequest] = []
        total_cost = 0
        total_context = 0
        max_context = 0
        effective_max_batch_size = self._effective_max_batch_size_locked(seed.mode)
        for queue in (self._ready_high, self._ready_normal):
            kept: Deque[_QueuedRequest] = deque()
            while queue:
                item = queue.popleft()
                if item is seed and len(batch) < effective_max_batch_size:
                    batch.append(item)
                    total_cost += item.cost_tokens
                    total_context += item.context_tokens
                    max_context = max(max_context, item.context_tokens)
                elif (
                    len(batch) < effective_max_batch_size
                    and self._is_compatible(seed, item)
                    and total_cost + item.cost_tokens <= self.config.max_total_target_tokens
                    and total_context + item.context_tokens
                    <= self.config.max_total_context_tokens
                    and not self._context_padding_ratio_exceeds(
                        batch_size=len(batch) + 1,
                        total_context=total_context + item.context_tokens,
                        max_context=max(max_context, item.context_tokens),
                    )
                ):
                    batch.append(item)
                    total_cost += item.cost_tokens
                    total_context += item.context_tokens
                    max_context = max(max_context, item.context_tokens)
                else:
                    kept.append(item)
            queue.extend(kept)
        return batch

    def _pop_selected_locked(
        self,
        selected: list[_QueuedRequest],
    ) -> list[_QueuedRequest]:
        selected_ids = {id(item) for item in selected}
        for queue in (self._ready_high, self._ready_normal):
            kept: Deque[_QueuedRequest] = deque()
            while queue:
                item = queue.popleft()
                if id(item) not in selected_ids:
                    kept.append(item)
            queue.extend(kept)
        return selected

    def _remove_queued_locked(self, queued: _QueuedRequest) -> bool:
        removed = False
        target_id = id(queued)
        for queue in (self._ready_high, self._ready_normal):
            kept: Deque[_QueuedRequest] = deque()
            while queue:
                item = queue.popleft()
                if id(item) == target_id:
                    removed = True
                else:
                    kept.append(item)
            queue.extend(kept)
        return removed

    def _effective_max_batch_size_locked(self, mode: str) -> int:
        if not self.config.adaptive_memory_batch_cap:
            return self.config.max_batch_size
        return min(
            self.config.max_batch_size,
            self._adaptive_batch_caps.get(mode, self.config.max_batch_size),
        )

    def _remove_control_locked(self, control: _ControlRequest) -> bool:
        removed = False
        target_id = id(control)
        kept: Deque[_ControlRequest] = deque()
        while self._control:
            item = self._control.popleft()
            if id(item) == target_id:
                removed = True
            else:
                kept.append(item)
        self._control.extend(kept)
        return removed

    def _is_compatible(self, seed: _QueuedRequest, item: _QueuedRequest) -> bool:
        if item.mode != seed.mode:
            return False
        low_cost = max(1, min(seed.cost_tokens, item.cost_tokens))
        high_cost = max(seed.cost_tokens, item.cost_tokens)
        if high_cost / low_cost > self.config.max_cost_ratio:
            return False

        low_context = max(1, min(seed.context_tokens, item.context_tokens))
        high_context = max(seed.context_tokens, item.context_tokens)
        return high_context / low_context <= self.config.max_context_ratio

    def _context_padding_ratio_exceeds(
        self,
        *,
        batch_size: int,
        total_context: int,
        max_context: int,
    ) -> bool:
        limit = self.config.max_context_padding_ratio
        if limit <= 0.0 or batch_size <= 1 or total_context <= 0:
            return False
        return (batch_size * max_context / total_context) > limit

    def _next_wait_seconds_locked(self) -> float:
        seeds = self._dispatch_seed_candidates_locked()
        if not seeds:
            return 0.1
        now = time.monotonic()
        remaining_ms = min(
            max(1.0, self.config.max_wait_ms - (now - seed.enqueued_at) * 1000.0)
            for seed in seeds
        )
        return remaining_ms / 1000.0

    def _execute_batch(self, batch: Sequence[_QueuedRequest], reason: str) -> None:
        batch = [queued for queued in batch if not queued.future.done()]
        if not batch:
            return

        batch_cost_tokens = sum(queued.cost_tokens for queued in batch)
        batch_max_cost_tokens = max((queued.cost_tokens for queued in batch), default=0)
        batch_context_tokens = sum(queued.context_tokens for queued in batch)
        batch_max_context_tokens = max(
            (queued.context_tokens for queued in batch),
            default=0,
        )
        batch_context_padding_ratio = (
            len(batch) * batch_max_context_tokens / batch_context_tokens
            if batch_context_tokens > 0
            else 0.0
        )
        start = time.monotonic()
        with self._metrics_lock:
            self._running_batch_size = len(batch)
            self._running_batch_reason = reason
            self._running_batch_started_at = start
            self._running_batch_cost_tokens = batch_cost_tokens
            self._running_batch_context_tokens = batch_context_tokens
        try:
            run = self._run_model_batch_with_split_retry(batch)
            audios = run.audios
            if len(audios) != len(batch):
                raise RuntimeError(
                    "model returned a different number of audio outputs than requests"
                )
            self._record_adaptive_batch_result(batch, run)
            infer_s = time.monotonic() - start
            generated_audio_s = sum(len(audio) / self.sample_rate for audio in audios)
            for queued, audio in zip(batch, audios):
                result = OmniVoiceBatchResult(
                    request_id=queued.request.request_id,
                    audio=audio,
                    sample_rate=self.sample_rate,
                    batch_size=len(batch),
                    queue_wait_ms=(start - queued.enqueued_at) * 1000.0,
                    batch_infer_s=infer_s,
                    batch_reason=reason,
                    batch_cost_tokens=batch_cost_tokens,
                    batch_max_cost_tokens=batch_max_cost_tokens,
                    batch_context_tokens=batch_context_tokens,
                    batch_max_context_tokens=batch_max_context_tokens,
                    batch_context_padding_ratio=batch_context_padding_ratio,
                    generation_profile=run.generation_profile,
                )
                self._set_result(queued, result)
            with self._metrics_lock:
                self._total_batches += 1
                self._total_requests += len(batch)
                self._last_dispatch_reason = reason
                self._running_batch_size = 0
                self._running_batch_reason = "idle"
                self._running_batch_started_at = None
                self._running_batch_cost_tokens = 0
                self._running_batch_context_tokens = 0
                self._last_batch_size = len(batch)
                self._last_batch_infer_s = infer_s
                self._last_batch_cost_tokens = batch_cost_tokens
                self._last_batch_max_cost_tokens = batch_max_cost_tokens
                self._last_batch_context_tokens = batch_context_tokens
                self._last_batch_max_context_tokens = batch_max_context_tokens
                self._last_batch_context_padding_ratio = (
                    batch_context_padding_ratio
                )
                self._total_infer_s += infer_s
                self._total_generated_audio_s += generated_audio_s
                self._split_retry_batches += run.split_retries
                self._last_batch_split_retries = run.split_retries
                self._last_batch_model_calls = run.model_calls
                self._last_batch_max_execution_size = run.max_execution_batch_size
                self._last_batch_profile = dict(run.generation_profile)
        except Exception as exc:  # pragma: no cover - exercised in real failures
            with self._metrics_lock:
                self._failed_batches += 1
                self._running_batch_size = 0
                self._running_batch_reason = "idle"
                self._running_batch_started_at = None
                self._running_batch_cost_tokens = 0
                self._running_batch_context_tokens = 0
                self._last_dispatch_reason = "failed"
                self._last_batch_split_retries = 0
                self._last_batch_model_calls = 0
                self._last_batch_max_execution_size = 0
                self._last_batch_profile = {}
            for queued in batch:
                self._set_exception(queued, exc)

    def _execute_control(self, control: _ControlRequest) -> None:
        if control.future.done():
            return
        try:
            result = control.func()
            self._set_control_result(control, result)
        except Exception as exc:  # pragma: no cover - exercised in real failures
            self._set_control_exception(control, exc)

    def _set_exception(self, queued: _QueuedRequest, exc: Exception) -> None:
        queued.loop.call_soon_threadsafe(self._safe_set_exception, queued.future, exc)

    def _set_control_exception(
        self,
        control: _ControlRequest,
        exc: Exception,
    ) -> None:
        control.loop.call_soon_threadsafe(
            self._safe_set_exception,
            control.future,
            exc,
        )

    def _set_control_result(self, control: _ControlRequest, result: Any) -> None:
        control.loop.call_soon_threadsafe(self._safe_set_result, control.future, result)

    def _set_result(
        self,
        queued: _QueuedRequest,
        result: OmniVoiceBatchResult,
    ) -> None:
        queued.loop.call_soon_threadsafe(self._safe_set_result, queued.future, result)

    @staticmethod
    def _safe_set_exception(
        future: asyncio.Future[Any],
        exc: Exception,
    ) -> None:
        if not future.done():
            future.set_exception(exc)

    @staticmethod
    def _safe_set_result(
        future: asyncio.Future[Any],
        result: Any,
    ) -> None:
        if not future.done():
            future.set_result(result)

    def _run_model_batch_with_split_retry(
        self,
        batch: Sequence[_QueuedRequest],
    ) -> _BatchRunResult:
        try:
            call = self._run_model_batch(batch)
            return _BatchRunResult(
                audios=call.audios,
                split_retries=0,
                model_calls=1,
                max_execution_batch_size=len(batch),
                generation_profile=call.generation_profile,
            )
        except Exception as exc:
            if (
                not self.config.split_retry_on_memory_error
                or len(batch) <= 1
                or not self._is_retryable_memory_error(exc)
            ):
                raise
            self._clear_accelerator_cache()
            midpoint = len(batch) // 2
            left = self._run_model_batch_with_split_retry(batch[:midpoint])
            right = self._run_model_batch_with_split_retry(batch[midpoint:])
            return _BatchRunResult(
                audios=[*left.audios, *right.audios],
                split_retries=left.split_retries + right.split_retries + 1,
                model_calls=left.model_calls + right.model_calls + 1,
                max_execution_batch_size=max(
                    left.max_execution_batch_size,
                    right.max_execution_batch_size,
                ),
                generation_profile={
                    "split_children": [
                        left.generation_profile,
                        right.generation_profile,
                    ]
                },
            )

    @staticmethod
    def _is_retryable_memory_error(exc: Exception) -> bool:
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        retryable_markers = (
            "outofmemoryerror",
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "hip out of memory",
            "mps backend out of memory",
        )
        return any(marker in name or marker in message for marker in retryable_markers)

    @staticmethod
    def _clear_accelerator_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def _record_adaptive_batch_result(
        self,
        batch: Sequence[_QueuedRequest],
        run: _BatchRunResult,
    ) -> None:
        if not self.config.adaptive_memory_batch_cap or not batch:
            return

        mode = batch[0].mode
        requested_size = len(batch)
        with self._state:
            current_cap = self._adaptive_batch_caps.get(
                mode,
                self.config.max_batch_size,
            )
            if run.split_retries > 0 and run.max_execution_batch_size < requested_size:
                learned_cap = max(1, run.max_execution_batch_size)
                self._adaptive_batch_caps[mode] = min(current_cap, learned_cap)
                self._adaptive_cap_success_streaks[mode] = 0
                return

            effective_cap = min(self.config.max_batch_size, current_cap)
            if (
                requested_size >= effective_cap
                and current_cap < self.config.max_batch_size
            ):
                next_streak = self._adaptive_cap_success_streaks.get(mode, 0) + 1
                threshold = max(1, self.config.adaptive_memory_cap_recovery_successes)
                if next_streak >= threshold:
                    recovered_cap = min(
                        self.config.max_batch_size,
                        current_cap + 1,
                    )
                    if recovered_cap >= self.config.max_batch_size:
                        self._adaptive_batch_caps.pop(mode, None)
                        self._adaptive_cap_success_streaks.pop(mode, None)
                    else:
                        self._adaptive_batch_caps[mode] = recovered_cap
                        self._adaptive_cap_success_streaks[mode] = 0
                else:
                    self._adaptive_cap_success_streaks[mode] = next_streak
                return

            if current_cap < self.config.max_batch_size:
                self._adaptive_cap_success_streaks[mode] = 0
            else:
                self._adaptive_cap_success_streaks.pop(mode, None)

    def _run_model_batch(self, batch: Sequence[_QueuedRequest]) -> _ModelBatchCallResult:
        requests = [item.request for item in batch]
        texts = [req.text for req in requests]
        languages = [req.language for req in requests]
        durations = [req.duration for req in requests]
        speeds = [req.speed for req in requests]
        default_enforce_output_duration = self.generation_kwargs.get(
            "enforce_output_duration",
            False,
        )
        enforce_flags = resolve_optional_bool_flags(
            [req.enforce_output_duration for req in requests],
            len(requests),
            "enforce_output_duration",
            default=default_enforce_output_duration,
        )
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.pop("enforce_output_duration", None)
        kwargs: dict[str, Any] = {
            "text": texts,
            "language": languages,
            "enforce_output_duration": enforce_flags,
            **generation_kwargs,
        }
        if any(duration is not None for duration in durations):
            kwargs["duration"] = durations
        if any(speed is not None for speed in speeds):
            kwargs["speed"] = speeds

        mode = batch[0].mode
        if mode == "clone":
            kwargs["voice_clone_prompt"] = [
                self._resolve_voice_clone_prompt(req) for req in requests
            ]
        else:
            instructs = [req.instruct for req in requests]
            if any(instruct is not None for instruct in instructs):
                kwargs["instruct"] = instructs

        audios = self.model.generate(**kwargs)
        profile = getattr(self.model, "last_generation_profile", None) or {}
        if profile:
            profile = dict(profile)
            metadata = dict(profile.get("metadata", {}))
            metadata["online_enforce_output_duration"] = enforce_flags
            profile["metadata"] = metadata
        return _ModelBatchCallResult(audios=audios, generation_profile=profile)

    def _resolve_voice_clone_prompt(
        self,
        request: OmniVoiceBatchRequest,
    ) -> VoiceClonePrompt:
        if request.voice_clone_prompt is not None:
            return request.voice_clone_prompt
        if request.ref_audio is None:
            raise ValueError("voice clone request requires ref_audio or prompt")
        preprocess_prompt = ensure_bool(
            self.generation_kwargs.get("preprocess_prompt", True),
            "preprocess_prompt",
        )
        return self._prompt_cache.get_or_create(
            model=self.model,
            ref_audio=request.ref_audio,
            ref_text=request.ref_text,
            preprocess_prompt=preprocess_prompt,
            cache_key=request.ref_audio_cache_key,
        )
