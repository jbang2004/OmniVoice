"""Experimental step-level online scheduler for OmniVoice.

Unlike ``OmniVoiceBatchScheduler`` which batches only before ``generate()``
starts, this scheduler keeps per-request generation state and repacks active
requests before every diffusion/unmasking step. New compatible requests can join
between steps, which is the core mechanism behind continuous batching.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional

import torch

from omnivoice.models.generation import OmniVoiceGenerationConfig, fit_audio_to_duration
from omnivoice.serving.batcher import (
    OmniVoiceBatchRequest,
    OmniVoiceBatchResult,
    _VoiceClonePromptCache,
)
from omnivoice.serving.stepwise import (
    StepwiseGenerationState,
    create_stepwise_states,
    run_generation_step,
)


@dataclass(frozen=True)
class StepwiseSchedulerConfig:
    max_running_requests: int = 8
    max_wait_ms: float = 120.0
    partial_batch_floor: int = 2
    max_total_target_tokens: int = 4096
    max_total_context_tokens: int = 8192
    max_cost_ratio: float = 2.0
    max_context_ratio: float = 2.0
    max_context_padding_ratio: float = 2.0
    ready_queue_capacity: int = 64
    control_queue_capacity: int = 16
    prompt_cache_entries: int = 128
    frame_rate: int = 25
    profile_cuda: bool = False
    compile_static_shape: bool = False
    seq_len_bucket_multiple: int = 64
    target_len_bucket_multiple: int = 64
    lookahead_for_full_batch: bool = True
    max_seed_lookahead: int = 32


@dataclass(frozen=True)
class StepwiseSchedulerSnapshot:
    pending: int
    pending_control: int
    running: int
    queue_capacity: int
    control_queue_capacity: int
    total_requests: int
    total_steps: int
    avg_step_batch_size: float
    max_step_batch_size: int
    prompt_cache_hits: int
    prompt_cache_misses: int
    prepare_s: float
    step_s: float
    pack_s: float
    forward_s: float
    update_s: float
    decode_s: float
    unique_step_shapes: int


@dataclass
class _WaitingRequest:
    request: OmniVoiceBatchRequest
    future: asyncio.Future[OmniVoiceBatchResult]
    loop: asyncio.AbstractEventLoop
    enqueued_at: float
    estimated_target_tokens: int
    estimated_context_tokens: int


@dataclass
class _RunningRequest:
    waiting: _WaitingRequest
    state: StepwiseGenerationState
    ref_rms: Optional[float]
    started_at: float
    requested_duration: Optional[float] = None
    max_step_batch_size: int = 1


@dataclass
class _ControlRequest:
    func: Callable[[], Any]
    future: asyncio.Future[Any]
    loop: asyncio.AbstractEventLoop


class StepwiseOmniVoiceScheduler:
    """Continuous-style scheduler for one resident OmniVoice model instance."""

    def __init__(
        self,
        model: Any,
        *,
        scheduler_config: Optional[StepwiseSchedulerConfig] = None,
        generation_config: Optional[OmniVoiceGenerationConfig] = None,
        sample_rate: int = 24000,
    ):
        self.model = model
        self.scheduler_config = scheduler_config or StepwiseSchedulerConfig()
        self.generation_config = generation_config or OmniVoiceGenerationConfig()
        self.sample_rate = sample_rate
        self._validate_config()
        self._prompt_cache = _VoiceClonePromptCache(
            self.scheduler_config.prompt_cache_entries
        )

        self._state = threading.Condition()
        self._waiting: Deque[_WaitingRequest] = deque()
        self._control: Deque[_ControlRequest] = deque()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._metrics_lock = threading.Lock()
        self._running_count = 0
        self._total_requests = 0
        self._total_steps = 0
        self._total_step_batch_size = 0
        self._max_step_batch_size = 0
        self._prepare_s = 0.0
        self._step_s = 0.0
        self._pack_s = 0.0
        self._forward_s = 0.0
        self._update_s = 0.0
        self._decode_s = 0.0
        self._step_shapes_seen: set[tuple[int, int, int]] = set()

    async def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._engine_loop,
            name="omnivoice-stepwise-scheduler",
            daemon=True,
        )
        self._thread.start()

    async def stop(self) -> None:
        self._stop.set()
        with self._state:
            self._state.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        pending: list[_WaitingRequest] = []
        controls: list[_ControlRequest] = []
        with self._state:
            pending.extend(self._waiting)
            controls.extend(self._control)
            self._waiting.clear()
            self._control.clear()
        for item in pending:
            self._set_exception(item, RuntimeError("scheduler stopped"))
        for item in controls:
            self._set_control_exception(item, RuntimeError("scheduler stopped"))

    async def submit(self, request: OmniVoiceBatchRequest) -> OmniVoiceBatchResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[OmniVoiceBatchResult] = loop.create_future()
        waiting = _WaitingRequest(
            request=request,
            future=future,
            loop=loop,
            enqueued_at=time.monotonic(),
            estimated_target_tokens=self._estimate_target_tokens(request),
            estimated_context_tokens=self._estimate_context_tokens(request),
        )
        with self._state:
            if len(self._waiting) >= self.scheduler_config.ready_queue_capacity:
                raise RuntimeError("OmniVoice stepwise scheduler queue is full")
            self._waiting.append(waiting)
            self._state.notify()
        try:
            return await future
        except asyncio.CancelledError:
            with self._state:
                self._remove_waiting_locked(waiting)
            raise

    async def create_voice_clone_prompt(
        self,
        *,
        ref_audio: Any,
        ref_text: Optional[str] = None,
        preprocess_prompt: Optional[bool] = None,
    ) -> Any:
        if preprocess_prompt is None:
            preprocess_prompt = bool(self.generation_config.preprocess_prompt)
        return await self._run_control(
            lambda: self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                preprocess_prompt=bool(preprocess_prompt),
            )
        )

    async def _run_control(self, func: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        control = _ControlRequest(func=func, future=future, loop=loop)
        with self._state:
            if len(self._control) >= self.scheduler_config.control_queue_capacity:
                raise RuntimeError("OmniVoice stepwise scheduler control queue is full")
            self._control.append(control)
            self._state.notify()
        try:
            return await future
        except asyncio.CancelledError:
            with self._state:
                self._remove_control_locked(control)
            raise

    def snapshot(self) -> StepwiseSchedulerSnapshot:
        with self._state:
            pending = len(self._waiting)
            pending_control = len(self._control)
        with self._metrics_lock:
            avg = (
                self._total_step_batch_size / self._total_steps
                if self._total_steps
                else 0.0
            )
            return StepwiseSchedulerSnapshot(
                pending=pending,
                pending_control=pending_control,
                running=self._running_count,
                queue_capacity=self.scheduler_config.ready_queue_capacity,
                control_queue_capacity=self.scheduler_config.control_queue_capacity,
                total_requests=self._total_requests,
                total_steps=self._total_steps,
                avg_step_batch_size=avg,
                max_step_batch_size=self._max_step_batch_size,
                prompt_cache_hits=self._prompt_cache.hits,
                prompt_cache_misses=self._prompt_cache.misses,
                prepare_s=self._prepare_s,
                step_s=self._step_s,
                pack_s=self._pack_s,
                forward_s=self._forward_s,
                update_s=self._update_s,
                decode_s=self._decode_s,
                unique_step_shapes=len(self._step_shapes_seen),
            )

    def _engine_loop(self) -> None:
        running: list[_RunningRequest] = []
        while not self._stop.is_set():
            to_admit: list[_WaitingRequest] = []
            control: Optional[_ControlRequest] = None
            running = self._drop_done_running(running)
            with self._state:
                while (
                    not self._stop.is_set()
                    and not running
                    and not self._control
                    and not self._should_start_waiting_locked()
                ):
                    self._state.wait(timeout=self._next_wait_seconds_locked())
                if self._stop.is_set():
                    break
                if (
                    not running
                    and self._control
                    and not self._should_start_waiting_locked()
                ):
                    control = self._control.popleft()
                else:
                    to_admit = self._pop_admissible_locked(
                        capacity=self.scheduler_config.max_running_requests
                        - len(running),
                        running=running,
                    )

            if control is not None:
                self._execute_control(control)
                continue

            for waiting in to_admit:
                if waiting.future.done():
                    continue
                try:
                    prepare_start = time.monotonic()
                    running.append(self._prepare_running(waiting))
                    with self._metrics_lock:
                        self._prepare_s += time.monotonic() - prepare_start
                except Exception as exc:
                    self._set_exception(waiting, exc)

            running = self._drop_done_running(running)
            with self._metrics_lock:
                self._running_count = len(running)

            if not running:
                continue

            try:
                active_states = [item.state for item in running]
                static_shape = self._step_shape_for_running(running)
                step_start = time.monotonic()
                with torch.inference_mode():
                    timings = run_generation_step(
                        self.model,
                        active_states,
                        self.generation_config,
                        profile_cuda=self.scheduler_config.profile_cuda,
                        **static_shape,
                    )
                with self._metrics_lock:
                    self._step_s += time.monotonic() - step_start
                    self._pack_s += timings.pack_s
                    self._forward_s += timings.forward_s
                    self._update_s += timings.update_s
                    self._step_shapes_seen.add(
                        (
                            static_shape.get("batch_size_pad") or len(running),
                            static_shape.get("seq_len_pad")
                            or max(item.state.cond_len for item in running),
                            static_shape.get("target_len_pad")
                            or max(item.state.target_len for item in running),
                        )
                    )
            except Exception as exc:
                for item in running:
                    self._set_exception(item.waiting, exc)
                running.clear()
                continue

            step_batch_size = len(running)
            for item in running:
                item.max_step_batch_size = max(item.max_step_batch_size, step_batch_size)
            with self._metrics_lock:
                self._total_steps += 1
                self._total_step_batch_size += step_batch_size
                self._max_step_batch_size = max(
                    self._max_step_batch_size,
                    step_batch_size,
                )

            still_running = []
            for item in running:
                if item.waiting.future.done():
                    continue
                if item.state.completed:
                    self._finish_request(item)
                else:
                    still_running.append(item)
            running = still_running
            with self._metrics_lock:
                self._running_count = len(running)

        for item in running:
            self._set_exception(item.waiting, RuntimeError("scheduler stopped"))
        with self._metrics_lock:
            self._running_count = 0

    def _should_start_waiting_locked(self) -> bool:
        self._drop_done_waiting_locked()
        if not self._waiting:
            return False
        if len(self._waiting) >= self.scheduler_config.max_running_requests:
            return True
        oldest_wait_ms = (time.monotonic() - self._waiting[0].enqueued_at) * 1000.0
        if (
            len(self._waiting) >= self.scheduler_config.partial_batch_floor
            and oldest_wait_ms >= self.scheduler_config.max_wait_ms
        ):
            return True
        return oldest_wait_ms >= self.scheduler_config.max_wait_ms * 2

    def _next_wait_seconds_locked(self) -> float:
        if not self._waiting:
            return 0.1
        elapsed_ms = (time.monotonic() - self._waiting[0].enqueued_at) * 1000.0
        return max(0.001, (self.scheduler_config.max_wait_ms - elapsed_ms) / 1000.0)

    def _pop_admissible_locked(
        self,
        capacity: int,
        running: list[_RunningRequest],
    ) -> list[_WaitingRequest]:
        self._drop_done_waiting_locked()
        if capacity <= 0:
            return []

        running_costs = [item.state.target_len for item in running]
        running_contexts = [item.state.cond_len for item in running]
        total_cost = sum(running_costs)
        total_context = sum(running_contexts)
        seed_cost = running_costs[0] if running_costs else None
        seed_context = running_contexts[0] if running_contexts else None
        max_context = max(running_contexts, default=0)

        popped = []
        if seed_cost is None:
            selected = self._select_waiting_batch_locked(capacity)
            selected_ids = {id(item) for item in selected}
            self._waiting = deque(
                item for item in self._waiting if id(item) not in selected_ids
            )
            return selected

        kept: Deque[_WaitingRequest] = deque()
        while self._waiting:
            item = self._waiting.popleft()
            if (
                len(popped) < capacity
                and self._is_admissible(
                    item.estimated_target_tokens,
                    item.estimated_context_tokens,
                    seed_cost,
                    seed_context,
                    total_cost,
                    total_context,
                    max_context,
                    current_batch_size=len(running) + len(popped),
                )
            ):
                popped.append(item)
                total_cost += item.estimated_target_tokens
                total_context += item.estimated_context_tokens
                max_context = max(max_context, item.estimated_context_tokens)
            else:
                kept.append(item)
        self._waiting.extend(kept)
        return popped

    def _select_waiting_batch_locked(
        self,
        capacity: int,
    ) -> list[_WaitingRequest]:
        if not self._waiting:
            return []
        fifo_seed = self._waiting[0]
        fifo_batch = self._candidate_waiting_batch_for_seed_locked(
            fifo_seed,
            capacity,
            sort_by_cost=False,
        )
        if not self.scheduler_config.lookahead_for_full_batch:
            return fifo_batch

        oldest_wait_ms = (time.monotonic() - fifo_seed.enqueued_at) * 1000.0
        if oldest_wait_ms >= self.scheduler_config.max_wait_ms:
            return fifo_batch

        best_batch = fifo_batch
        best_score = _waiting_batch_candidate_score(best_batch)

        for index, candidate in enumerate(self._waiting):
            if index >= self.scheduler_config.max_seed_lookahead:
                break
            batch = self._candidate_waiting_batch_for_seed_locked(
                candidate,
                capacity,
                sort_by_cost=True,
            )
            score = _waiting_batch_candidate_score(batch)
            if score > best_score:
                best_batch = batch
                best_score = score

        if len(best_batch) >= capacity:
            return best_batch
        return fifo_batch

    def _select_waiting_seed_locked(
        self,
        capacity: int,
    ) -> Optional[_WaitingRequest]:
        batch = self._select_waiting_batch_locked(capacity)
        return batch[0] if batch else None

    def _candidate_waiting_batch_for_seed_locked(
        self,
        seed: _WaitingRequest,
        capacity: int,
        *,
        sort_by_cost: bool,
    ) -> list[_WaitingRequest]:
        seed_cost = seed.estimated_target_tokens
        batch = [seed]
        total_cost = seed.estimated_target_tokens
        total_context = seed.estimated_context_tokens
        max_context = seed.estimated_context_tokens
        candidates = [
            item
            for item in self._waiting
            if item is not seed
            and self._is_admissible(
                item.estimated_target_tokens,
                item.estimated_context_tokens,
                seed_cost,
                seed.estimated_context_tokens,
                total_cost=total_cost,
                total_context=total_context,
                max_context=max_context,
                current_batch_size=1,
            )
        ]
        if sort_by_cost:
            candidates.sort(
                key=lambda item: (
                    abs(item.estimated_target_tokens - seed_cost),
                    abs(item.estimated_context_tokens - seed.estimated_context_tokens),
                    item.enqueued_at,
                )
            )

        for item in candidates:
            if len(batch) >= capacity:
                break
            if not self._is_admissible(
                item.estimated_target_tokens,
                item.estimated_context_tokens,
                seed_cost,
                seed.estimated_context_tokens,
                total_cost,
                total_context,
                max_context,
                current_batch_size=len(batch),
            ):
                continue
            batch.append(item)
            total_cost += item.estimated_target_tokens
            total_context += item.estimated_context_tokens
            max_context = max(max_context, item.estimated_context_tokens)
        return batch

    def _is_admissible(
        self,
        candidate_cost: int,
        candidate_context: int,
        seed_cost: Optional[int],
        seed_context: Optional[int],
        total_cost: int,
        total_context: int,
        max_context: int,
        *,
        current_batch_size: int,
    ) -> bool:
        if total_cost + candidate_cost > self.scheduler_config.max_total_target_tokens:
            return False
        next_total_context = total_context + candidate_context
        if next_total_context > self.scheduler_config.max_total_context_tokens:
            return False
        if seed_cost is None:
            return True
        low = max(1, min(seed_cost, candidate_cost))
        high = max(seed_cost, candidate_cost)
        if high / low > self.scheduler_config.max_cost_ratio:
            return False
        if seed_context is not None:
            low_context = max(1, min(seed_context, candidate_context))
            high_context = max(seed_context, candidate_context)
            if high_context / low_context > self.scheduler_config.max_context_ratio:
                return False
        return not self._context_padding_ratio_exceeds(
            batch_size=current_batch_size + 1,
            total_context=next_total_context,
            max_context=max(max_context, candidate_context),
        )

    def _context_padding_ratio_exceeds(
        self,
        *,
        batch_size: int,
        total_context: int,
        max_context: int,
    ) -> bool:
        limit = self.scheduler_config.max_context_padding_ratio
        if limit <= 0.0 or batch_size <= 1 or total_context <= 0:
            return False
        return (batch_size * max_context / total_context) > limit

    def _validate_config(self) -> None:
        if self.scheduler_config.max_running_requests < 1:
            raise ValueError("max_running_requests must be >= 1")
        if self.scheduler_config.max_total_context_tokens < 1:
            raise ValueError("max_total_context_tokens must be >= 1")
        if self.scheduler_config.max_context_ratio < 1.0:
            raise ValueError("max_context_ratio must be >= 1.0")
        if self.scheduler_config.seq_len_bucket_multiple < 1:
            raise ValueError("seq_len_bucket_multiple must be >= 1")
        if self.scheduler_config.target_len_bucket_multiple < 1:
            raise ValueError("target_len_bucket_multiple must be >= 1")
        if self.scheduler_config.max_seed_lookahead < 1:
            raise ValueError("max_seed_lookahead must be >= 1")

    def _step_shape_for_running(
        self,
        running: list[_RunningRequest],
    ) -> dict[str, int]:
        return build_static_step_shape(
            [item.state for item in running],
            self.scheduler_config,
        )

    def _estimate_target_tokens(self, request: OmniVoiceBatchRequest) -> int:
        if request.duration is not None:
            return max(1, int(request.duration * self.scheduler_config.frame_rate))

        char_count = max(1, len(request.text))
        speed = request.speed if request.speed and request.speed > 0 else 1.0
        return max(16, int(char_count * 3.0 / speed))

    def _estimate_context_tokens(self, request: OmniVoiceBatchRequest) -> int:
        cost_tokens = self._estimate_target_tokens(request)
        text_tokens = max(1, len(request.text) // 2)
        ref_text_tokens = max(0, len(request.ref_text or "") // 2)
        ref_audio_tokens = 0
        if request.voice_clone_prompt is not None:
            ref_audio_tokens = int(request.voice_clone_prompt.ref_audio_tokens.size(-1))
            ref_text_tokens = max(
                ref_text_tokens,
                len(request.voice_clone_prompt.ref_text) // 2,
            )
        return max(
            1,
            int(cost_tokens + text_tokens + ref_text_tokens + ref_audio_tokens + 16),
        )

    def _prepare_running(self, waiting: _WaitingRequest) -> _RunningRequest:
        request = waiting.request
        kwargs: dict[str, Any] = {
            "text": request.text,
            "language": request.language,
            "instruct": request.instruct,
            "duration": request.duration,
            "speed": request.speed,
            "preprocess_prompt": self.generation_config.preprocess_prompt,
        }

        if request.voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = request.voice_clone_prompt
        elif request.ref_audio is not None:
            kwargs["voice_clone_prompt"] = self._prompt_cache.get_or_create(
                model=self.model,
                ref_audio=request.ref_audio,
                ref_text=request.ref_text,
                preprocess_prompt=self.generation_config.preprocess_prompt,
            )

        with torch.inference_mode():
            task = self.model._preprocess_all(**kwargs)
            state = create_stepwise_states(
                self.model,
                task,
                self.generation_config,
                request_ids=[request.request_id],
            )[0]
        with self._metrics_lock:
            self._total_requests += 1
        return _RunningRequest(
            waiting=waiting,
            state=state,
            ref_rms=task.ref_rms[0],
            requested_duration=task.requested_durations[0]
            if task.requested_durations
            else None,
            started_at=time.monotonic(),
        )

    def _finish_request(self, item: _RunningRequest) -> None:
        try:
            decode_start = time.monotonic()
            with torch.inference_mode():
                audio = self.model._decode_and_post_process(
                    item.state.tokens,
                    item.ref_rms,
                    self.generation_config,
                )
                if self.generation_config.enforce_output_duration:
                    audio = fit_audio_to_duration(
                        audio,
                        item.requested_duration,
                        self.sample_rate,
                    )
            with self._metrics_lock:
                self._decode_s += time.monotonic() - decode_start
            result = OmniVoiceBatchResult(
                request_id=item.waiting.request.request_id,
                audio=audio,
                sample_rate=self.sample_rate,
                batch_size=item.max_step_batch_size,
                queue_wait_ms=(item.started_at - item.waiting.enqueued_at) * 1000.0,
                batch_infer_s=time.monotonic() - item.started_at,
                batch_reason="stepwise",
            )
            self._set_result(item.waiting, result)
        except Exception as exc:
            self._set_exception(item.waiting, exc)

    def _set_exception(self, waiting: _WaitingRequest, exc: Exception) -> None:
        waiting.loop.call_soon_threadsafe(self._safe_set_exception, waiting.future, exc)

    def _set_result(
        self,
        waiting: _WaitingRequest,
        result: OmniVoiceBatchResult,
    ) -> None:
        waiting.loop.call_soon_threadsafe(self._safe_set_result, waiting.future, result)

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

    def _execute_control(self, control: _ControlRequest) -> None:
        if control.future.done():
            return
        try:
            self._set_control_result(control, control.func())
        except Exception as exc:
            self._set_control_exception(control, exc)

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

    def _drop_done_waiting_locked(self) -> None:
        if self._waiting:
            self._waiting = deque(item for item in self._waiting if not item.future.done())

    def _drop_done_running(
        self,
        running: list[_RunningRequest],
    ) -> list[_RunningRequest]:
        return [item for item in running if not item.waiting.future.done()]

    def _remove_waiting_locked(self, waiting: _WaitingRequest) -> bool:
        removed = False
        target_id = id(waiting)
        kept: Deque[_WaitingRequest] = deque()
        while self._waiting:
            item = self._waiting.popleft()
            if id(item) == target_id:
                removed = True
                continue
            kept.append(item)
        self._waiting.extend(kept)
        return removed

    def _remove_control_locked(self, control: _ControlRequest) -> bool:
        removed = False
        target_id = id(control)
        kept: Deque[_ControlRequest] = deque()
        while self._control:
            item = self._control.popleft()
            if id(item) == target_id:
                removed = True
                continue
            kept.append(item)
        self._control.extend(kept)
        return removed


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _waiting_batch_candidate_score(
    batch: list[_WaitingRequest],
) -> tuple[int, int, int, int, float]:
    costs = [item.estimated_target_tokens for item in batch]
    contexts = [item.estimated_context_tokens for item in batch]
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


def build_static_step_shape(
    states: list[StepwiseGenerationState],
    config: StepwiseSchedulerConfig,
) -> dict[str, int]:
    if not config.compile_static_shape:
        return {}
    max_cond_len = max(state.cond_len for state in states)
    max_target_len = max(state.target_len for state in states)
    target_len_pad = _ceil_to_multiple(
        max_target_len,
        config.target_len_bucket_multiple,
    )
    seq_len_pad = _ceil_to_multiple(
        max(max_cond_len, target_len_pad),
        config.seq_len_bucket_multiple,
    )
    return {
        "batch_size_pad": config.max_running_requests,
        "seq_len_pad": seq_len_pad,
        "target_len_pad": target_len_pad,
    }
