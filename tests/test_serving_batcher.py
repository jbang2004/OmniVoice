import asyncio
import tempfile
import threading
import time
import unittest
from collections import deque
from unittest import mock

import numpy as np
import soundfile as sf
import torch

from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.serving.batcher import _ControlRequest, _QueuedRequest
from omnivoice.serving import (
    BatchSchedulerConfig,
    OmniVoiceBatchRequest,
    OmniVoiceBatchResult,
    OmniVoiceBatchScheduler,
)


class FakeModel:
    def __init__(self):
        self.calls = []
        self.prompt_calls = []

    def generate(self, **kwargs):
        texts = kwargs["text"]
        self.calls.append(list(texts))
        return [np.zeros(240, dtype=np.float32) for _ in texts]

    def create_voice_clone_prompt(self, ref_audio, ref_text, preprocess_prompt=True):
        self.prompt_calls.append(
            {
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "preprocess_prompt": preprocess_prompt,
                "thread": threading.current_thread().name,
            }
        )
        return VoiceClonePrompt(
            ref_audio_tokens=torch.zeros((1, 1), dtype=torch.long),
            ref_text=ref_text or "auto text",
            ref_rms=0.1,
        )


class ObservedModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.scheduler = None
        self.snapshot_inside_generate = None

    def generate(self, **kwargs):
        self.snapshot_inside_generate = self.scheduler.snapshot()
        return super().generate(**kwargs)


class ProfiledModel(FakeModel):
    def generate(self, **kwargs):
        audios = super().generate(**kwargs)
        self.last_generation_profile = {
            "timings_s": {"total_s": 1.0, "iterative_forward_s": 0.7},
            "metadata": {"batch_size": len(kwargs["text"])},
        }
        return audios


class FixedLengthModel(FakeModel):
    def __init__(self, length: int):
        super().__init__()
        self.length = length
        self.kwargs = None
        self.sampling_rate = 1000

    def generate(self, **kwargs):
        self.kwargs = kwargs
        texts = kwargs["text"]
        self.calls.append(list(texts))
        audios = [
            np.ones(self.length, dtype=np.float32) * (index + 1)
            for index, _ in enumerate(texts)
        ]
        durations = kwargs.get("duration") or [None] * len(audios)
        enforce_flags = kwargs.get("enforce_output_duration") or [False] * len(audios)
        fitted = []
        for audio, duration, enforce in zip(audios, durations, enforce_flags):
            if not enforce or duration is None:
                fitted.append(audio)
                continue
            target_samples = max(1, int(round(float(duration) * self.sampling_rate)))
            if len(audio) > target_samples:
                fitted.append(audio[:target_samples].copy())
            elif len(audio) < target_samples:
                fitted.append(
                    np.pad(
                        audio,
                        (0, target_samples - len(audio)),
                        mode="constant",
                    )
                )
            else:
                fitted.append(audio)
        return fitted


class FailingModel(FakeModel):
    def generate(self, **kwargs):
        raise ValueError("model failed")


class OutOfMemoryOnLargeBatchModel(FakeModel):
    def generate(self, **kwargs):
        texts = list(kwargs["text"])
        self.calls.append(texts)
        if len(texts) > 2:
            raise RuntimeError("CUDA out of memory")
        return [np.zeros(240, dtype=np.float32) for _ in texts]


class EstimatingModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.duration_estimator = object()
        self.estimate_calls = []

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        self.estimate_calls.append(
            {
                "text": text,
                "ref_text": ref_text,
                "num_ref_audio_tokens": num_ref_audio_tokens,
                "speed": speed,
            }
        )
        return 123 / speed


class FakeThread:
    def __init__(self, *, alive: bool):
        self.alive = alive
        self.join_timeouts = []

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def is_alive(self):
        return self.alive


class BatchSchedulerTests(unittest.IsolatedAsyncioTestCase):
    def _queued(
        self,
        request_id,
        cost,
        *,
        context=None,
        enqueued_at=None,
        mode="auto",
        **request_kwargs,
    ):
        loop = asyncio.get_running_loop()
        return _QueuedRequest(
            request=OmniVoiceBatchRequest(
                request_id=request_id,
                text=request_id,
                **request_kwargs,
            ),
            future=loop.create_future(),
            loop=loop,
            enqueued_at=time.monotonic() if enqueued_at is None else enqueued_at,
            cost_tokens=cost,
            context_tokens=cost if context is None else context,
            mode=mode,
        )

    async def test_batch_request_rejects_invalid_values(self):
        invalid_cases = (
            ({"request_id": "", "text": "hello"}, "request_id.*non-empty string"),
            ({"request_id": "r", "text": ""}, "text.*non-empty string"),
            ({"request_id": "r", "text": "hello", "language": 123}, "language.*string"),
            (
                {"request_id": "r", "text": "hello", "duration": "1.0"},
                "duration.*positive number",
            ),
            (
                {"request_id": "r", "text": "hello", "speed": False},
                "speed.*positive number",
            ),
            (
                {
                    "request_id": "r",
                    "text": "hello",
                    "enforce_output_duration": "false",
                },
                "enforce_output_duration.*bool",
            ),
            (
                {"request_id": "r", "text": "hello", "cost_tokens_hint": 1.5},
                "cost_tokens_hint.*positive integer",
            ),
            ({"request_id": "r", "text": "hello", "priority": "urgent"}, "priority"),
        )

        for kwargs, message in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    OmniVoiceBatchRequest(**kwargs)

    async def test_batch_request_normalizes_numpy_scalar_controls(self):
        request = OmniVoiceBatchRequest(
            request_id="r",
            text="hello",
            duration=np.float32(1.25),
            speed=np.float64(1.5),
            enforce_output_duration=np.bool_(True),
            cost_tokens_hint=np.int64(77),
            priority="high",
        )

        self.assertEqual(request.duration, 1.25)
        self.assertEqual(request.speed, 1.5)
        self.assertTrue(request.enforce_output_duration)
        self.assertEqual(request.cost_tokens_hint, 77)
        self.assertEqual(request.priority, "high")

    async def test_full_batch_dispatches_together(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(
            model,
            config=BatchSchedulerConfig(
                max_batch_size=4,
                max_wait_ms=10_000.0,
                partial_batch_floor=2,
            ),
        )
        await scheduler.start()
        try:
            tasks = [
                asyncio.create_task(
                    scheduler.submit(
                        OmniVoiceBatchRequest(request_id=str(i), text=f"text {i}")
                    )
                )
                for i in range(4)
            ]
            results = await asyncio.gather(*tasks)
        finally:
            await scheduler.stop()

        self.assertEqual([len(call) for call in model.calls], [4])
        self.assertEqual([result.batch_size for result in results], [4, 4, 4, 4])

    async def test_submit_many_dispatches_together(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(
            model,
            config=BatchSchedulerConfig(
                max_batch_size=4,
                max_wait_ms=10_000.0,
                partial_batch_floor=2,
            ),
        )
        await scheduler.start()
        try:
            results = await scheduler.submit_many(
                [
                    OmniVoiceBatchRequest(request_id=str(i), text=f"text {i}")
                    for i in range(4)
                ]
            )
        finally:
            await scheduler.stop()

        self.assertEqual([len(call) for call in model.calls], [4])
        self.assertEqual([result.batch_size for result in results], [4, 4, 4, 4])

    async def test_submit_many_queue_capacity_is_all_or_none(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(ready_queue_capacity=1),
        )

        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            await scheduler.submit_many(
                [
                    OmniVoiceBatchRequest(request_id="a", text="A"),
                    OmniVoiceBatchRequest(request_id="b", text="B"),
                ]
            )

        self.assertEqual(scheduler.snapshot().pending_total, 0)

    async def test_submit_many_cancellation_removes_pending_requests(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(max_wait_ms=10_000.0),
        )
        task = asyncio.create_task(
            scheduler.submit_many(
                [
                    OmniVoiceBatchRequest(request_id="a", text="A"),
                    OmniVoiceBatchRequest(request_id="b", text="B"),
                ]
            )
        )
        await asyncio.sleep(0)

        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.pending_total, 2)
        self.assertEqual(snapshot.pending_normal, 2)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(scheduler.snapshot().pending_total, 0)

    async def test_submit_and_control_reject_after_stop(self):
        scheduler = OmniVoiceBatchScheduler(FakeModel())
        await scheduler.stop()

        with self.assertRaisesRegex(RuntimeError, "scheduler stopped"):
            await scheduler.submit(OmniVoiceBatchRequest(request_id="x", text="x"))
        with self.assertRaisesRegex(RuntimeError, "scheduler stopped"):
            await scheduler.submit_many(
                [OmniVoiceBatchRequest(request_id="y", text="y")]
            )
        with self.assertRaisesRegex(RuntimeError, "scheduler stopped"):
            await scheduler._run_control(lambda: "ok")

    async def test_stop_preserves_thread_reference_when_join_times_out(self):
        scheduler = OmniVoiceBatchScheduler(FakeModel())
        thread = FakeThread(alive=True)
        scheduler._thread = thread

        await scheduler.stop()

        self.assertIs(scheduler._thread, thread)
        self.assertEqual(thread.join_timeouts, [10.0])
        with self.assertRaisesRegex(RuntimeError, "scheduler stopped"):
            await scheduler.submit(OmniVoiceBatchRequest(request_id="x", text="x"))

    async def test_start_replaces_stale_dead_thread_reference(self):
        scheduler = OmniVoiceBatchScheduler(FakeModel())
        scheduler._thread = FakeThread(alive=False)

        await scheduler.start()
        try:
            self.assertIsNotNone(scheduler._thread)
            self.assertTrue(scheduler._thread.is_alive())
        finally:
            await scheduler.stop()

    async def test_incompatible_modes_do_not_mix(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(
            model,
            config=BatchSchedulerConfig(
                max_batch_size=4,
                max_wait_ms=1.0,
                partial_batch_floor=1,
            ),
        )
        await scheduler.start()
        try:
            auto_task = asyncio.create_task(
                scheduler.submit(OmniVoiceBatchRequest(request_id="a", text="auto"))
            )
            design_task = asyncio.create_task(
                scheduler.submit(
                    OmniVoiceBatchRequest(
                        request_id="d",
                        text="design",
                        instruct="calm voice",
                    )
                )
            )
            results = await asyncio.gather(auto_task, design_task)
        finally:
            await scheduler.stop()

        self.assertEqual([len(call) for call in model.calls], [1, 1])
        self.assertEqual([result.batch_size for result in results], [1, 1])

    async def test_lookahead_dispatches_full_cost_bucket_before_young_outlier(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=3,
                max_wait_ms=10_000.0,
                max_cost_ratio=1.2,
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("long", 500),
                self._queued("short-1", 100),
                self._queued("short-2", 105),
                self._queued("short-3", 110),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "full")
        self.assertEqual(
            [item.request.request_id for item in batch],
            ["short-1", "short-2", "short-3"],
        )
        self.assertEqual(
            [item.request.request_id for item in scheduler._ready_normal],
            ["long"],
        )

    async def test_ready_normal_batch_runs_when_high_seed_is_not_ready(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=3,
                max_wait_ms=10_000.0,
                max_cost_ratio=1.2,
            ),
        )
        scheduler._ready_high = deque([self._queued("high-design", 100, mode="design")])
        scheduler._ready_normal = deque(
            [
                self._queued("normal-1", 100),
                self._queued("normal-2", 105),
                self._queued("normal-3", 110),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "full")
        self.assertEqual(
            [item.request.request_id for item in batch],
            ["normal-1", "normal-2", "normal-3"],
        )
        self.assertEqual(
            [item.request.request_id for item in scheduler._ready_high],
            ["high-design"],
        )

    async def test_ready_high_batch_still_preempts_normal_batch(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=3,
                max_wait_ms=10_000.0,
                max_cost_ratio=1.2,
            ),
        )
        scheduler._ready_high = deque(
            [
                self._queued("high-1", 100),
                self._queued("high-2", 105),
                self._queued("high-3", 110),
            ]
        )
        scheduler._ready_normal = deque(
            [
                self._queued("normal-1", 100, mode="design"),
                self._queued("normal-2", 105, mode="design"),
                self._queued("normal-3", 110, mode="design"),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "full")
        self.assertEqual(
            [item.request.request_id for item in batch],
            ["high-1", "high-2", "high-3"],
        )
        self.assertEqual(
            [item.request.request_id for item in scheduler._ready_normal],
            ["normal-1", "normal-2", "normal-3"],
        )

    async def test_next_wait_uses_earliest_ready_queue_head(self):
        now = time.monotonic()
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(max_wait_ms=1_000.0),
        )
        scheduler._ready_high = deque([self._queued("high", 100, enqueued_at=now)])
        scheduler._ready_normal = deque(
            [self._queued("normal", 100, enqueued_at=now - 0.9)]
        )

        wait_s = scheduler._next_wait_seconds_locked()

        self.assertLess(wait_s, 0.2)

    async def test_lookahead_prefers_tighter_full_cost_bucket(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=3,
                max_wait_ms=10_000.0,
                max_cost_ratio=2.1,
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("broad-1", 100),
                self._queued("broad-2", 190),
                self._queued("broad-3", 195),
                self._queued("tight-1", 300),
                self._queued("tight-2", 305),
                self._queued("tight-3", 310),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "full")
        self.assertEqual(
            [item.request.request_id for item in batch],
            ["tight-1", "tight-2", "tight-3"],
        )

    async def test_lookahead_uses_context_ratio_for_same_target_costs(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=3,
                max_wait_ms=10_000.0,
                max_cost_ratio=10.0,
                max_context_ratio=1.2,
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("long-context", 100, context=1000),
                self._queued("short-1", 100, context=110),
                self._queued("short-2", 100, context=112),
                self._queued("short-3", 100, context=115),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "full")
        self.assertEqual(
            [item.request.request_id for item in batch],
            ["short-1", "short-2", "short-3"],
        )

    async def test_lookahead_uses_total_context_cap(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=3,
                max_wait_ms=10_000.0,
                max_total_target_tokens=10_000,
                max_total_context_tokens=500,
                max_cost_ratio=10.0,
                max_context_ratio=10.0,
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("long-1", 100, context=400),
                self._queued("long-2", 100, context=400),
                self._queued("long-3", 100, context=400),
                self._queued("short-1", 100, context=100),
                self._queued("short-2", 100, context=100),
                self._queued("short-3", 100, context=100),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "full")
        self.assertEqual(
            [item.request.request_id for item in batch],
            ["short-1", "short-2", "short-3"],
        )

    async def test_lookahead_tries_context_first_packing_for_same_seed(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=4,
                max_wait_ms=10_000.0,
                max_total_target_tokens=10_000,
                max_total_context_tokens=1000,
                max_cost_ratio=2.0,
                max_context_ratio=10.0,
                max_context_padding_ratio=0.0,
                max_seed_lookahead=1,
                candidate_pack_policy="target_context",
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("seed", 100, context=100),
                self._queued("target-near-context-heavy", 101, context=800),
                self._queued("fit-1", 130, context=100),
                self._queued("fit-2", 131, context=100),
                self._queued("fit-3", 132, context=100),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "full")
        self.assertEqual(
            [item.request.request_id for item in batch],
            ["seed", "fit-1", "fit-2", "fit-3"],
        )
        self.assertEqual(
            [item.request.request_id for item in scheduler._ready_normal],
            ["target-near-context-heavy"],
        )

    async def test_target_only_pack_policy_preserves_original_target_ordering(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=4,
                max_wait_ms=10_000.0,
                max_total_target_tokens=10_000,
                max_total_context_tokens=1000,
                max_cost_ratio=2.0,
                max_context_ratio=10.0,
                max_context_padding_ratio=0.0,
                max_seed_lookahead=1,
                candidate_pack_policy="target",
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("seed", 100, context=100),
                self._queued("target-near-context-heavy", 101, context=800),
                self._queued("fit-1", 130, context=100),
                self._queued("fit-2", 131, context=100),
                self._queued("fit-3", 132, context=100),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(batch, [])
        self.assertEqual(reason, "wait")

    async def test_invalid_candidate_pack_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "candidate_pack_policy"):
            OmniVoiceBatchScheduler(
                FakeModel(),
                config=BatchSchedulerConfig(candidate_pack_policy="unknown"),
            )

    async def test_batch_scheduler_config_rejects_invalid_values(self):
        invalid_cases = (
            ("max_batch_size", 0, "positive integer"),
            ("max_batch_size", "8", "positive integer"),
            ("prompt_cache_entries", -1, "non-negative integer"),
            ("max_wait_ms", "120.0", "non-negative number"),
            ("max_cost_ratio", 0.9, ">= 1.0"),
            ("partial_lookahead_max_wait_multiplier", 0.0, "positive number"),
            ("lookahead_for_full_batch", "true", "must be bool"),
        )

        for field_name, value, message in invalid_cases:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, f"{field_name}.*{message}"):
                    BatchSchedulerConfig(**{field_name: value})

    async def test_batch_scheduler_config_allows_explicit_disabled_limits(self):
        config = BatchSchedulerConfig(
            max_wait_ms=0.0,
            max_context_padding_ratio=0.0,
            prompt_cache_entries=0,
        )

        self.assertEqual(config.max_wait_ms, 0.0)
        self.assertEqual(config.max_context_padding_ratio, 0.0)
        self.assertEqual(config.prompt_cache_entries, 0)

    async def test_context_padding_ratio_limits_hidden_padding_work(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=8,
                max_wait_ms=10_000.0,
                max_total_target_tokens=10_000,
                max_total_context_tokens=10_000,
                max_cost_ratio=10.0,
                max_context_ratio=100.0,
                max_context_padding_ratio=2.0,
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("long-1", 100, context=240),
                self._queued("short-1", 100, context=60),
                self._queued("short-2", 100, context=60),
                self._queued("short-3", 100, context=60),
                self._queued("short-4", 100, context=60),
                self._queued("long-2", 100, context=240),
                self._queued("short-5", 100, context=60),
                self._queued("short-6", 100, context=60),
            ]
        )

        batch = scheduler._select_batch_locked(scheduler._ready_normal[0])

        request_ids = [item.request.request_id for item in batch]
        self.assertEqual(len(batch), 6)
        self.assertIn("long-1", request_ids)
        self.assertIn("long-2", request_ids)
        self.assertLessEqual(
            len(batch) * max(item.context_tokens for item in batch)
            / sum(item.context_tokens for item in batch),
            2.0,
        )

        no_padding_limit = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=8,
                max_wait_ms=10_000.0,
                max_total_target_tokens=10_000,
                max_total_context_tokens=10_000,
                max_cost_ratio=10.0,
                max_context_ratio=100.0,
                max_context_padding_ratio=0.0,
            ),
        )
        no_padding_limit._ready_normal = deque(scheduler._ready_normal)
        unlimited_batch = no_padding_limit._select_batch_locked(
            no_padding_limit._ready_normal[0]
        )

        self.assertEqual(len(unlimited_batch), 8)
        self.assertGreater(
            len(unlimited_batch) * max(item.context_tokens for item in unlimited_batch)
            / sum(item.context_tokens for item in unlimited_batch),
            2.0,
        )

    async def test_partial_lookahead_dispatches_efficient_timed_out_bucket(self):
        now = time.monotonic()
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=12,
                max_wait_ms=1_000.0,
                partial_batch_floor=2,
                max_total_target_tokens=10_000,
                max_total_context_tokens=10_000,
                max_cost_ratio=10.0,
                max_context_ratio=100.0,
                max_context_padding_ratio=2.0,
                lookahead_for_partial_batch=True,
                partial_lookahead_max_wait_multiplier=2.0,
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("long-1", 100, context=240, enqueued_at=now - 1.1),
                *[
                    self._queued(
                        f"short-{index}",
                        100,
                        context=60,
                        enqueued_at=now - 1.0 + index * 0.001,
                    )
                    for index in range(1, 11)
                ],
                self._queued("long-2", 100, context=240, enqueued_at=now - 0.9),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "timeout_lookahead")
        self.assertEqual(len(batch), 10)
        self.assertTrue(
            all(item.request.request_id.startswith("short-") for item in batch)
        )
        self.assertEqual(
            [item.request.request_id for item in scheduler._ready_normal],
            ["long-1", "long-2"],
        )

    async def test_partial_lookahead_respects_hard_fifo_wait(self):
        now = time.monotonic()
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=12,
                max_wait_ms=1_000.0,
                partial_batch_floor=2,
                max_total_target_tokens=10_000,
                max_total_context_tokens=10_000,
                max_cost_ratio=10.0,
                max_context_ratio=100.0,
                max_context_padding_ratio=2.0,
                lookahead_for_partial_batch=True,
                partial_lookahead_max_wait_multiplier=2.0,
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("long-1", 100, context=240, enqueued_at=now - 2.1),
                *[
                    self._queued(
                        f"short-{index}",
                        100,
                        context=60,
                        enqueued_at=now - 1.0 + index * 0.001,
                    )
                    for index in range(1, 11)
                ],
                self._queued("long-2", 100, context=240, enqueued_at=now - 0.9),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "timeout")
        self.assertEqual(batch[0].request.request_id, "long-1")
        self.assertIn("long-2", [item.request.request_id for item in batch])

    async def test_lookahead_does_not_bypass_timed_out_fifo_seed(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=3,
                max_wait_ms=1.0,
                max_cost_ratio=1.2,
            ),
        )
        old = time.monotonic() - 1.0
        scheduler._ready_normal = deque(
            [
                self._queued("long", 500, enqueued_at=old),
                self._queued("short-1", 100),
                self._queued("short-2", 105),
                self._queued("short-3", 110),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "aging")
        self.assertEqual([item.request.request_id for item in batch], ["long"])

    async def test_oversized_fifo_seed_can_run_alone_after_aging(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=3,
                max_wait_ms=1.0,
                max_total_target_tokens=200,
                max_cost_ratio=1.2,
            ),
        )
        scheduler._ready_normal = deque(
            [
                self._queued("oversized", 500, enqueued_at=time.monotonic() - 1.0),
                self._queued("short", 100),
            ]
        )

        batch, reason = scheduler._try_pop_batch_locked()

        self.assertEqual(reason, "aging")
        self.assertEqual([item.request.request_id for item in batch], ["oversized"])

    async def test_reset_metrics_clears_warmup_counters_only(self):
        scheduler = OmniVoiceBatchScheduler(FakeModel())
        scheduler._total_batches = 2
        scheduler._total_requests = 5
        scheduler._last_dispatch_reason = "full"
        scheduler._running_batch_size = 3
        scheduler._running_batch_reason = "full"
        scheduler._running_batch_started_at = time.monotonic()
        scheduler._running_batch_cost_tokens = 123
        scheduler._running_batch_context_tokens = 456
        scheduler._last_batch_size = 4
        scheduler._last_batch_infer_s = 0.25
        scheduler._last_batch_cost_tokens = 400
        scheduler._last_batch_max_cost_tokens = 100
        scheduler._last_batch_context_tokens = 900
        scheduler._last_batch_max_context_tokens = 300
        scheduler._total_infer_s = 0.5
        scheduler._total_generated_audio_s = 10.0
        scheduler._failed_batches = 1
        scheduler._split_retry_batches = 2
        scheduler._last_batch_split_retries = 1
        scheduler._last_batch_model_calls = 3
        scheduler._last_batch_max_execution_size = 2
        scheduler._adaptive_batch_caps["auto"] = 2
        scheduler._adaptive_cap_success_streaks["auto"] = 1
        scheduler._prompt_cache.hits = 3
        scheduler._prompt_cache.misses = 4
        scheduler._prompt_cache._entries[("ref",)] = object()

        scheduler.reset_metrics()

        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.total_batches, 0)
        self.assertEqual(snapshot.total_requests, 0)
        self.assertEqual(snapshot.avg_batch_size, 0.0)
        self.assertEqual(snapshot.prompt_cache_hits, 0)
        self.assertEqual(snapshot.prompt_cache_misses, 0)
        self.assertEqual(snapshot.last_dispatch_reason, "idle")
        self.assertEqual(snapshot.running_batch_size, 0)
        self.assertEqual(snapshot.running_batch_reason, "idle")
        self.assertEqual(snapshot.running_batch_cost_tokens, 0)
        self.assertEqual(snapshot.running_batch_context_tokens, 0)
        self.assertEqual(snapshot.last_batch_size, 0)
        self.assertEqual(snapshot.last_batch_infer_s, 0.0)
        self.assertEqual(snapshot.last_batch_cost_tokens, 0)
        self.assertEqual(snapshot.last_batch_max_cost_tokens, 0)
        self.assertEqual(snapshot.last_batch_context_tokens, 0)
        self.assertEqual(snapshot.last_batch_max_context_tokens, 0)
        self.assertEqual(snapshot.last_batch_context_padding_ratio, 0.0)
        self.assertEqual(snapshot.total_infer_s, 0.0)
        self.assertEqual(snapshot.total_generated_audio_s, 0.0)
        self.assertEqual(snapshot.infer_rtf, 0.0)
        self.assertEqual(snapshot.failed_batches, 0)
        self.assertEqual(snapshot.split_retry_batches, 0)
        self.assertEqual(snapshot.last_batch_split_retries, 0)
        self.assertEqual(snapshot.last_batch_model_calls, 0)
        self.assertEqual(snapshot.last_batch_max_execution_size, 0)
        self.assertEqual(snapshot.adaptive_batch_caps, {"auto": 2})
        self.assertEqual(snapshot.adaptive_cap_success_streaks, {"auto": 1})
        self.assertEqual(len(scheduler._prompt_cache._entries), 1)

    async def test_snapshot_reports_runtime_and_completed_batch_metrics(self):
        model = ObservedModel()
        scheduler = OmniVoiceBatchScheduler(model)
        model.scheduler = scheduler
        batch = [self._queued("a", 25), self._queued("b", 30)]

        scheduler._execute_batch(batch, "full")
        await asyncio.sleep(0)

        inside = model.snapshot_inside_generate
        self.assertEqual(inside.running_batch_size, 2)
        self.assertEqual(inside.running_batch_reason, "full")
        self.assertEqual(inside.running_batch_cost_tokens, 55)
        self.assertEqual(inside.running_batch_context_tokens, 55)
        self.assertGreaterEqual(inside.running_batch_elapsed_ms, 0.0)

        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.running_batch_size, 0)
        self.assertEqual(snapshot.last_batch_size, 2)
        self.assertEqual(snapshot.last_batch_cost_tokens, 55)
        self.assertEqual(snapshot.last_batch_max_cost_tokens, 30)
        self.assertEqual(snapshot.last_batch_context_tokens, 55)
        self.assertEqual(snapshot.last_batch_max_context_tokens, 30)
        self.assertAlmostEqual(
            snapshot.last_batch_context_padding_ratio,
            2 * 30 / 55,
        )
        self.assertAlmostEqual(
            batch[0].future.result().batch_context_padding_ratio,
            2 * 30 / 55,
        )
        self.assertAlmostEqual(snapshot.total_generated_audio_s, 0.02)
        self.assertGreater(snapshot.total_infer_s, 0.0)
        self.assertGreater(snapshot.infer_rtf, 0.0)
        self.assertEqual(snapshot.failed_batches, 0)

    async def test_completed_batch_propagates_generation_profile(self):
        scheduler = OmniVoiceBatchScheduler(ProfiledModel())
        queued = [self._queued("a", 25), self._queued("b", 25)]

        scheduler._execute_batch(queued, "full")
        await asyncio.sleep(0)

        result = queued[0].future.result()
        self.assertEqual(result.generation_profile["metadata"]["batch_size"], 2)
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.last_batch_profile["timings_s"]["total_s"], 1.0)
        self.assertEqual(
            snapshot.last_batch_profile["metadata"]["online_enforce_output_duration"],
            [False, False],
        )

    async def test_per_request_output_duration_enforcement_is_delegated_to_model(self):
        model = FixedLengthModel(length=5)
        scheduler = OmniVoiceBatchScheduler(
            model,
            sample_rate=1000,
            generation_kwargs={"enforce_output_duration": True},
        )
        queued = [
            self._queued("default", 25, duration=0.008),
            self._queued("disabled", 25, duration=0.008, enforce_output_duration=False),
            self._queued("enabled", 25, duration=0.003, enforce_output_duration=True),
        ]

        scheduler._execute_batch(queued, "full")
        await asyncio.sleep(0)

        self.assertEqual([len(item.future.result().audio) for item in queued], [8, 5, 3])
        self.assertEqual(model.kwargs["enforce_output_duration"], [True, False, True])

    async def test_invalid_output_duration_enforcement_flag_fails_at_request_boundary(self):
        with self.assertRaisesRegex(ValueError, "enforce_output_duration.*bool"):
            self._queued(
                "bad",
                25,
                enforce_output_duration="false",
            )

    async def test_invalid_default_output_duration_enforcement_fails_fast(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(
            model,
            generation_kwargs={"enforce_output_duration": "false"},
        )
        queued = [self._queued("bad-default", 25)]

        scheduler._execute_batch(queued, "full")
        await asyncio.sleep(0)

        self.assertIsInstance(queued[0].future.exception(), ValueError)
        self.assertEqual(model.calls, [])

    async def test_invalid_preprocess_prompt_default_fails_before_prompt_cache(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(
            model,
            generation_kwargs={"preprocess_prompt": "false"},
        )
        queued = [
            self._queued(
                "bad-preprocess",
                25,
                mode="clone",
                ref_audio="/tmp/ref.wav",
                ref_text="reference",
            )
        ]

        scheduler._execute_batch(queued, "full")
        await asyncio.sleep(0)

        self.assertIsInstance(queued[0].future.exception(), ValueError)
        self.assertEqual(model.prompt_calls, [])
        self.assertEqual(model.calls, [])

    async def test_failed_batch_updates_failure_metrics_and_clears_running_state(self):
        scheduler = OmniVoiceBatchScheduler(FailingModel())
        queued = self._queued("a", 25)

        scheduler._execute_batch([queued], "full")
        await asyncio.sleep(0)

        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.failed_batches, 1)
        self.assertEqual(snapshot.last_dispatch_reason, "failed")
        self.assertEqual(snapshot.running_batch_size, 0)
        self.assertEqual(snapshot.running_batch_reason, "idle")
        self.assertEqual(snapshot.total_batches, 0)
        self.assertTrue(queued.future.done())
        with self.assertRaises(ValueError):
            queued.future.result()

    async def test_memory_error_batch_is_split_and_retried(self):
        model = OutOfMemoryOnLargeBatchModel()
        scheduler = OmniVoiceBatchScheduler(model)
        batch = [
            self._queued("a", 25),
            self._queued("b", 25),
            self._queued("c", 25),
            self._queued("d", 25),
        ]

        scheduler._execute_batch(batch, "full")
        await asyncio.sleep(0)

        self.assertEqual(
            model.calls,
            [["a", "b", "c", "d"], ["a", "b"], ["c", "d"]],
        )
        for queued in batch:
            self.assertEqual(queued.future.result().batch_size, 4)
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.total_batches, 1)
        self.assertEqual(snapshot.total_requests, 4)
        self.assertEqual(snapshot.failed_batches, 0)
        self.assertEqual(snapshot.split_retry_batches, 1)
        self.assertEqual(snapshot.last_batch_split_retries, 1)
        self.assertEqual(snapshot.last_batch_model_calls, 3)
        self.assertEqual(snapshot.last_batch_max_execution_size, 2)
        self.assertEqual(snapshot.adaptive_batch_caps, {"auto": 2})

        scheduler._ready_normal = deque(
            [
                self._queued("e", 25),
                self._queued("f", 25),
                self._queued("g", 25),
                self._queued("h", 25),
            ]
        )
        next_batch, reason = scheduler._try_pop_batch_locked()
        self.assertEqual(reason, "full")
        self.assertEqual([item.request.request_id for item in next_batch], ["e", "f"])

    async def test_adaptive_memory_batch_cap_recovers_after_successes(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=4,
                adaptive_memory_cap_recovery_successes=2,
            ),
        )
        scheduler._adaptive_batch_caps["auto"] = 2

        scheduler._execute_batch([self._queued("a", 25), self._queued("b", 25)], "full")
        await asyncio.sleep(0)
        self.assertEqual(scheduler.snapshot().adaptive_batch_caps, {"auto": 2})
        self.assertEqual(scheduler.snapshot().adaptive_cap_success_streaks, {"auto": 1})

        scheduler._execute_batch([self._queued("c", 25), self._queued("d", 25)], "full")
        await asyncio.sleep(0)
        self.assertEqual(scheduler.snapshot().adaptive_batch_caps, {"auto": 3})
        self.assertEqual(scheduler.snapshot().adaptive_cap_success_streaks, {"auto": 0})

    async def test_adaptive_memory_batch_cap_can_be_disabled(self):
        model = OutOfMemoryOnLargeBatchModel()
        scheduler = OmniVoiceBatchScheduler(
            model,
            config=BatchSchedulerConfig(
                adaptive_memory_batch_cap=False,
            ),
        )
        batch = [
            self._queued("a", 25),
            self._queued("b", 25),
            self._queued("c", 25),
            self._queued("d", 25),
        ]

        scheduler._execute_batch(batch, "full")
        await asyncio.sleep(0)

        self.assertEqual(scheduler.snapshot().adaptive_batch_caps, {})

    async def test_memory_error_split_retry_can_be_disabled(self):
        model = OutOfMemoryOnLargeBatchModel()
        scheduler = OmniVoiceBatchScheduler(
            model,
            config=BatchSchedulerConfig(split_retry_on_memory_error=False),
        )
        batch = [self._queued("a", 25), self._queued("b", 25), self._queued("c", 25)]

        scheduler._execute_batch(batch, "full")
        await asyncio.sleep(0)

        self.assertEqual(model.calls, [["a", "b", "c"]])
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.failed_batches, 1)
        self.assertEqual(snapshot.split_retry_batches, 0)
        for queued in batch:
            with self.assertRaises(RuntimeError):
                queued.future.result()

    async def test_cancelled_pending_request_is_removed_from_queue(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(max_wait_ms=10_000.0),
        )
        task = asyncio.create_task(
            scheduler.submit(OmniVoiceBatchRequest(request_id="cancel", text="hello"))
        )
        await asyncio.sleep(0)

        self.assertEqual(scheduler.snapshot().pending_normal, 1)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(scheduler.snapshot().pending_normal, 0)

    async def test_safe_future_setters_ignore_completed_futures(self):
        scheduler = OmniVoiceBatchScheduler(FakeModel())
        loop = asyncio.get_running_loop()

        result_future = loop.create_future()
        result_future.cancel()
        scheduler._safe_set_result(
            result_future,
            OmniVoiceBatchResult(
                request_id="done",
                audio=np.zeros(1, dtype=np.float32),
                sample_rate=24000,
                batch_size=1,
                queue_wait_ms=0.0,
                batch_infer_s=0.0,
                batch_reason="full",
            ),
        )
        self.assertTrue(result_future.cancelled())

        exc_future = loop.create_future()
        exc_future.set_result("already done")
        scheduler._safe_set_exception(exc_future, RuntimeError("late failure"))
        self.assertEqual(exc_future.result(), "already done")

    async def test_execute_batch_skips_all_cancelled_requests(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(model)
        queued = self._queued("cancelled", 100)
        queued.future.cancel()

        scheduler._execute_batch([queued], "full")

        self.assertEqual(model.calls, [])
        self.assertEqual(scheduler.snapshot().total_batches, 0)
        self.assertEqual(scheduler.snapshot().total_requests, 0)

    async def test_execute_batch_filters_cancelled_requests(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(model)
        cancelled = self._queued("cancelled", 100)
        active = self._queued("active", 100)
        cancelled.future.cancel()

        scheduler._execute_batch([cancelled, active], "full")
        await asyncio.sleep(0)

        self.assertEqual(model.calls, [["active"]])
        self.assertTrue(cancelled.future.cancelled())
        self.assertEqual(active.future.result().request_id, "active")
        self.assertEqual(active.future.result().batch_size, 1)
        self.assertEqual(scheduler.snapshot().total_batches, 1)
        self.assertEqual(scheduler.snapshot().total_requests, 1)

    async def test_create_voice_clone_prompt_runs_on_scheduler_thread(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(model)
        await scheduler.start()
        try:
            prompt = await scheduler.create_voice_clone_prompt(
                ref_audio="/tmp/ref.wav",
                ref_text="reference",
                preprocess_prompt=False,
                cache_key=("voice", "a"),
            )
        finally:
            await scheduler.stop()

        self.assertEqual(prompt.ref_text, "reference")
        self.assertEqual(len(model.prompt_calls), 1)
        self.assertEqual(
            model.prompt_calls[0]["thread"],
            "omnivoice-batch-scheduler",
        )
        self.assertFalse(model.prompt_calls[0]["preprocess_prompt"])

    async def test_create_voice_clone_prompt_rejects_non_bool_preprocess_prompt(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(model)

        with self.assertRaisesRegex(ValueError, "preprocess_prompt must be bool"):
            await scheduler.create_voice_clone_prompt(
                ref_audio="/tmp/ref.wav",
                ref_text="reference",
                preprocess_prompt="false",
            )

        self.assertEqual(model.prompt_calls, [])

    async def test_create_voice_clone_prompt_uses_prompt_cache(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(model)
        await scheduler.start()
        try:
            first = await scheduler.create_voice_clone_prompt(
                ref_audio="/tmp/ref.wav",
                ref_text="reference",
                cache_key=("voice", "cached"),
            )
            second = await scheduler.create_voice_clone_prompt(
                ref_audio="/tmp/ref.wav",
                ref_text="reference",
                cache_key=("voice", "cached"),
            )
        finally:
            await scheduler.stop()

        self.assertIs(first, second)
        self.assertEqual(len(model.prompt_calls), 1)
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.prompt_cache_misses, 1)
        self.assertEqual(snapshot.prompt_cache_hits, 1)

    async def test_tuple_ref_audio_uses_prompt_cache_without_custom_key(self):
        model = FakeModel()
        scheduler = OmniVoiceBatchScheduler(model)
        ref_audio = (np.zeros(24000, dtype=np.float32), 24000)
        await scheduler.start()
        try:
            first = await scheduler.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text="reference",
            )
            second = await scheduler.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text="reference",
            )
        finally:
            await scheduler.stop()

        self.assertIs(first, second)
        self.assertEqual(len(model.prompt_calls), 1)
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.prompt_cache_misses, 1)
        self.assertEqual(snapshot.prompt_cache_hits, 1)

    async def test_ready_generation_batch_runs_before_control_request(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=2,
                max_wait_ms=10_000.0,
            ),
        )
        loop = asyncio.get_running_loop()
        control = _ControlRequest(
            func=lambda: "ok",
            future=loop.create_future(),
            loop=loop,
        )
        scheduler._control.append(control)
        scheduler._ready_normal = deque([self._queued("a", 25), self._queued("b", 25)])

        next_control, batch, reason = scheduler._pop_next_work_locked()

        self.assertIsNone(next_control)
        self.assertEqual(reason, "full")
        self.assertEqual([item.request.request_id for item in batch], ["a", "b"])
        self.assertEqual(list(scheduler._control), [control])

    async def test_control_request_has_fairness_cap_under_ready_batches(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=2,
                max_wait_ms=10_000.0,
                max_generation_batches_before_control=2,
            ),
        )
        loop = asyncio.get_running_loop()
        control = _ControlRequest(
            func=lambda: "ok",
            future=loop.create_future(),
            loop=loop,
        )
        scheduler._control.append(control)
        scheduler._ready_normal = deque([self._queued("a", 25), self._queued("b", 25)])
        scheduler._generation_batches_since_control = 2

        next_control, batch, reason = scheduler._pop_next_work_locked()

        self.assertIs(next_control, control)
        self.assertEqual(batch, [])
        self.assertEqual(reason, "control")
        self.assertEqual(scheduler._generation_batches_since_control, 0)
        self.assertEqual(
            [item.request.request_id for item in scheduler._ready_normal],
            ["a", "b"],
        )

    async def test_control_request_runs_when_generation_batch_is_not_ready(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(
                max_batch_size=2,
                max_wait_ms=10_000.0,
            ),
        )
        loop = asyncio.get_running_loop()
        control = _ControlRequest(
            func=lambda: "ok",
            future=loop.create_future(),
            loop=loop,
        )
        scheduler._control.append(control)
        scheduler._ready_normal = deque([self._queued("a", 25)])

        next_control, batch, reason = scheduler._pop_next_work_locked()

        self.assertIs(next_control, control)
        self.assertEqual(batch, [])
        self.assertEqual(reason, "control")
        self.assertEqual(len(scheduler._ready_normal), 1)

    async def test_control_queue_capacity_is_enforced(self):
        scheduler = OmniVoiceBatchScheduler(
            FakeModel(),
            config=BatchSchedulerConfig(control_queue_capacity=1),
        )
        loop = asyncio.get_running_loop()
        scheduler._control.append(
            _ControlRequest(
                func=lambda: "existing",
                future=loop.create_future(),
                loop=loop,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "control queue is full"):
            await scheduler._run_control(lambda: "new")

    async def test_cost_tokens_hint_overrides_scheduler_estimate(self):
        scheduler = OmniVoiceBatchScheduler(EstimatingModel())

        cost = scheduler._estimate_cost_tokens(
            OmniVoiceBatchRequest(
                request_id="hinted",
                text="hello",
                cost_tokens_hint=77,
            )
        )

        self.assertEqual(cost, 77)

    async def test_model_duration_estimator_is_used_for_cost_bucket(self):
        model = EstimatingModel()
        scheduler = OmniVoiceBatchScheduler(model)
        prompt = VoiceClonePrompt(
            ref_audio_tokens=torch.zeros((8, 37), dtype=torch.long),
            ref_text="reference words",
            ref_rms=0.1,
        )

        cost = scheduler._estimate_cost_tokens(
            OmniVoiceBatchRequest(
                request_id="estimated",
                text="target words",
                voice_clone_prompt=prompt,
                speed=2.0,
            )
        )

        self.assertEqual(cost, 61)
        self.assertEqual(
            model.estimate_calls,
            [
                {
                    "text": "target words",
                    "ref_text": "reference words",
                    "num_ref_audio_tokens": 37,
                    "speed": 2.0,
                }
            ],
        )

    async def test_tuple_ref_audio_uses_cheap_token_count_for_estimate(self):
        model = EstimatingModel()
        scheduler = OmniVoiceBatchScheduler(model)

        scheduler._estimate_cost_tokens(
            OmniVoiceBatchRequest(
                request_id="tuple-ref",
                text="target words",
                ref_audio=(np.zeros(48_000, dtype=np.float32), 24_000),
                ref_text="reference words",
            )
        )

        self.assertEqual(model.estimate_calls[0]["num_ref_audio_tokens"], 50)

    async def test_path_ref_audio_uses_file_duration_for_estimate(self):
        model = EstimatingModel()
        scheduler = OmniVoiceBatchScheduler(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_audio = f"{tmpdir}/ref.wav"
            sf.write(ref_audio, np.zeros(48_000, dtype=np.float32), 24_000)

            context_tokens = scheduler._estimate_context_tokens(
                OmniVoiceBatchRequest(
                    request_id="path-ref",
                    text="target words",
                    ref_audio=ref_audio,
                    ref_text="reference words",
                ),
                cost_tokens=30,
            )
            scheduler._estimate_cost_tokens(
                OmniVoiceBatchRequest(
                    request_id="path-ref",
                    text="target words",
                    ref_audio=ref_audio,
                    ref_text="reference words",
                )
            )

        self.assertEqual(model.estimate_calls[0]["num_ref_audio_tokens"], 50)
        self.assertEqual(context_tokens, 30 + len("target words") // 2 + 50 + 16 + 7)

    async def test_path_ref_audio_token_estimate_is_stat_cached(self):
        scheduler = OmniVoiceBatchScheduler(FakeModel())
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_audio = f"{tmpdir}/ref.wav"
            sf.write(ref_audio, np.zeros(24_000, dtype=np.float32), 24_000)
            original_info = sf.info

            with mock.patch(
                "omnivoice.serving.batcher.sf.info",
                wraps=original_info,
            ) as info:
                self.assertEqual(scheduler._cheap_ref_audio_token_count(ref_audio), 25)
                self.assertEqual(scheduler._cheap_ref_audio_token_count(ref_audio), 25)
                sf.write(ref_audio, np.zeros(48_000, dtype=np.float32), 24_000)
                self.assertEqual(scheduler._cheap_ref_audio_token_count(ref_audio), 50)

        self.assertEqual(info.call_count, 2)

    async def test_model_duration_estimator_can_be_disabled(self):
        model = EstimatingModel()
        scheduler = OmniVoiceBatchScheduler(
            model,
            config=BatchSchedulerConfig(use_model_duration_estimator=False),
        )

        cost = scheduler._estimate_cost_tokens(
            OmniVoiceBatchRequest(request_id="fallback", text="hello")
        )

        self.assertEqual(cost, 16)
        self.assertEqual(model.estimate_calls, [])


if __name__ == "__main__":
    unittest.main()
