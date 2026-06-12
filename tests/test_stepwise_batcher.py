import asyncio
import time
import unittest
from collections import deque

import torch

from omnivoice.models.generation import OmniVoiceGenerationConfig
from omnivoice.serving.batcher import OmniVoiceBatchRequest
from omnivoice.serving.stepwise import (
    StepwiseGenerationState,
    _pack_active_states,
    run_generation_step,
)
from omnivoice.serving.stepwise_batcher import (
    StepwiseOmniVoiceScheduler,
    StepwiseSchedulerConfig,
    _ControlRequest,
    _RunningRequest,
    _WaitingRequest,
)


class _State:
    def __init__(self, target_len, cond_len=None):
        self.target_len = target_len
        self._cond_len = cond_len or target_len

    @property
    def cond_len(self):
        return self._cond_len


class _Config:
    num_audio_codebook = 2
    audio_mask_id = 99


class _Model:
    config = _Config()
    device = torch.device("cpu")


class _VectorModel:
    class Config:
        num_audio_codebook = 1
        audio_mask_id = 99

    config = Config()
    device = torch.device("cpu")

    def __init__(self):
        self.predict_calls = 0

    def _forward_audio_logits_for_slices(
        self,
        *,
        input_ids,
        audio_mask,
        attention_mask,
        target_slices,
        target_len_pad=None,
    ):
        max_target_len = target_len_pad or max(end - start for start, end in target_slices)
        return torch.zeros(
            (input_ids.size(0), 1, max_target_len, 4),
            dtype=torch.float32,
        )

    def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
        self.predict_calls += 1
        batch_size, codebooks, target_len, _ = c_logits.shape
        pred_tokens = torch.ones(
            (batch_size, codebooks, target_len),
            dtype=torch.long,
        )
        scores = torch.arange(target_len, dtype=torch.float32).view(1, 1, target_len)
        return pred_tokens, scores.expand(batch_size, codebooks, target_len).clone()


class _PreprocessCountingModel:
    def __init__(self):
        self.preprocess_calls = 0

    def _preprocess_all(self, **kwargs):
        self.preprocess_calls += 1
        raise AssertionError("_preprocess_all should not be called")


class StepwiseAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()

    def _waiting(self, request_id, cost, *, context=None, **request_kwargs):
        return _WaitingRequest(
            request=OmniVoiceBatchRequest(
                request_id=request_id,
                text=request_id,
                **request_kwargs,
            ),
            future=self.loop.create_future(),
            loop=self.loop,
            enqueued_at=time.monotonic(),
            estimated_target_tokens=cost,
            estimated_context_tokens=cost if context is None else context,
        )

    def _old_waiting(self, request_id, cost):
        waiting = self._waiting(request_id, cost)
        waiting.enqueued_at = time.monotonic() - 1.0
        return waiting

    def test_waiting_admission_respects_cost_ratio_and_total_budget(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(
                max_running_requests=4,
                max_total_target_tokens=260,
                max_cost_ratio=1.5,
            ),
        )
        scheduler._waiting = deque(
            [
                self._waiting("seed", 100),
                self._waiting("near", 130),
                self._waiting("too-long", 260),
            ]
        )

        admitted = scheduler._pop_admissible_locked(capacity=4, running=[])

        self.assertEqual([item.request.request_id for item in admitted], ["seed", "near"])
        self.assertEqual(
            [item.request.request_id for item in scheduler._waiting],
            ["too-long"],
        )

    def test_running_batch_seed_limits_new_admissions(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(
                max_running_requests=4,
                max_total_target_tokens=360,
                max_cost_ratio=1.4,
            ),
        )
        scheduler._waiting = deque(
            [
                self._waiting("near", 130),
                self._waiting("ratio-outlier", 180),
                self._waiting("budget-outlier", 230),
            ]
        )
        running = [
            _RunningRequest(
                waiting=self._waiting("running", 100),
                state=_State(100),
                ref_rms=None,
                started_at=0.0,
            )
        ]

        admitted = scheduler._pop_admissible_locked(capacity=3, running=running)

        self.assertEqual([item.request.request_id for item in admitted], ["near"])
        self.assertEqual(
            [item.request.request_id for item in scheduler._waiting],
            ["ratio-outlier", "budget-outlier"],
        )

    def test_waiting_admission_respects_context_ratio_and_total_budget(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(
                max_running_requests=4,
                max_total_target_tokens=1000,
                max_total_context_tokens=360,
                max_cost_ratio=10.0,
                max_context_ratio=1.5,
            ),
        )
        scheduler._waiting = deque(
            [
                self._waiting("seed", 100, context=100),
                self._waiting("near-context", 100, context=140),
                self._waiting("ratio-outlier", 100, context=220),
                self._waiting("budget-outlier", 100, context=150),
            ]
        )

        admitted = scheduler._pop_admissible_locked(capacity=4, running=[])

        self.assertEqual(
            [item.request.request_id for item in admitted],
            ["seed", "near-context"],
        )
        self.assertEqual(
            [item.request.request_id for item in scheduler._waiting],
            ["ratio-outlier", "budget-outlier"],
        )

    def test_running_admission_respects_context_padding_ratio(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(
                max_running_requests=8,
                max_total_target_tokens=10_000,
                max_total_context_tokens=10_000,
                max_cost_ratio=10.0,
                max_context_ratio=100.0,
                max_context_padding_ratio=2.0,
            ),
        )
        scheduler._waiting = deque(
            [
                self._waiting("short-1", 100, context=60),
                self._waiting("short-2", 100, context=60),
                self._waiting("short-3", 100, context=60),
                self._waiting("long-2", 100, context=240),
            ]
        )
        running = [
            _RunningRequest(
                waiting=self._waiting("long-1", 100, context=240),
                state=_State(100, cond_len=240),
                ref_rms=None,
                started_at=0.0,
            )
        ]

        admitted = scheduler._pop_admissible_locked(capacity=7, running=running)

        self.assertEqual(
            [item.request.request_id for item in admitted],
            ["short-1", "short-2", "long-2"],
        )
        self.assertEqual(
            [item.request.request_id for item in scheduler._waiting],
            ["short-3"],
        )

    def test_waiting_lookahead_prefers_full_cost_bucket(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(
                max_running_requests=3,
                max_total_target_tokens=400,
                max_cost_ratio=1.2,
            ),
        )
        scheduler._waiting = deque(
            [
                self._waiting("long", 500),
                self._waiting("short-1", 100),
                self._waiting("short-2", 105),
                self._waiting("short-3", 110),
            ]
        )

        admitted = scheduler._pop_admissible_locked(capacity=3, running=[])

        self.assertEqual(
            [item.request.request_id for item in admitted],
            ["short-1", "short-2", "short-3"],
        )
        self.assertEqual(
            [item.request.request_id for item in scheduler._waiting],
            ["long"],
        )

    def test_waiting_lookahead_prefers_tighter_full_cost_bucket(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(
                max_running_requests=3,
                max_total_target_tokens=1000,
                max_cost_ratio=2.1,
            ),
        )
        scheduler._waiting = deque(
            [
                self._waiting("broad-1", 100),
                self._waiting("broad-2", 190),
                self._waiting("broad-3", 195),
                self._waiting("tight-1", 300),
                self._waiting("tight-2", 305),
                self._waiting("tight-3", 310),
            ]
        )

        admitted = scheduler._pop_admissible_locked(capacity=3, running=[])

        self.assertEqual(
            [item.request.request_id for item in admitted],
            ["tight-1", "tight-2", "tight-3"],
        )

    def test_waiting_lookahead_prefers_tighter_context_bucket(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(
                max_running_requests=3,
                max_total_target_tokens=1000,
                max_total_context_tokens=10_000,
                max_cost_ratio=10.0,
                max_context_ratio=10.0,
            ),
        )
        scheduler._waiting = deque(
            [
                self._waiting("broad-1", 100, context=100),
                self._waiting("broad-2", 100, context=180),
                self._waiting("broad-3", 100, context=185),
                self._waiting("tight-1", 100, context=300),
                self._waiting("tight-2", 100, context=305),
                self._waiting("tight-3", 100, context=310),
            ]
        )

        admitted = scheduler._pop_admissible_locked(capacity=3, running=[])

        self.assertEqual(
            [item.request.request_id for item in admitted],
            ["tight-1", "tight-2", "tight-3"],
        )

    def test_waiting_lookahead_does_not_bypass_timed_out_fifo_seed(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(
                max_running_requests=3,
                max_wait_ms=1.0,
                max_total_target_tokens=1000,
                max_cost_ratio=1.2,
            ),
        )
        scheduler._waiting = deque(
            [
                self._old_waiting("long", 500),
                self._waiting("short-1", 100),
                self._waiting("short-2", 105),
                self._waiting("short-3", 110),
            ]
        )

        admitted = scheduler._pop_admissible_locked(capacity=3, running=[])

        self.assertEqual([item.request.request_id for item in admitted], ["long"])

    def test_duration_estimate_uses_configured_frame_rate(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(frame_rate=50),
        )
        req = OmniVoiceBatchRequest(request_id="x", text="ignored", duration=2.5)

        self.assertEqual(scheduler._estimate_target_tokens(req), 125)

    def test_invalid_enforce_output_duration_flag_fails_before_preprocess(self):
        model = _PreprocessCountingModel()
        scheduler = StepwiseOmniVoiceScheduler(model=model)
        waiting = self._waiting(
            "bad",
            25,
            enforce_output_duration="false",
        )

        with self.assertRaisesRegex(ValueError, "enforce_output_duration.*bool"):
            scheduler._prepare_running(waiting)

        self.assertEqual(model.preprocess_calls, 0)

    def test_invalid_default_enforce_output_duration_fails_before_preprocess(self):
        model = _PreprocessCountingModel()
        generation_config = OmniVoiceGenerationConfig()
        generation_config.enforce_output_duration = "false"
        scheduler = StepwiseOmniVoiceScheduler(
            model=model,
            generation_config=generation_config,
        )
        waiting = self._waiting("bad-default", 25)

        with self.assertRaisesRegex(ValueError, "enforce_output_duration.*bool"):
            scheduler._prepare_running(waiting)

        self.assertEqual(model.preprocess_calls, 0)

    def test_static_step_shape_uses_fixed_slots_and_length_buckets(self):
        scheduler = StepwiseOmniVoiceScheduler(
            model=None,
            scheduler_config=StepwiseSchedulerConfig(
                max_running_requests=4,
                compile_static_shape=True,
                seq_len_bucket_multiple=8,
                target_len_bucket_multiple=4,
            ),
        )
        running = [
            _RunningRequest(
                waiting=self._waiting("a", 5),
                state=_State(target_len=5, cond_len=17),
                ref_rms=None,
                started_at=0.0,
            ),
            _RunningRequest(
                waiting=self._waiting("b", 7),
                state=_State(target_len=7, cond_len=19),
                ref_rms=None,
                started_at=0.0,
            ),
        ]

        self.assertEqual(
            scheduler._step_shape_for_running(running),
            {
                "batch_size_pad": 4,
                "seq_len_pad": 24,
                "target_len_pad": 8,
            },
        )

    def test_pack_active_states_supports_static_padding_slots(self):
        state = StepwiseGenerationState(
            request_id="a",
            cond_input_ids=torch.arange(20, dtype=torch.long).view(1, 2, 10),
            cond_audio_mask=torch.tensor(
                [[False, False, False, False, False, False, True, True, True, True]]
            ),
            target_len=4,
            schedule=[1],
            step_index=0,
            tokens=torch.full((2, 4), 99, dtype=torch.long),
        )

        (
            input_ids,
            audio_mask,
            attention_mask,
            target_slices,
            uncond_offset,
        ) = _pack_active_states(
            _Model(),
            [state],
            batch_size_pad=4,
            seq_len_pad=16,
            target_len_pad=8,
        )

        self.assertEqual(tuple(input_ids.shape), (8, 2, 16))
        self.assertEqual(tuple(audio_mask.shape), (8, 16))
        self.assertEqual(tuple(attention_mask.shape), (8, 1, 16, 16))
        self.assertEqual(uncond_offset, 4)
        self.assertEqual(
            target_slices,
            [
                (6, 10),
                (0, 8),
                (0, 8),
                (0, 8),
                (0, 4),
                (0, 8),
                (0, 8),
                (0, 8),
            ],
        )
        self.assertTrue(bool(attention_mask[1, 0, 0, 0]))
        self.assertTrue(bool(attention_mask[5, 0, 0, 0]))

    def test_run_generation_step_vectorizes_same_shape_updates(self):
        model = _VectorModel()
        states = [
            StepwiseGenerationState(
                request_id="a",
                cond_input_ids=torch.full((1, 1, 2), 99, dtype=torch.long),
                cond_audio_mask=torch.tensor([[True, True]]),
                target_len=2,
                schedule=[1],
                step_index=0,
                tokens=torch.full((1, 2), 99, dtype=torch.long),
            ),
            StepwiseGenerationState(
                request_id="b",
                cond_input_ids=torch.full((1, 1, 2), 99, dtype=torch.long),
                cond_audio_mask=torch.tensor([[True, True]]),
                target_len=2,
                schedule=[1],
                step_index=0,
                tokens=torch.full((1, 2), 99, dtype=torch.long),
            ),
        ]

        run_generation_step(
            model,
            states,
            OmniVoiceGenerationConfig(
                num_step=1,
                position_temperature=0.0,
                layer_penalty_factor=0.0,
            ),
        )

        self.assertEqual(model.predict_calls, 1)
        self.assertTrue(all(state.completed for state in states))
        self.assertTrue(all(int(state.tokens[0, -1]) == 1 for state in states))

    def test_submit_cancellation_removes_waiting_request(self):
        async def run():
            scheduler = StepwiseOmniVoiceScheduler(
                model=None,
                scheduler_config=StepwiseSchedulerConfig(max_wait_ms=10000.0),
            )
            task = asyncio.create_task(
                scheduler.submit(OmniVoiceBatchRequest(request_id="x", text="x"))
            )
            await asyncio.sleep(0)

            self.assertEqual(
                [item.request.request_id for item in scheduler._waiting],
                ["x"],
            )

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(list(scheduler._waiting), [])

        self.loop.run_until_complete(run())

    def test_control_cancellation_removes_pending_control(self):
        async def run():
            scheduler = StepwiseOmniVoiceScheduler(
                model=None,
                scheduler_config=StepwiseSchedulerConfig(control_queue_capacity=1),
            )
            task = asyncio.create_task(scheduler._run_control(lambda: "ok"))
            await asyncio.sleep(0)

            self.assertEqual(len(scheduler._control), 1)

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(list(scheduler._control), [])

        self.loop.run_until_complete(run())

    def test_control_queue_capacity_is_enforced(self):
        async def run():
            scheduler = StepwiseOmniVoiceScheduler(
                model=None,
                scheduler_config=StepwiseSchedulerConfig(control_queue_capacity=1),
            )
            task = asyncio.create_task(scheduler._run_control(lambda: "first"))
            await asyncio.sleep(0)

            with self.assertRaisesRegex(RuntimeError, "control queue is full"):
                await scheduler._run_control(lambda: "second")

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.loop.run_until_complete(run())

    def test_control_execution_sets_result_safely(self):
        scheduler = StepwiseOmniVoiceScheduler(model=None)
        future = self.loop.create_future()
        control = _ControlRequest(func=lambda: "ok", future=future, loop=self.loop)

        scheduler._execute_control(control)
        self.loop.run_until_complete(asyncio.sleep(0))

        self.assertEqual(future.result(), "ok")

    def test_running_done_futures_are_dropped(self):
        scheduler = StepwiseOmniVoiceScheduler(model=None)
        cancelled = self._waiting("cancelled", 100)
        cancelled.future.cancel()
        keep = self._waiting("keep", 100)
        running = [
            _RunningRequest(
                waiting=cancelled,
                state=_State(100),
                ref_rms=None,
                started_at=0.0,
            ),
            _RunningRequest(
                waiting=keep,
                state=_State(100),
                ref_rms=None,
                started_at=0.0,
            ),
        ]

        remaining = scheduler._drop_done_running(running)

        self.assertEqual([item.waiting.request.request_id for item in remaining], ["keep"])


if __name__ == "__main__":
    unittest.main()
