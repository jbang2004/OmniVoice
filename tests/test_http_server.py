import base64
import io
import unittest

import numpy as np
import soundfile as sf
import torch

from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.serving import OnlineBatchServerState, SchedulerSnapshot
from omnivoice.serving.http_server import create_online_batch_app


class FakeScheduler:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.requests = []
        self.prompt_requests = []
        self.reset_metrics_calls = []
        self.single_submit_count = 0
        self.submit_many_count = 0

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def submit(self, request):
        self.single_submit_count += 1
        self.requests.append(request)
        return self._result_for(request)

    async def submit_many(self, requests):
        self.submit_many_count += 1
        self.requests.extend(requests)
        return [self._result_for(request) for request in requests]

    def _result_for(self, request):
        return type(
            "Result",
            (),
            {
                "request_id": request.request_id,
                "audio": np.zeros(240, dtype=np.float32),
                "sample_rate": 24000,
                "batch_size": 2,
                "queue_wait_ms": 3.0,
                "batch_infer_s": 0.25,
                "batch_reason": "full",
                "batch_cost_tokens": 120,
                "batch_max_cost_tokens": 60,
                "batch_context_tokens": 360,
                "batch_max_context_tokens": 180,
                "batch_context_padding_ratio": 1.25,
            },
        )()

    async def create_voice_clone_prompt(
        self,
        *,
        ref_audio,
        ref_text=None,
        preprocess_prompt=None,
        cache_key=None,
    ):
        self.prompt_requests.append(
            {
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "preprocess_prompt": preprocess_prompt,
                "cache_key": cache_key,
            }
        )
        return VoiceClonePrompt(
            ref_audio_tokens=torch.zeros((1, 1), dtype=torch.long),
            ref_text=ref_text or "auto text",
            ref_rms=0.1,
        )

    def snapshot(self):
        return SchedulerSnapshot(
            pending_high=0,
            pending_normal=0,
            total_batches=1,
            total_requests=len(self.requests),
            avg_batch_size=2.0,
            prompt_cache_hits=0,
            prompt_cache_misses=0,
            last_dispatch_reason="full",
        )

    def reset_metrics(self, *, reset_prompt_cache_stats=True):
        self.reset_metrics_calls.append(
            {"reset_prompt_cache_stats": reset_prompt_cache_stats}
        )


class FailingScheduler(FakeScheduler):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    async def submit(self, request):
        raise self.exc

    async def submit_many(self, requests):
        raise self.exc


class FailingPromptScheduler(FakeScheduler):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    async def create_voice_clone_prompt(
        self,
        *,
        ref_audio,
        ref_text=None,
        preprocess_prompt=None,
        cache_key=None,
    ):
        raise self.exc


def _wav_base64() -> str:
    audio = np.zeros(240, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class HttpServerTests(unittest.TestCase):
    def _client(self, scheduler):
        from fastapi.testclient import TestClient

        app = create_online_batch_app(
            OnlineBatchServerState(
                scheduler=scheduler,
                sample_rate=24000,
                max_request_text_chars=20,
            )
        )
        return TestClient(app)

    def _client_with_state(self, state):
        from fastapi.testclient import TestClient

        return TestClient(create_online_batch_app(state))

    def test_healthz_reports_disabled_startup_warmup(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["server"]["sample_rate"], 24000)
        self.assertEqual(body["server"]["max_request_text_chars"], 20)
        self.assertEqual(body["server"]["max_voice_prompts"], 256)
        self.assertEqual(body["scheduler"]["total_batches"], 1)
        self.assertEqual(body["voices"]["size"], 0)
        self.assertEqual(
            body["startup_warmup"],
            {
                "enabled": False,
                "status": "disabled",
                "started_at": None,
                "completed_at": None,
                "duration_s": None,
                "error": None,
            },
        )

    def test_server_snapshot_reports_runtime_scheduler_voice_and_warmup(self):
        scheduler = FakeScheduler()
        state = OnlineBatchServerState(
            scheduler=scheduler,
            sample_rate=22050,
            max_request_text_chars=123,
            max_voice_prompts=7,
        )
        with self._client_with_state(state) as client:
            response = client.get("/v1/server")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["server"],
            {
                "sample_rate": 22050,
                "max_request_text_chars": 123,
                "max_voice_prompts": 7,
            },
        )
        self.assertEqual(body["scheduler"]["total_batches"], 1)
        self.assertEqual(body["voices"]["max_entries"], 7)
        self.assertEqual(body["startup_warmup"]["status"], "disabled")

    def test_healthz_reports_completed_startup_warmup(self):
        from fastapi.testclient import TestClient

        scheduler = FakeScheduler()
        state = OnlineBatchServerState(
            scheduler=scheduler,
            sample_rate=24000,
            max_request_text_chars=20,
        )
        warmup_calls = []

        async def warmup():
            warmup_calls.append("called")

        app = create_online_batch_app(state, startup_warmup=warmup)
        with TestClient(app) as client:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(warmup_calls, ["called"])
        self.assertTrue(scheduler.started)
        self.assertTrue(scheduler.stopped)
        warmup_status = response.json()["startup_warmup"]
        self.assertTrue(warmup_status["enabled"])
        self.assertEqual(warmup_status["status"], "completed")
        self.assertIsNotNone(warmup_status["started_at"])
        self.assertIsNotNone(warmup_status["completed_at"])
        self.assertGreaterEqual(warmup_status["duration_s"], 0.0)
        self.assertIsNone(warmup_status["error"])

    def test_startup_warmup_failure_records_status_and_stops_scheduler(self):
        from fastapi.testclient import TestClient

        scheduler = FakeScheduler()
        state = OnlineBatchServerState(
            scheduler=scheduler,
            sample_rate=24000,
            max_request_text_chars=20,
        )

        async def warmup():
            raise RuntimeError("warmup failed")

        app = create_online_batch_app(state, startup_warmup=warmup)
        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            with TestClient(app):
                pass

        self.assertTrue(scheduler.started)
        self.assertTrue(scheduler.stopped)
        self.assertTrue(state.startup_warmup.enabled)
        self.assertEqual(state.startup_warmup.status, "failed")
        self.assertIsNotNone(state.startup_warmup.started_at)
        self.assertIsNotNone(state.startup_warmup.completed_at)
        self.assertIsNotNone(state.startup_warmup.duration_s)
        self.assertEqual(state.startup_warmup.error, "RuntimeError: warmup failed")

    def test_tts_returns_wav_and_scheduler_headers(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/tts",
                json={
                    "request_id": "req-1",
                    "text": "hello",
                    "language": "en",
                    "duration": 1.5,
                    "enforce_output_duration": True,
                    "cost_tokens_hint": 77,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-omnivoice-request-id"], "req-1")
        self.assertEqual(response.headers["x-omnivoice-batch-size"], "2")
        self.assertEqual(response.headers["x-omnivoice-batch-cost-tokens"], "120")
        self.assertEqual(
            response.headers["x-omnivoice-batch-max-cost-tokens"],
            "60",
        )
        self.assertEqual(response.headers["x-omnivoice-batch-context-tokens"], "360")
        self.assertEqual(
            response.headers["x-omnivoice-batch-max-context-tokens"],
            "180",
        )
        self.assertEqual(
            response.headers["x-omnivoice-batch-context-padding-ratio"],
            "1.250000",
        )
        self.assertTrue(response.content.startswith(b"RIFF"))
        self.assertTrue(scheduler.started)
        self.assertTrue(scheduler.stopped)
        self.assertEqual(scheduler.requests[0].text, "hello")
        self.assertEqual(scheduler.requests[0].duration, 1.5)
        self.assertTrue(scheduler.requests[0].enforce_output_duration)
        self.assertEqual(scheduler.requests[0].cost_tokens_hint, 77)

    def test_tts_rejects_coerced_generation_field_types(self):
        cases = [
            {"text": "hello", "enforce_output_duration": "false"},
            {"text": "hello", "duration": "1.5"},
            {"text": "hello", "duration": True},
            {"text": "hello", "speed": "1.2"},
            {"text": "hello", "cost_tokens_hint": "77"},
            {"text": "hello", "cost_tokens_hint": 77.0},
        ]
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            for payload in cases:
                with self.subTest(payload=payload):
                    response = client.post("/v1/tts", json=payload)
                    self.assertEqual(response.status_code, 422)

        self.assertEqual(scheduler.requests, [])

    def test_rejects_unknown_request_fields(self):
        cases = (
            ("/v1/tts", {"text": "hello", "duraton": 1.5}, "duraton"),
            (
                "/v1/tts_batch",
                {"requests": [{"text": "hello", "duraton": 1.5}]},
                "duraton",
            ),
            (
                "/v1/tts_batch",
                {"requests": [{"text": "hello"}], "unknown": True},
                "unknown",
            ),
            (
                "/v1/voices",
                {"ref_audio_base64": _wav_base64(), "unknown": True},
                "unknown",
            ),
            (
                "/v1/scheduler/reset_metrics",
                {"reset_prompt_cache_stats": False, "unknown": True},
                "unknown",
            ),
        )

        for path, payload, field_name in cases:
            scheduler = FakeScheduler()
            with self.subTest(path=path, field_name=field_name):
                with self._client(scheduler) as client:
                    response = client.post(path, json=payload)

                self.assertEqual(response.status_code, 422)
                self.assertIn(field_name, response.text)
                self.assertEqual(scheduler.requests, [])
                self.assertEqual(scheduler.prompt_requests, [])

    def test_reset_scheduler_metrics_endpoint(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/scheduler/reset_metrics",
                json={"reset_prompt_cache_stats": False},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["reset"])
        self.assertEqual(
            scheduler.reset_metrics_calls,
            [{"reset_prompt_cache_stats": False}],
        )

    def test_reset_scheduler_metrics_rejects_string_bool(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/scheduler/reset_metrics",
                json={"reset_prompt_cache_stats": "false"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(scheduler.reset_metrics_calls, [])

    def test_base64_ref_audio_gets_decoded_and_cache_keyed(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/tts",
                json={
                    "request_id": "clone-1",
                    "text": "你好",
                    "language_id": "zh",
                    "ref_audio_base64": _wav_base64(),
                    "ref_text": "参考音频",
                },
            )

        self.assertEqual(response.status_code, 200)
        request = scheduler.requests[0]
        self.assertIsInstance(request.ref_audio, tuple)
        self.assertEqual(request.ref_audio[1], 24000)
        self.assertIsNotNone(request.ref_audio_cache_key)
        self.assertEqual(request.ref_audio_cache_key[0], "base64-sha256")

    def test_batch_endpoint_returns_base64_audio_per_request(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/tts_batch",
                json={
                    "requests": [
                        {"request_id": "a", "text": "A"},
                        {"request_id": "b", "text": "B"},
                    ]
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([item["request_id"] for item in body["results"]], ["a", "b"])
        self.assertTrue(body["results"][0]["audio_base64"])
        self.assertEqual(body["results"][0]["batch_cost_tokens"], 120)
        self.assertEqual(body["results"][0]["batch_max_cost_tokens"], 60)
        self.assertEqual(body["results"][0]["batch_context_tokens"], 360)
        self.assertEqual(body["results"][0]["batch_max_context_tokens"], 180)
        self.assertEqual(
            body["results"][0]["batch_context_padding_ratio"],
            1.25,
        )
        self.assertEqual(
            [request.request_id for request in scheduler.requests],
            ["a", "b"],
        )
        self.assertEqual(scheduler.single_submit_count, 0)
        self.assertEqual(scheduler.submit_many_count, 1)

    def test_rejects_conflicting_voice_modes(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/tts",
                json={
                    "text": "hello",
                    "ref_audio": "/tmp/ref.wav",
                    "instruct": "calm",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "bad_request")

    def test_queue_full_maps_to_429(self):
        scheduler = FailingScheduler(RuntimeError("OmniVoice scheduler queue is full"))
        with self._client(scheduler) as client:
            response = client.post("/v1/tts", json={"text": "hello"})

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "1")
        self.assertEqual(response.json()["detail"]["code"], "queue_full")

    def test_batch_queue_full_maps_to_429_without_partial_submit(self):
        scheduler = FailingScheduler(RuntimeError("OmniVoice scheduler queue is full"))
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/tts_batch",
                json={
                    "requests": [
                        {"request_id": "a", "text": "A"},
                        {"request_id": "b", "text": "B"},
                    ]
                },
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "1")
        self.assertEqual(response.json()["detail"]["code"], "queue_full")
        self.assertEqual(scheduler.requests, [])

    def test_voice_prompt_control_queue_full_maps_to_429(self):
        scheduler = FailingPromptScheduler(
            RuntimeError("OmniVoice scheduler control queue is full")
        )
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/voices",
                json={
                    "voice_id": "speaker-a",
                    "ref_audio_base64": _wav_base64(),
                    "ref_text": "参考音频",
                },
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "1")
        self.assertEqual(response.json()["detail"]["code"], "queue_full")

    def test_scheduler_stopped_maps_to_503(self):
        scheduler = FailingScheduler(RuntimeError("scheduler stopped"))
        with self._client(scheduler) as client:
            response = client.post("/v1/tts", json={"text": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "scheduler_stopped")

    def test_generation_failure_maps_to_structured_500(self):
        scheduler = FailingScheduler(ValueError("model exploded"))
        with self._client(scheduler) as client:
            response = client.post("/v1/tts", json={"text": "hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "generation_failed")
        self.assertIn("model exploded", response.json()["detail"]["message"])

    def test_tts_request_boundary_validation_maps_to_structured_400(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/tts",
                json={
                    "request_id": "   ",
                    "text": "hello",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "bad_request")
        self.assertIn("request_id", response.json()["detail"]["message"])
        self.assertEqual(scheduler.requests, [])

    def test_rejects_blank_request_mode_fields(self):
        cases = (
            ("/v1/tts", {"text": "hello", "language_id": "   "}, "language_id"),
            ("/v1/tts", {"text": "hello", "voice_id": "   "}, "voice_id"),
            ("/v1/tts", {"text": "hello", "ref_audio": "   "}, "ref_audio"),
            (
                "/v1/tts",
                {"text": "hello", "ref_audio_base64": "   "},
                "ref_audio_base64",
            ),
            ("/v1/tts", {"text": "hello", "instruct": "   "}, "instruct"),
            (
                "/v1/voices",
                {"voice_id": "   ", "ref_audio_base64": _wav_base64()},
                "voice_id",
            ),
            ("/v1/voices", {"ref_audio": "   "}, "ref_audio"),
        )

        for path, payload, field_name in cases:
            scheduler = FakeScheduler()
            with self.subTest(path=path, field_name=field_name):
                with self._client(scheduler) as client:
                    response = client.post(path, json=payload)

                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["detail"]["code"], "bad_request")
                self.assertIn(field_name, response.json()["detail"]["message"])
                self.assertEqual(scheduler.requests, [])
                self.assertEqual(scheduler.prompt_requests, [])

    def test_register_voice_prompt_and_use_voice_id(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            register = client.post(
                "/v1/voices",
                json={
                    "voice_id": "speaker-a",
                    "ref_audio_base64": _wav_base64(),
                    "ref_text": "参考音频",
                    "preprocess_prompt": False,
                },
            )
            response = client.post(
                "/v1/tts",
                json={
                    "request_id": "tts-a",
                    "text": "你好",
                    "language_id": "zh",
                    "voice_id": "speaker-a",
                },
            )

        self.assertEqual(register.status_code, 200)
        self.assertEqual(register.json()["voice_id"], "speaker-a")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(scheduler.prompt_requests), 1)
        self.assertFalse(scheduler.prompt_requests[0]["preprocess_prompt"])
        self.assertEqual(
            scheduler.prompt_requests[0]["cache_key"][0],
            "base64-sha256",
        )
        self.assertIsNotNone(scheduler.requests[0].voice_clone_prompt)
        self.assertIsNone(scheduler.requests[0].ref_audio)

    def test_register_voice_rejects_string_preprocess_prompt(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/voices",
                json={
                    "voice_id": "speaker-a",
                    "ref_audio_base64": _wav_base64(),
                    "ref_text": "参考音频",
                    "preprocess_prompt": "false",
                },
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(scheduler.prompt_requests, [])

    def test_voice_prompt_validation_failure_maps_to_structured_400(self):
        scheduler = FailingPromptScheduler(ValueError("invalid prompt audio"))
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/voices",
                json={
                    "voice_id": "speaker-a",
                    "ref_audio_base64": _wav_base64(),
                    "ref_text": "参考音频",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "bad_request")
        self.assertIn("invalid prompt audio", response.json()["detail"]["message"])

    def test_voice_registry_is_bounded_and_lru_evicted(self):
        scheduler = FakeScheduler()
        state = OnlineBatchServerState(
            scheduler=scheduler,
            sample_rate=24000,
            max_request_text_chars=20,
            max_voice_prompts=1,
        )
        with self._client_with_state(state) as client:
            first = client.post(
                "/v1/voices",
                json={
                    "voice_id": "speaker-a",
                    "ref_audio_base64": _wav_base64(),
                    "ref_text": "第一段",
                },
            )
            second = client.post(
                "/v1/voices",
                json={
                    "voice_id": "speaker-b",
                    "ref_audio_base64": _wav_base64(),
                    "ref_text": "第二段",
                },
            )
            voices = client.get("/v1/voices")
            missing = client.post(
                "/v1/tts",
                json={"text": "hello", "voice_id": "speaker-a"},
            )
            present = client.post(
                "/v1/tts",
                json={"text": "hello", "voice_id": "speaker-b"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(voices.status_code, 200)
        self.assertEqual(voices.json()["size"], 1)
        self.assertEqual(voices.json()["max_entries"], 1)
        self.assertEqual(voices.json()["evictions"], 1)
        self.assertEqual(voices.json()["voice_ids"], ["speaker-b"])
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(present.status_code, 200)

    def test_voice_registry_delete_endpoint(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            client.post(
                "/v1/voices",
                json={
                    "voice_id": "speaker-a",
                    "ref_audio_base64": _wav_base64(),
                    "ref_text": "参考音频",
                },
            )
            deleted = client.delete("/v1/voices/speaker-a")
            missing_delete = client.delete("/v1/voices/speaker-a")
            response = client.post(
                "/v1/tts",
                json={"text": "hello", "voice_id": "speaker-a"},
            )

        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(deleted.json()["voices"]["size"], 0)
        self.assertFalse(missing_delete.json()["deleted"])
        self.assertEqual(response.status_code, 404)

    def test_voice_prompts_dict_initialization_remains_supported(self):
        scheduler = FakeScheduler()
        prompt = VoiceClonePrompt(
            ref_audio_tokens=torch.zeros((1, 1), dtype=torch.long),
            ref_text="existing",
            ref_rms=0.1,
        )
        state = OnlineBatchServerState(
            scheduler=scheduler,
            voice_prompts={"speaker-a": prompt},
        )
        with self._client_with_state(state) as client:
            voices = client.get("/v1/voices")
            response = client.post(
                "/v1/tts",
                json={"text": "hello", "voice_id": "speaker-a"},
            )

        self.assertEqual(voices.json()["voice_ids"], ["speaker-a"])
        self.assertEqual(response.status_code, 200)
        self.assertIs(scheduler.requests[0].voice_clone_prompt, prompt)

    def test_unknown_voice_id_returns_404(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/tts",
                json={
                    "text": "hello",
                    "voice_id": "missing",
                },
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "voice_not_found")

    def test_voice_id_rejects_conflicting_voice_inputs(self):
        scheduler = FakeScheduler()
        with self._client(scheduler) as client:
            response = client.post(
                "/v1/tts",
                json={
                    "text": "hello",
                    "voice_id": "speaker-a",
                    "ref_audio": "/tmp/ref.wav",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "bad_request")


if __name__ == "__main__":
    unittest.main()
