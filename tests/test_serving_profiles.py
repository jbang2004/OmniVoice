import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

from omnivoice.cli.recommend_serving_profile import (
    build_recommended_serve_command,
    get_parser as get_recommend_parser,
)
from omnivoice.cli.serve_online_batch import (
    _scheduler_config,
    get_parser as get_serve_parser,
)
from omnivoice.serving import (
    BatchSchedulerConfig,
    get_serving_profile,
    recommend_serving_profile,
    recommended_runtime_config,
    summarize_workload,
)


class ServingProfileTests(unittest.TestCase):
    def test_default_http_server_profile_uses_measured_balanced_config(self):
        args = get_serve_parser().parse_args([])

        config = _scheduler_config(args)

        self.assertEqual(args.scheduler_profile, "balanced12")
        self.assertEqual(config.max_batch_size, 12)
        self.assertEqual(config.max_wait_ms, 20.0)
        self.assertEqual(config.max_total_target_tokens, 2048)
        self.assertEqual(config.max_total_context_tokens, 4096)
        self.assertEqual(config.max_cost_ratio, 1.4)
        self.assertEqual(config.max_context_ratio, 2.0)
        self.assertEqual(config.max_context_padding_ratio, 2.0)
        self.assertTrue(config.lookahead_for_partial_batch)
        self.assertEqual(config.partial_lookahead_max_wait_multiplier, 2.0)
        self.assertEqual(config.candidate_pack_policy, "target_context")
        self.assertTrue(config.use_model_duration_estimator)
        self.assertEqual(config.max_generation_batches_before_control, 8)

    def test_profile_can_be_partially_overridden(self):
        args = get_serve_parser().parse_args(
            [
                "--scheduler_profile",
                "burst24",
                "--batch_size",
                "16",
                "--max_total_context_tokens",
                "5000",
                "--max_context_ratio",
                "1.5",
                "--max_context_padding_ratio",
                "1.7",
                "--lookahead_for_partial_batch",
                "false",
                "--partial_lookahead_max_wait_multiplier",
                "3.0",
                "--candidate_pack_policy",
                "target",
                "--max_generation_batches_before_control",
                "4",
            ]
        )

        config = _scheduler_config(args)

        self.assertEqual(config.max_batch_size, 16)
        self.assertEqual(config.max_wait_ms, 20.0)
        self.assertEqual(config.max_total_target_tokens, 4096)
        self.assertEqual(config.max_total_context_tokens, 5000)
        self.assertEqual(config.max_cost_ratio, 2.0)
        self.assertEqual(config.max_context_ratio, 1.5)
        self.assertEqual(config.max_context_padding_ratio, 1.7)
        self.assertFalse(config.lookahead_for_partial_batch)
        self.assertEqual(config.partial_lookahead_max_wait_multiplier, 3.0)
        self.assertEqual(config.max_seed_lookahead, 96)
        self.assertEqual(config.candidate_pack_policy, "target")
        self.assertEqual(config.max_generation_batches_before_control, 4)

    def test_custom_profile_uses_batch_scheduler_defaults(self):
        args = get_serve_parser().parse_args(["--scheduler_profile", "custom"])

        config = _scheduler_config(args)
        defaults = BatchSchedulerConfig()

        self.assertEqual(config, defaults)

    def test_wide32_profile_documents_large_burst_configuration(self):
        profile = get_serving_profile("wide32")

        self.assertEqual(profile.scheduler["max_batch_size"], 32)
        self.assertEqual(profile.scheduler["max_wait_ms"], 100.0)
        self.assertEqual(profile.scheduler["max_total_target_tokens"], 6144)
        self.assertEqual(profile.scheduler["max_total_context_tokens"], 12288)
        self.assertEqual(profile.scheduler["max_cost_ratio"], 2.0)
        self.assertEqual(profile.scheduler["max_context_ratio"], 2.0)
        self.assertEqual(profile.scheduler["max_context_padding_ratio"], 2.0)
        self.assertEqual(profile.scheduler["candidate_pack_policy"], "target_context")
        self.assertTrue(profile.scheduler["lookahead_for_partial_batch"])

    def test_wide48_profile_documents_optional_larger_burst_configuration(self):
        profile = get_serving_profile("wide48")

        self.assertEqual(profile.scheduler["max_batch_size"], 48)
        self.assertEqual(profile.scheduler["max_wait_ms"], 100.0)
        self.assertEqual(profile.scheduler["max_total_target_tokens"], 8192)
        self.assertEqual(profile.scheduler["max_total_context_tokens"], 18432)
        self.assertEqual(profile.scheduler["max_cost_ratio"], 2.0)
        self.assertEqual(profile.scheduler["max_context_ratio"], 2.0)
        self.assertEqual(profile.scheduler["max_context_padding_ratio"], 2.0)
        self.assertEqual(profile.scheduler["ready_queue_capacity"], 384)
        self.assertEqual(profile.scheduler["max_seed_lookahead"], 192)
        self.assertEqual(profile.scheduler["candidate_pack_policy"], "target_context")
        self.assertTrue(profile.scheduler["lookahead_for_partial_batch"])

    def test_burst24_profile_allows_full_clone_batch_context(self):
        profile = get_serving_profile("burst24")

        self.assertEqual(profile.scheduler["max_batch_size"], 24)
        self.assertEqual(profile.scheduler["max_total_target_tokens"], 4096)
        self.assertEqual(profile.scheduler["max_total_context_tokens"], 8192)
        self.assertEqual(profile.scheduler["ready_queue_capacity"], 192)
        self.assertEqual(profile.scheduler["max_seed_lookahead"], 96)
        self.assertTrue(profile.scheduler["lookahead_for_partial_batch"])

    def test_context12_profile_documents_context_heterogeneous_burst_config(self):
        profile = get_serving_profile("context12")

        self.assertEqual(profile.scheduler["max_batch_size"], 12)
        self.assertEqual(profile.scheduler["max_wait_ms"], 20.0)
        self.assertEqual(profile.scheduler["max_total_target_tokens"], 2048)
        self.assertEqual(profile.scheduler["max_total_context_tokens"], 4096)
        self.assertEqual(profile.scheduler["max_cost_ratio"], 1.4)
        self.assertEqual(profile.scheduler["max_context_ratio"], 2.0)
        self.assertEqual(profile.scheduler["max_context_padding_ratio"], 2.0)
        self.assertEqual(profile.scheduler["ready_queue_capacity"], 128)
        self.assertEqual(profile.scheduler["max_seed_lookahead"], 64)
        self.assertTrue(profile.scheduler["lookahead_for_partial_batch"])

    def test_context8_profile_remains_available_for_explicit_ab_tests(self):
        profile = get_serving_profile("context8")

        self.assertEqual(profile.scheduler["max_batch_size"], 8)
        self.assertEqual(profile.scheduler["max_total_context_tokens"], 4096)
        self.assertEqual(profile.scheduler["max_context_padding_ratio"], 2.0)

    def test_context16_profile_remains_available_for_explicit_ab_tests(self):
        profile = get_serving_profile("context16")

        self.assertEqual(profile.scheduler["max_batch_size"], 16)
        self.assertEqual(profile.scheduler["max_total_context_tokens"], 4096)
        self.assertEqual(profile.scheduler["ready_queue_capacity"], 192)
        self.assertEqual(profile.scheduler["max_seed_lookahead"], 96)

    def test_profile_recommendation_selects_context12_for_context_outliers(self):
        samples = [
            {
                "id": f"short-{index}",
                "text": "short",
                "ref_audio": "/tmp/ref.wav",
                "ref_text": "reference",
                "duration": 1.2,
            }
            for index in range(20)
        ]
        samples.extend(
            [
                {
                    "id": "long-1",
                    "text": "long text " * 80,
                    "ref_audio": "/tmp/ref.wav",
                    "ref_text": "long reference " * 30,
                    "duration": 1.2,
                },
                {
                    "id": "long-2",
                    "text": "long text " * 80,
                    "ref_audio": "/tmp/ref.wav",
                    "ref_text": "long reference " * 30,
                    "duration": 1.2,
                },
            ]
        )

        recommendation = recommend_serving_profile(samples, concurrency=24)

        self.assertEqual(recommendation.profile.name, "context12")
        self.assertTrue(recommendation.stats["context_heterogeneous"])
        self.assertGreater(recommendation.stats["context_p95_to_p50_ratio"], 2.5)
        self.assertIn("context12", recommendation.reasons[0])

    def test_profile_recommendation_prefers_context12_over_wide48_for_outliers(self):
        samples = [
            {
                "id": f"short-{index}",
                "text": "short",
                "ref_audio": "/tmp/ref.wav",
                "ref_text": "reference",
                "duration": 1.2,
            }
            for index in range(45)
        ]
        samples.extend(
            [
                {
                    "id": "long-1",
                    "text": "long text " * 80,
                    "ref_audio": "/tmp/ref.wav",
                    "ref_text": "long reference " * 30,
                    "duration": 1.2,
                },
                {
                    "id": "long-2",
                    "text": "long text " * 80,
                    "ref_audio": "/tmp/ref.wav",
                    "ref_text": "long reference " * 30,
                    "duration": 1.2,
                },
                {
                    "id": "long-3",
                    "text": "long text " * 80,
                    "ref_audio": "/tmp/ref.wav",
                    "ref_text": "long reference " * 30,
                    "duration": 1.2,
                },
            ]
        )

        recommendation = recommend_serving_profile(samples, concurrency=48)

        self.assertEqual(recommendation.profile.name, "context12")
        self.assertTrue(recommendation.stats["context_heterogeneous"])

    def test_profile_recommendation_uses_ref_audio_path_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            short_ref = f"{tmpdir}/short.wav"
            long_ref = f"{tmpdir}/long.wav"
            sf.write(short_ref, np.zeros(24_000, dtype=np.float32), 24_000)
            sf.write(long_ref, np.zeros(288_000, dtype=np.float32), 24_000)
            samples = [
                {
                    "id": f"short-{index}",
                    "text": "short",
                    "ref_audio": short_ref,
                    "ref_text": "same reference",
                    "duration": 1.2,
                }
                for index in range(20)
            ]
            samples.extend(
                [
                    {
                        "id": "long-ref-1",
                        "text": "short",
                        "ref_audio": long_ref,
                        "ref_text": "same reference",
                        "duration": 1.2,
                    },
                    {
                        "id": "long-ref-2",
                        "text": "short",
                        "ref_audio": long_ref,
                        "ref_text": "same reference",
                        "duration": 1.2,
                    },
                ]
            )
            original_info = sf.info

            with mock.patch(
                "omnivoice.serving.profiles.sf.info",
                wraps=original_info,
            ) as info:
                recommendation = recommend_serving_profile(samples, concurrency=24)

        self.assertEqual(recommendation.profile.name, "context12")
        self.assertTrue(recommendation.stats["context_heterogeneous"])
        self.assertGreater(recommendation.stats["context_tokens"]["max"], 300)
        self.assertEqual(info.call_count, 2)

    def test_profile_recommendation_selects_burst_and_wide_for_homogeneous_load(self):
        samples = [
            {
                "id": str(index),
                "text": "same length request",
                "duration": 1.2,
            }
            for index in range(56)
        ]

        burst = recommend_serving_profile(samples[:24], concurrency=24)
        wide = recommend_serving_profile(samples[:40], concurrency=32)
        wide48 = recommend_serving_profile(samples, concurrency=48)

        self.assertEqual(burst.profile.name, "burst24")
        self.assertEqual(wide.profile.name, "wide32")
        self.assertEqual(wide48.profile.name, "wide48")

    def test_profile_recommendation_selects_balanced_for_low_concurrency(self):
        samples = [{"id": "a", "text": "short", "duration": 1.0}]

        recommendation = recommend_serving_profile(samples, concurrency=4)

        self.assertEqual(recommendation.profile.name, "balanced12")

    def test_workload_summary_reports_clone_share_and_parser_accepts_inputs(self):
        samples = [
            {
                "id": "a",
                "text": "short",
                "ref_audio": "/tmp/ref.wav",
                "ref_text": "reference",
                "duration": 1.0,
            },
            {"id": "b", "text": "auto", "duration": 1.0},
        ]

        stats = summarize_workload(samples, concurrency=2)
        args = get_recommend_parser().parse_args(
            [
                "--test_list",
                "/tmp/test.jsonl",
                "--concurrency",
                "24",
                "--latency_sensitive",
                "true",
                "--model",
                "/models/OmniVoice",
                "--host",
                "127.0.0.1",
                "--port",
                "18080",
                "--warmup_test_list",
                "/tmp/warmup.jsonl",
            ]
        )

        self.assertEqual(stats["mode_counts"], {"clone": 1, "auto": 1})
        self.assertEqual(stats["clone_request_share"], 0.5)
        self.assertEqual(args.concurrency, 24)
        self.assertTrue(args.latency_sensitive)
        self.assertEqual(args.model, "/models/OmniVoice")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 18080)
        self.assertEqual(args.warmup_test_list, "/tmp/warmup.jsonl")

    def test_recommendation_includes_runtime_and_serve_command(self):
        samples = [
            {
                "id": str(index),
                "text": "same length request",
                "duration": 1.2,
            }
            for index in range(24)
        ]

        recommendation = recommend_serving_profile(samples, concurrency=24)
        payload = recommendation.to_dict()
        command = build_recommended_serve_command(
            recommendation,
            model="/models/OmniVoice",
            host="127.0.0.1",
            port=18080,
            warmup_test_list="/tmp/warmup.jsonl",
        )

        self.assertEqual(payload["recommended_profile"], "burst24")
        self.assertTrue(payload["runtime"]["compile_llm"])
        self.assertEqual(payload["runtime"]["compile_mode"], "default")
        self.assertTrue(payload["runtime"]["startup_warmup_required"])
        self.assertIn("omnivoice-serve-online-batch", command)
        self.assertEqual(command[command.index("--scheduler_profile") + 1], "burst24")
        self.assertEqual(command[command.index("--compile_llm") + 1], "true")
        self.assertEqual(command[command.index("--compile_mode") + 1], "default")
        self.assertEqual(command[command.index("--split_guidance_forward") + 1], "auto")
        self.assertEqual(command[command.index("--split_guidance_min_batch_size") + 1], "8")
        self.assertEqual(
            command[command.index("--split_guidance_min_saved_context_ratio") + 1],
            "0.25",
        )
        self.assertEqual(command[command.index("--warmup_test_list") + 1], "/tmp/warmup.jsonl")
        self.assertEqual(command[command.index("--warmup_batches") + 1], "2")
        self.assertEqual(command[command.index("--warmup_fill_batch") + 1], "true")

    def test_recommended_runtime_keeps_quality_step_count(self):
        runtime = recommended_runtime_config()

        self.assertEqual(runtime["num_step"], 32)
        self.assertTrue(runtime["reuse_static_input_embeds"])
        self.assertTrue(runtime["batched_decode"])
        self.assertFalse(runtime["compile_audio_heads"])
        self.assertEqual(runtime["split_guidance_forward"], "auto")
        self.assertEqual(runtime["split_guidance_min_batch_size"], 8)
        self.assertEqual(runtime["split_guidance_min_saved_context_ratio"], 0.25)
        self.assertTrue(runtime["warmup_fill_batch"])


if __name__ == "__main__":
    unittest.main()
