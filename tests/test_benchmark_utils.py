import argparse
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from omnivoice.cli.benchmark_http_batch import (
    _audio_seconds_from_wav,
    _expand_samples_for_request_repeats,
    _pre_register_voices,
    _payload_from_sample,
    _result_from_response,
    _sample_with_registered_voice_id,
    _voice_id_for_registration_key,
    _voice_register_url_from_tts_url,
    _voice_registration_key,
    _voice_registration_payload,
)
from omnivoice.cli.benchmark_http_sweep import (
    _compact_result,
    _reset_url_from_scheduler_url,
)
from omnivoice.cli import infer_batch as infer_batch_cli
from omnivoice.cli.benchmark_server_sweep import (
    _build_server_command,
    _compact_server_result,
    _http_sweep_args_for_server,
    _parse_nvidia_smi_gpu_rows,
    _summarize_gpu_samples,
)
from omnivoice.cli.benchmark_utils import summarize_request_results
from omnivoice.cli.benchmark_utils import (
    build_http_sweep_profiles,
    build_server_sweep_profiles,
    diagnose_server_sweep_result,
    parse_bool_grid,
    parse_float_grid,
    parse_int_grid,
    rank_http_sweep_results,
    rank_server_sweep_results,
    recommend_server_sweep_config,
    summarize_best_server_sweep_result,
    summarize_best_http_sweep_result,
)
from omnivoice.cli.infer_online_batch import (
    _effective_compile_mode as _online_effective_compile_mode,
    _request_from_sample as _online_request_from_sample,
    _run_scheduler_warmup,
    _scheduler_config_from_args as _online_scheduler_config_from_args,
    _select_representative_warmup_batches,
    _select_representative_warmup_samples,
    get_parser as get_online_parser,
)
from omnivoice.cli.infer_stepwise_online_batch import (
    _effective_compile_mode as _stepwise_effective_compile_mode,
    _generation_config_from_args as _stepwise_generation_config_from_args,
    get_parser as get_stepwise_parser,
    _local_voice_registration_key,
    _pre_register_local_voice_prompts,
    _request_from_sample as _stepwise_request_from_sample,
    _sample_with_local_voice_prompt,
    _scheduler_config_from_args as _stepwise_scheduler_config_from_args,
)
from omnivoice.cli.make_heterogeneous_test_list import build_samples
from omnivoice.serving import BatchSchedulerConfig
from omnivoice.utils.data_utils import read_test_list


class BenchmarkUtilsTests(unittest.TestCase):
    def test_summarize_request_results_reports_histograms_and_latency(self):
        summary = summarize_request_results(
            [
                {
                    "batch_size": 1,
                    "queue_wait_ms": 0.0,
                    "batch_infer_s": 1.0,
                    "batch_reason": "aging",
                    "batch_cost_tokens": 120,
                    "batch_max_cost_tokens": 120,
                    "batch_context_tokens": 240,
                    "batch_max_context_tokens": 240,
                    "batch_context_padding_ratio": 1.0,
                },
                {
                    "batch_size": 4,
                    "queue_wait_ms": 10.0,
                    "batch_infer_s": 2.0,
                    "batch_reason": "full",
                    "batch_cost_tokens": 360,
                    "batch_max_cost_tokens": 90,
                    "batch_context_tokens": 720,
                    "batch_max_context_tokens": 180,
                    "batch_context_padding_ratio": 1.0,
                },
                {
                    "batch_size": 4,
                    "queue_wait_ms": 20.0,
                    "batch_infer_s": 3.0,
                    "batch_reason": "full",
                    "batch_cost_tokens": 400,
                    "batch_max_cost_tokens": 100,
                    "batch_context_tokens": 800,
                    "batch_max_context_tokens": 200,
                    "batch_context_padding_ratio": 1.1,
                },
            ],
            batch_size_key="batch_size",
            infer_s_key="batch_infer_s",
        )

        self.assertEqual(summary["batch_size_histogram"], {"1": 1, "4": 2})
        self.assertEqual(summary["batch_reason_histogram"], {"aging": 1, "full": 2})
        self.assertEqual(summary["queue_wait_ms"]["p50"], 10.0)
        self.assertEqual(summary["queue_wait_ms"]["max"], 20.0)
        self.assertEqual(summary["request_infer_s"]["p50"], 2.0)
        self.assertEqual(summary["batch_cost_tokens"]["p50"], 360)
        self.assertEqual(summary["batch_cost_tokens"]["max"], 400)
        self.assertEqual(summary["batch_max_cost_tokens"]["p50"], 100)
        self.assertEqual(summary["batch_context_tokens"]["p50"], 720)
        self.assertEqual(summary["batch_context_tokens"]["max"], 800)
        self.assertEqual(summary["batch_max_context_tokens"]["p50"], 200)
        self.assertEqual(summary["batch_context_padding_ratio"]["max"], 1.1)

    def test_http_sweep_grid_and_ranking_helpers(self):
        profiles = build_http_sweep_profiles(
            concurrency_values=parse_int_grid("1, 4"),
            arrival_gap_ms_values=parse_float_grid("0, 2.5"),
            repeats=2,
            warmup_repeats=1,
        )

        self.assertEqual(len(profiles), 12)
        self.assertEqual(profiles[0]["name"], "c1_gap0ms_warmup1")
        self.assertTrue(profiles[0]["is_warmup"])
        self.assertEqual(profiles[1]["name"], "c1_gap0ms_r1")
        self.assertFalse(profiles[1]["is_warmup"])
        self.assertEqual(profiles[-1]["name"], "c4_gap2p5ms_r2")

        slow = {
            "profile": {"name": "slow"},
            "summary": {
                "num_requests": 4,
                "num_successful": 4,
                "rtf_wall": 0.20,
                "wall_s": 2.0,
                "scheduler_after": {"avg_batch_size": 4.0},
                "request_metrics": {"queue_wait_ms": {"p95": 1.0}},
            },
        }
        fast = {
            "profile": {"name": "fast"},
            "summary": {
                "num_requests": 4,
                "num_successful": 4,
                "rtf_wall": 0.10,
                "wall_s": 1.0,
                "scheduler_after": {
                    "avg_batch_size": 2.0,
                    "split_retry_batches": 1,
                    "last_batch_split_retries": 1,
                    "last_batch_model_calls": 3,
                    "last_batch_max_execution_size": 2,
                },
                "request_metrics": {"queue_wait_ms": {"p95": 5.0}},
                "voice_registration": {"enabled": True, "num_registered_voices": 1},
            },
        }
        failing = {
            "profile": {"name": "failing"},
            "summary": {
                "num_requests": 4,
                "num_successful": 3,
                "rtf_wall": 0.01,
                "wall_s": 0.1,
                "scheduler_after": {"avg_batch_size": 8.0},
                "request_metrics": {"queue_wait_ms": {"p95": 0.0}},
            },
        }
        warmup = {
            "profile": {"name": "warmup", "is_warmup": True},
            "summary": {
                "num_requests": 4,
                "num_successful": 4,
                "rtf_wall": 0.001,
                "wall_s": 0.01,
                "scheduler_after": {"avg_batch_size": 8.0},
                "request_metrics": {"queue_wait_ms": {"p95": 0.0}},
            },
        }

        ranked = rank_http_sweep_results([slow, fast, failing, warmup])

        self.assertEqual([row["profile"]["name"] for row in ranked], ["fast", "slow", "failing"])
        self.assertIs(summarize_best_http_sweep_result([slow, fast, failing, warmup]), fast)
        self.assertEqual(_compact_result(fast)["profile"]["name"], "fast")
        self.assertTrue(_compact_result(fast)["voice_registration"]["enabled"])
        self.assertEqual(_compact_result(fast)["split_retry_batches"], 1)
        self.assertEqual(_compact_result(fast)["last_batch_model_calls"], 3)

    def test_bool_grid_parser_accepts_common_spellings(self):
        self.assertEqual(
            parse_bool_grid("true,false,1,0,on,off"),
            [True, False, True, False, True, False],
        )
        with self.assertRaises(ValueError):
            parse_bool_grid("true,maybe")

    def test_http_batch_request_repeats_expand_samples_with_unique_ids(self):
        samples = [
            {"id": "a", "text": "first"},
            {"save_name": "b", "text": "second"},
            {"text": "third"},
        ]

        expanded = _expand_samples_for_request_repeats(
            samples,
            request_repeats=2,
        )

        self.assertEqual(len(expanded), 6)
        self.assertEqual(
            [sample["id"] for sample in expanded],
            [
                "a_rep1",
                "b_rep1",
                "sample_0003_rep1",
                "a_rep2",
                "b_rep2",
                "sample_0003_rep2",
            ],
        )
        self.assertEqual(samples[0]["id"], "a")
        with self.assertRaises(ValueError):
            _expand_samples_for_request_repeats(samples, request_repeats=0)

    def test_http_sweep_reset_url_from_scheduler_url(self):
        self.assertEqual(
            _reset_url_from_scheduler_url("http://127.0.0.1:8000/v1/scheduler"),
            "http://127.0.0.1:8000/v1/scheduler/reset_metrics",
        )
        self.assertEqual(
            _reset_url_from_scheduler_url("http://127.0.0.1:8000/custom"),
            "http://127.0.0.1:8000/custom/reset_metrics",
        )
        self.assertIsNone(_reset_url_from_scheduler_url(None))

    def test_server_sweep_profile_builder_and_ranking(self):
        profiles = build_server_sweep_profiles(
            batch_size_values=[4, 8],
            max_wait_ms_values=[20.0],
            max_cost_ratio_values=[1.2, 1.4],
            max_total_target_tokens_values=[4096],
            max_total_context_tokens_values=[4096, 8192],
            max_context_ratio_values=[1.5, 2.0],
            max_context_padding_ratio_values=[1.5, 2.0],
            lookahead_for_partial_batch_values=[False, True],
            partial_lookahead_max_wait_multiplier_values=[1.5, 2.0],
            num_step_values=[2],
        )

        self.assertEqual(len(profiles), 128)
        self.assertEqual(
            profiles[0]["name"],
            "b4_wait20ms_ratio1p2_tok4096_ctx4096_ctxr1p5_pad1p5_pl0x1p5_step2",
        )
        self.assertEqual(
            profiles[-1]["name"],
            "b8_wait20ms_ratio1p4_tok4096_ctx8192_ctxr2_pad2_pl1x2_step2",
        )

        slower = {
            "server_profile": {"name": "slower"},
            "summary": {
                "num_requests": 8,
                "num_successful": 8,
                "rtf_wall": 0.12,
                "wall_s": 1.2,
                "scheduler_after": {"avg_batch_size": 4.0},
                "request_metrics": {"queue_wait_ms": {"p95": 1.0}},
            },
        }
        faster = {
            "server_profile": {"name": "faster"},
            "summary": {
                "num_requests": 8,
                "num_successful": 8,
                "rtf_wall": 0.08,
                "wall_s": 0.8,
                "scheduler_after": {"avg_batch_size": 2.0},
                "request_metrics": {"queue_wait_ms": {"p95": 3.0}},
            },
            "best": {"compact": {"rtf_wall": 0.08}},
            "gpu": {
                "enabled": True,
                "summary": {
                    "num_samples": 2,
                    "gpu_util_percent": {"p50": 55.0, "p95": 58.0, "max": 60.0},
                    "memory_used_mib": {"p50": 4096.0, "p95": 4096.0, "max": 4096.0},
                    "memory_total_mib": 24576.0,
                },
            },
            "server_log": "/tmp/server.log",
        }
        failed = {
            "server_profile": {"name": "failed"},
            "summary": {
                "num_requests": 8,
                "num_successful": 7,
                "rtf_wall": 0.01,
                "wall_s": 0.1,
                "scheduler_after": {"avg_batch_size": 8.0},
                "request_metrics": {"queue_wait_ms": {"p95": 0.0}},
            },
        }

        ranked = rank_server_sweep_results([slower, faster, failed])

        self.assertEqual(
            [row["server_profile"]["name"] for row in ranked],
            ["faster", "slower", "failed"],
        )
        self.assertIs(
            summarize_best_server_sweep_result([slower, faster, failed]),
            faster,
        )
        self.assertEqual(_compact_server_result(faster)["best"], {"rtf_wall": 0.08})
        self.assertEqual(
            _compact_server_result(faster)["gpu"]["gpu_util_percent"]["max"],
            60.0,
        )

    def test_server_sweep_diagnostics_explain_underfilled_memory_and_queue(self):
        underfilled = {
            "server_profile": {"name": "underfilled", "batch_size": 8},
            "best": {
                "compact": {
                    "profile": {"name": "c2"},
                    "num_requests": 8,
                    "num_successful": 8,
                    "rtf_wall": 0.1,
                    "avg_batch_size": 2.0,
                    "queue_wait_ms": {"p95": 4.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {
                    "gpu_util_percent": {"max": 35.0},
                    "memory_used_mib": {"max": 4096.0},
                },
            },
        }
        memory_pressure = {
            "server_profile": {"name": "memory", "batch_size": 16},
            "best": {
                "compact": {
                    "num_requests": 8,
                    "num_successful": 8,
                    "rtf_wall": 0.08,
                    "avg_batch_size": 16.0,
                    "queue_wait_ms": {"p95": 4.0},
                    "split_retry_batches": 1,
                    "adaptive_batch_caps": {"clone": 8},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 88.0}},
            },
        }
        high_queue = {
            "server_profile": {"name": "queue", "batch_size": 8},
            "best": {
                "compact": {
                    "num_requests": 8,
                    "num_successful": 8,
                    "rtf_wall": 0.09,
                    "avg_batch_size": 8.0,
                    "queue_wait_ms": {"p95": 200.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 92.0}},
            },
        }
        high_queue_underfilled = {
            "server_profile": {"name": "queue-underfilled", "batch_size": 32},
            "best": {
                "compact": {
                    "num_requests": 32,
                    "num_successful": 32,
                    "rtf_wall": 0.25,
                    "avg_batch_size": 16.0,
                    "queue_wait_ms": {"p95": 10000.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 100.0}},
            },
        }
        high_queue_serial_full = {
            "server_profile": {"name": "queue-serial-full", "batch_size": 16},
            "best": {
                "compact": {
                    "num_requests": 52,
                    "num_successful": 52,
                    "rtf_wall": 0.69,
                    "avg_batch_size": 10.4,
                    "queue_wait_ms": {"p95": 24789.0},
                    "batch_reason_histogram": {"full": 48, "timeout": 4},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 100.0}},
            },
        }
        padding_waste = {
            "server_profile": {"name": "padding-waste", "batch_size": 12},
            "best": {
                "compact": {
                    "num_requests": 52,
                    "num_successful": 52,
                    "rtf_wall": 2.1,
                    "avg_batch_size": 12.0,
                    "queue_wait_ms": {"p95": 4.0},
                    "batch_context_padding_ratio": {"p95": 5.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 100.0}},
            },
        }

        underfilled_diag = diagnose_server_sweep_result(
            underfilled,
            target_gpu_util_percent=70.0,
        )
        memory_diag = diagnose_server_sweep_result(memory_pressure)
        queue_diag = diagnose_server_sweep_result(high_queue)
        high_queue_underfilled_diag = diagnose_server_sweep_result(
            high_queue_underfilled,
            high_queue_wait_ms=100.0,
        )
        high_queue_serial_full_diag = diagnose_server_sweep_result(
            high_queue_serial_full,
            high_queue_wait_ms=100.0,
        )
        padding_waste_diag = diagnose_server_sweep_result(padding_waste)
        recommendation = recommend_server_sweep_config(
            [underfilled, memory_pressure, high_queue],
            target_gpu_util_percent=70.0,
            high_queue_wait_ms=100.0,
        )

        self.assertIn(
            "underfilled_batches",
            [issue["code"] for issue in underfilled_diag["issues"]],
        )
        self.assertIn(
            "memory_pressure",
            [issue["code"] for issue in memory_diag["issues"]],
        )
        self.assertIn(
            "queue_latency_high",
            [issue["code"] for issue in queue_diag["issues"]],
        )
        self.assertIn(
            "queue_latency_high_underfilled",
            [issue["code"] for issue in high_queue_underfilled_diag["issues"]],
        )
        self.assertIn(
            "queue_latency_serial_full_batches",
            [issue["code"] for issue in high_queue_serial_full_diag["issues"]],
        )
        self.assertEqual(high_queue_serial_full_diag["full_batch_share"], 48 / 52)
        self.assertIn(
            "Raising max_wait_ms will not help",
            high_queue_serial_full_diag["issues"][0]["action"],
        )
        self.assertNotIn(
            "queue_latency_high_underfilled",
            [issue["code"] for issue in high_queue_serial_full_diag["issues"]],
        )
        self.assertIn(
            "context_padding_waste",
            [issue["code"] for issue in padding_waste_diag["issues"]],
        )
        self.assertNotIn(
            "balanced",
            [issue["code"] for issue in padding_waste_diag["issues"]],
        )
        self.assertIn(
            "lower batch_size",
            high_queue_underfilled_diag["issues"][0]["action"],
        )
        self.assertEqual(
            recommendation["best"]["server_profile"]["name"],
            "memory",
        )
        self.assertEqual(
            recommendation["throughput_best"]["server_profile"]["name"],
            "memory",
        )
        self.assertIsNone(recommendation["balanced_best"])
        self.assertTrue(recommendation["next_actions"])

    def test_server_sweep_recommendation_reports_balanced_best(self):
        throughput = {
            "server_profile": {"name": "throughput", "batch_size": 8},
            "best": {
                "compact": {
                    "profile": {"name": "c12"},
                    "num_requests": 12,
                    "num_successful": 12,
                    "rtf_wall": 0.100,
                    "avg_batch_size": 6.0,
                    "queue_wait_ms": {"p95": 930.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 100.0}},
            },
        }
        balanced = {
            "server_profile": {"name": "balanced", "batch_size": 12},
            "best": {
                "compact": {
                    "profile": {"name": "c12"},
                    "num_requests": 12,
                    "num_successful": 12,
                    "rtf_wall": 0.103,
                    "avg_batch_size": 12.0,
                    "queue_wait_ms": {"p95": 4.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 100.0}},
            },
        }
        slow_balanced = {
            "server_profile": {"name": "slow-balanced", "batch_size": 16},
            "best": {
                "compact": {
                    "profile": {"name": "c12"},
                    "num_requests": 12,
                    "num_successful": 12,
                    "rtf_wall": 0.200,
                    "avg_batch_size": 12.0,
                    "queue_wait_ms": {"p95": 4.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 100.0}},
            },
        }

        recommendation = recommend_server_sweep_config(
            [slow_balanced, balanced, throughput],
            balanced_rtf_tolerance=0.05,
            high_queue_wait_ms=100.0,
        )

        self.assertEqual(
            recommendation["throughput_best"]["server_profile"]["name"],
            "throughput",
        )
        self.assertEqual(
            recommendation["balanced_best"]["server_profile"]["name"],
            "balanced",
        )
        self.assertIn("Use balanced_best", recommendation["next_actions"][0])

    def test_server_sweep_recommendation_rejects_low_num_step_for_production(self):
        low_step = {
            "server_profile": {"name": "fast-low-step", "batch_size": 12, "num_step": 2},
            "best": {
                "compact": {
                    "profile": {"name": "c12"},
                    "num_requests": 12,
                    "num_successful": 12,
                    "rtf_wall": 0.02,
                    "avg_batch_size": 12.0,
                    "queue_wait_ms": {"p95": 4.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 100.0}},
            },
        }
        production_step = {
            "server_profile": {
                "name": "production-step",
                "batch_size": 12,
                "num_step": 32,
            },
            "best": {
                "compact": {
                    "profile": {"name": "c12"},
                    "num_requests": 12,
                    "num_successful": 12,
                    "rtf_wall": 0.25,
                    "avg_batch_size": 12.0,
                    "queue_wait_ms": {"p95": 4.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 100.0}},
            },
        }

        low_step_diag = diagnose_server_sweep_result(low_step)
        recommendation = recommend_server_sweep_config(
            [low_step, production_step],
            balanced_rtf_tolerance=20.0,
        )

        self.assertIn(
            "quality_probe_num_step",
            [issue["code"] for issue in low_step_diag["issues"]],
        )
        self.assertEqual(
            recommendation["throughput_best"]["server_profile"]["name"],
            "fast-low-step",
        )
        self.assertEqual(
            recommendation["balanced_best"]["server_profile"]["name"],
            "production-step",
        )

    def test_server_sweep_recommendation_omits_balanced_action_when_same_profile(self):
        profile = {
            "server_profile": {"name": "balanced", "batch_size": 12, "num_step": 32},
            "best": {
                "compact": {
                    "profile": {"name": "c12"},
                    "num_requests": 12,
                    "num_successful": 12,
                    "rtf_wall": 0.25,
                    "avg_batch_size": 12.0,
                    "queue_wait_ms": {"p95": 4.0},
                    "split_retry_batches": 0,
                    "adaptive_batch_caps": {},
                }
            },
            "gpu": {
                "enabled": True,
                "summary": {"gpu_util_percent": {"max": 100.0}},
            },
        }

        recommendation = recommend_server_sweep_config([profile])

        self.assertEqual(
            recommendation["throughput_best"],
            recommendation["balanced_best"],
        )
        self.assertNotIn("Use balanced_best", recommendation["next_actions"][0])

    def test_server_sweep_parses_and_summarizes_gpu_samples(self):
        samples = _parse_nvidia_smi_gpu_rows(
            "0, 10, 1024, 24576\n"
            "1, 80, 4096, 24576\n"
            "bad,row\n"
            "2, [Not Supported], N/A, 24576\n"
        )

        self.assertEqual([sample["index"] for sample in samples], [0, 1, 2])
        self.assertEqual(samples[0]["gpu_util_percent"], 10.0)
        self.assertIsNone(samples[2]["gpu_util_percent"])
        self.assertIsNone(samples[2]["memory_used_mib"])

        summary = _summarize_gpu_samples(samples)

        self.assertEqual(summary["num_samples"], 3)
        self.assertEqual(summary["gpu_indexes"], [0, 1, 2])
        self.assertEqual(summary["gpu_util_percent"]["p50"], 45.0)
        self.assertEqual(summary["gpu_util_percent"]["max"], 80.0)
        self.assertEqual(summary["memory_used_mib"]["max"], 4096.0)
        self.assertEqual(summary["memory_total_mib"], 24576.0)

    def test_server_sweep_builds_server_command_and_client_args(self):
        args = argparse.Namespace(
            python="/venv/bin/python",
            model="/models/OmniVoice",
            host="127.0.0.1",
            port=18080,
            device="cuda",
            dtype="float16",
            compile_llm=True,
            compile_audio_heads=False,
            compile_mode="default",
            allow_reduce_overhead_worker=False,
            matmul_precision="high",
            partial_batch_floor=2,
            ready_queue_capacity=128,
            control_queue_capacity=16,
            prompt_cache_entries=256,
            use_model_duration_estimator=True,
            lookahead_for_full_batch=True,
            lookahead_for_partial_batch=True,
            partial_lookahead_max_wait_multiplier=2.5,
            max_seed_lookahead=64,
            candidate_pack_policy="target_context",
            split_retry_on_memory_error=True,
            adaptive_memory_batch_cap=True,
            adaptive_memory_cap_recovery_successes=64,
            max_generation_batches_before_control=8,
            max_total_context_tokens=5000,
            max_context_ratio=1.6,
            max_context_padding_ratio=1.7,
            generation_mode="optimized",
            guidance_scale=2.0,
            t_shift=0.1,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
            layer_penalty_factor=5.0,
            position_temperature=5.0,
            class_temperature=0.0,
            audio_chunk_duration=15.0,
            audio_chunk_threshold=30.0,
            batched_decode=True,
            enforce_output_duration=True,
            batch_size_pad=16,
            seq_len_bucket_multiple=64,
            target_len_bucket_multiple=32,
            collect_profile=True,
            reuse_static_input_embeds=True,
            split_guidance_forward="auto",
            split_guidance_min_batch_size=8,
            split_guidance_min_saved_context_ratio=0.25,
            warmup_test_list="/tmp/warmup.jsonl",
            warmup_batches=2,
            warmup_fill_batch=True,
            max_request_text_chars=2000,
            max_voice_prompts=512,
            server_log_level="warning",
            test_list="/tmp/test.jsonl",
            res_dir="/tmp/results",
            client_concurrency_values="4,8",
            client_arrival_gap_ms_values="0",
            client_repeats=1,
            client_warmup_repeats=1,
            client_request_repeats=3,
            client_timeout_s=30.0,
            save_wavs=False,
            client_pre_register_voices=True,
        )
        profile = {
            "batch_size": 8,
            "max_wait_ms": 40.0,
            "max_cost_ratio": 1.4,
            "max_total_target_tokens": 4096,
            "max_total_context_tokens": 6000,
            "max_context_ratio": 1.8,
            "max_context_padding_ratio": 1.9,
            "lookahead_for_partial_batch": False,
            "partial_lookahead_max_wait_multiplier": 1.5,
            "num_step": 2,
        }

        command = _build_server_command(args, profile)
        http_args = _http_sweep_args_for_server(
            args,
            server_profile_dir=Path("/tmp/results/profile"),
        )

        self.assertEqual(command[:3], ["/venv/bin/python", "-m", "omnivoice.cli.serve_online_batch"])
        self.assertIn("--device", command)
        self.assertIn("cuda", command)
        self.assertIn("--warmup_test_list", command)
        self.assertIn("/tmp/warmup.jsonl", command)
        self.assertEqual(command[command.index("--warmup_fill_batch") + 1], "true")
        self.assertEqual(command[command.index("--max_voice_prompts") + 1], "512")
        self.assertEqual(command[command.index("--batch_size") + 1], "8")
        self.assertEqual(command[command.index("--max_wait_ms") + 1], "40.0")
        self.assertEqual(
            command[command.index("--split_retry_on_memory_error") + 1],
            "true",
        )
        self.assertEqual(command[command.index("--adaptive_memory_batch_cap") + 1], "true")
        self.assertEqual(
            command[command.index("--adaptive_memory_cap_recovery_successes") + 1],
            "64",
        )
        self.assertEqual(
            command[command.index("--max_generation_batches_before_control") + 1],
            "8",
        )
        self.assertEqual(
            command[command.index("--use_model_duration_estimator") + 1],
            "true",
        )
        self.assertEqual(
            command[command.index("--lookahead_for_partial_batch") + 1],
            "false",
        )
        self.assertEqual(
            command[command.index("--partial_lookahead_max_wait_multiplier") + 1],
            "1.5",
        )
        self.assertEqual(
            command[command.index("--candidate_pack_policy") + 1],
            "target_context",
        )
        self.assertEqual(command[command.index("--max_total_context_tokens") + 1], "6000")
        self.assertEqual(command[command.index("--max_context_ratio") + 1], "1.8")
        self.assertEqual(
            command[command.index("--max_context_padding_ratio") + 1],
            "1.9",
        )
        self.assertEqual(command[command.index("--generation_mode") + 1], "optimized")
        self.assertEqual(command[command.index("--num_step") + 1], "2")
        self.assertEqual(command[command.index("--batch_size_pad") + 1], "16")
        self.assertEqual(command[command.index("--seq_len_bucket_multiple") + 1], "64")
        self.assertEqual(command[command.index("--target_len_bucket_multiple") + 1], "32")
        self.assertEqual(command[command.index("--collect_profile") + 1], "true")
        self.assertEqual(command[command.index("--enforce_output_duration") + 1], "true")
        self.assertEqual(command[command.index("--reuse_static_input_embeds") + 1], "true")
        self.assertEqual(command[command.index("--split_guidance_forward") + 1], "auto")
        self.assertEqual(command[command.index("--split_guidance_min_batch_size") + 1], "8")
        self.assertEqual(
            command[command.index("--split_guidance_min_saved_context_ratio") + 1],
            "0.25",
        )
        self.assertEqual(http_args.url, "http://127.0.0.1:18080/v1/tts")
        self.assertEqual(http_args.scheduler_url, "http://127.0.0.1:18080/v1/scheduler")
        self.assertEqual(http_args.res_dir, "/tmp/results/profile/client_sweep")
        self.assertEqual(http_args.concurrency_values, "4,8")
        self.assertEqual(http_args.warmup_repeats, 1)
        self.assertEqual(http_args.request_repeats, 3)
        self.assertTrue(http_args.pre_register_voices)

    def test_heterogeneous_generator_uses_outlier_first_groups(self):
        args = argparse.Namespace(
            groups=2,
            shorts_per_group=3,
            long_duration=2.4,
            short_duration=0.8,
            long_text_repeat=1,
            long_ref_text_repeat=1,
            ref_audio="/tmp/ref.wav",
            ref_text="参考文本",
            language_id="zh",
            id_prefix="mix",
        )

        samples = build_samples(args)

        self.assertEqual(len(samples), 8)
        self.assertEqual(samples[0]["id"], "mix_g01_long")
        self.assertEqual(samples[1]["id"], "mix_g01_short01")
        self.assertEqual(samples[4]["id"], "mix_g02_long")
        self.assertEqual(samples[0]["duration"], 2.4)
        self.assertEqual(samples[1]["duration"], 0.8)
        self.assertEqual({sample["ref_audio"] for sample in samples}, {"/tmp/ref.wav"})

    def test_infer_batch_merges_sample_strict_duration_with_cli_default(self):
        class FakeBatchModel:
            sampling_rate = 1000

            def __init__(self):
                self.kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [
                    np.zeros(10, dtype=np.float32)
                    for _ in kwargs["text"]
                ]

        model = FakeBatchModel()
        original_model = infer_batch_cli.worker_model
        infer_batch_cli.worker_model = model
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                results = infer_batch_cli.run_inference_batch(
                    [
                        (
                            "a",
                            None,
                            None,
                            "first",
                            "en",
                            0.01,
                            None,
                            None,
                            False,
                        ),
                        (
                            "b",
                            None,
                            None,
                            "second",
                            "en",
                            0.02,
                            None,
                            None,
                            None,
                        ),
                    ],
                    tmpdir,
                    enforce_output_duration=True,
                    postprocess_output=False,
                )
        finally:
            infer_batch_cli.worker_model = original_model

        self.assertEqual(
            model.kwargs["enforce_output_duration"],
            [False, True],
        )
        self.assertEqual([result[0] for result in results], ["a", "b"])

    def test_heterogeneous_generator_can_create_context_outliers(self):
        args = argparse.Namespace(
            groups=1,
            shorts_per_group=2,
            long_duration=1.2,
            short_duration=1.2,
            long_text_repeat=4,
            long_ref_text_repeat=3,
            ref_audio="/tmp/ref.wav",
            ref_text="参考文本",
            language_id="zh",
            id_prefix="ctx",
        )

        samples = build_samples(args)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["id"], "ctx_g01_long")
        self.assertEqual(samples[0]["duration"], samples[1]["duration"])
        self.assertGreater(len(samples[0]["text"]), len(samples[1]["text"]) * 4)
        self.assertEqual(samples[0]["ref_text"], "参考文本 参考文本 参考文本")
        self.assertEqual(samples[1]["ref_text"], "参考文本")

    def test_scheduler_aware_warmup_selects_cost_bucket(self):
        args = argparse.Namespace(
            groups=1,
            shorts_per_group=4,
            long_duration=2.4,
            short_duration=0.8,
            long_text_repeat=1,
            long_ref_text_repeat=1,
            ref_audio="/tmp/ref.wav",
            ref_text="参考文本",
            language_id="zh",
            id_prefix="mix",
        )
        samples = build_samples(args)

        selected = _select_representative_warmup_samples(
            samples,
            BatchSchedulerConfig(
                max_batch_size=4,
                max_cost_ratio=1.4,
                max_total_target_tokens=4096,
                lookahead_for_full_batch=True,
            ),
        )

        self.assertEqual(
            [sample["id"] for sample in selected],
            [
                "mix_g01_short01",
                "mix_g01_short02",
                "mix_g01_short03",
                "mix_g01_short04",
            ],
        )

    def test_scheduler_aware_warmup_preserves_fifo_without_lookahead(self):
        args = argparse.Namespace(
            groups=1,
            shorts_per_group=4,
            long_duration=2.4,
            short_duration=0.8,
            long_text_repeat=1,
            long_ref_text_repeat=1,
            ref_audio="/tmp/ref.wav",
            ref_text="参考文本",
            language_id="zh",
            id_prefix="mix",
        )
        samples = build_samples(args)

        selected = _select_representative_warmup_samples(
            samples,
            BatchSchedulerConfig(
                max_batch_size=4,
                max_cost_ratio=1.4,
                max_total_target_tokens=4096,
                lookahead_for_full_batch=False,
            ),
        )

        self.assertEqual(selected[0]["id"], "mix_g01_long")

    def test_scheduler_aware_warmup_can_select_multiple_batches(self):
        args = argparse.Namespace(
            groups=2,
            shorts_per_group=4,
            long_duration=2.4,
            short_duration=0.8,
            long_text_repeat=1,
            long_ref_text_repeat=1,
            ref_audio="/tmp/ref.wav",
            ref_text="参考文本",
            language_id="zh",
            id_prefix="mix",
        )
        samples = build_samples(args)

        batches = _select_representative_warmup_batches(
            samples,
            BatchSchedulerConfig(
                max_batch_size=4,
                max_cost_ratio=1.4,
                max_total_target_tokens=4096,
                lookahead_for_full_batch=True,
            ),
            max_batches=3,
        )

        self.assertEqual(
            [[sample["id"] for sample in batch] for batch in batches],
            [
                [
                    "mix_g01_short01",
                    "mix_g01_short02",
                    "mix_g01_short03",
                    "mix_g01_short04",
                ],
                [
                    "mix_g02_short01",
                    "mix_g02_short02",
                    "mix_g02_short03",
                    "mix_g02_short04",
                ],
                ["mix_g01_long", "mix_g02_long"],
            ],
        )

    def test_scheduler_aware_warmup_can_fill_selected_batch(self):
        samples = [
            {
                "id": "short-a",
                "text": "short",
                "duration": 1.0,
            },
            {
                "id": "short-b",
                "text": "short",
                "duration": 1.0,
            },
        ]

        batches = _select_representative_warmup_batches(
            samples,
            BatchSchedulerConfig(
                max_batch_size=4,
                max_cost_ratio=1.4,
                max_total_target_tokens=4096,
                lookahead_for_full_batch=True,
            ),
            max_batches=1,
            fill_batch=True,
        )

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 4)
        self.assertEqual(
            [sample["id"] for sample in batches[0]],
            [
                "short-a",
                "short-b",
                "short-a__warmup_fill_3",
                "short-b__warmup_fill_4",
            ],
        )

    def test_online_request_from_sample_preserves_strict_duration_flag(self):
        request = _online_request_from_sample(
            {
                "id": "r1",
                "text": "hello",
                "language_id": "en",
                "duration": 1.2,
                "enforce_output_duration": True,
            }
        )

        self.assertEqual(request.request_id, "r1")
        self.assertEqual(request.duration, 1.2)
        self.assertTrue(request.enforce_output_duration)

    def test_scheduler_aware_warmup_fill_respects_target_token_cap(self):
        samples = [
            {
                "id": "short-a",
                "text": "short",
                "duration": 1.0,
            },
            {
                "id": "short-b",
                "text": "short",
                "duration": 1.0,
            },
        ]

        batches = _select_representative_warmup_batches(
            samples,
            BatchSchedulerConfig(
                max_batch_size=4,
                max_cost_ratio=1.4,
                max_total_target_tokens=50,
                max_total_context_tokens=4096,
                lookahead_for_full_batch=True,
            ),
            max_batches=1,
            fill_batch=True,
        )

        self.assertEqual(len(batches), 1)
        self.assertEqual([sample["id"] for sample in batches[0]], ["short-a", "short-b"])

    def test_threaded_compile_mode_downgrades_reduce_overhead_by_default(self):
        args = argparse.Namespace(
            compile_mode="reduce-overhead",
            allow_reduce_overhead_worker=False,
        )

        self.assertEqual(_online_effective_compile_mode(args), "default")
        self.assertEqual(_stepwise_effective_compile_mode(args), "default")

    def test_threaded_compile_mode_allows_explicit_reduce_overhead(self):
        args = argparse.Namespace(
            compile_mode="reduce-overhead",
            allow_reduce_overhead_worker=True,
        )

        self.assertEqual(_online_effective_compile_mode(args), "reduce-overhead")
        self.assertEqual(_stepwise_effective_compile_mode(args), "reduce-overhead")

    def test_online_parser_exposes_split_guidance_controls(self):
        defaults = get_online_parser().parse_args(
            ["--test_list", "/tmp/test.jsonl", "--res_dir", "/tmp/out"]
        )
        custom = get_online_parser().parse_args(
            [
                "--test_list",
                "/tmp/test.jsonl",
                "--res_dir",
                "/tmp/out",
                "--split_guidance_forward",
                "false",
                "--split_guidance_min_batch_size",
                "4",
                "--split_guidance_min_saved_context_ratio",
                "0.5",
            ]
        )

        self.assertEqual(defaults.split_guidance_forward, "auto")
        self.assertEqual(defaults.split_guidance_min_batch_size, 8)
        self.assertEqual(defaults.split_guidance_min_saved_context_ratio, 0.25)
        self.assertEqual(custom.split_guidance_forward, "false")
        self.assertEqual(custom.split_guidance_min_batch_size, 4)
        self.assertEqual(custom.split_guidance_min_saved_context_ratio, 0.5)

    def test_online_parser_exposes_full_scheduler_config_surface(self):
        args = get_online_parser().parse_args(
            [
                "--test_list",
                "/tmp/test.jsonl",
                "--res_dir",
                "/tmp/results",
                "--batch_size",
                "12",
                "--max_wait_ms",
                "20",
                "--max_total_target_tokens",
                "2048",
                "--max_total_context_tokens",
                "4096",
                "--max_cost_ratio",
                "1.4",
                "--max_context_ratio",
                "1.8",
                "--max_context_padding_ratio",
                "1.6",
                "--ready_queue_capacity",
                "128",
                "--control_queue_capacity",
                "4",
                "--prompt_cache_entries",
                "256",
                "--use_model_duration_estimator",
                "false",
                "--lookahead_for_full_batch",
                "false",
                "--lookahead_for_partial_batch",
                "true",
                "--partial_lookahead_max_wait_multiplier",
                "3.0",
                "--max_seed_lookahead",
                "64",
                "--candidate_pack_policy",
                "context",
                "--split_retry_on_memory_error",
                "false",
                "--adaptive_memory_batch_cap",
                "false",
                "--adaptive_memory_cap_recovery_successes",
                "7",
                "--max_generation_batches_before_control",
                "3",
            ]
        )

        config = _online_scheduler_config_from_args(args)

        self.assertEqual(config.max_batch_size, 12)
        self.assertEqual(config.max_wait_ms, 20.0)
        self.assertEqual(config.max_total_target_tokens, 2048)
        self.assertEqual(config.max_total_context_tokens, 4096)
        self.assertEqual(config.max_cost_ratio, 1.4)
        self.assertEqual(config.max_context_ratio, 1.8)
        self.assertEqual(config.max_context_padding_ratio, 1.6)
        self.assertEqual(config.ready_queue_capacity, 128)
        self.assertEqual(config.control_queue_capacity, 4)
        self.assertEqual(config.prompt_cache_entries, 256)
        self.assertFalse(config.use_model_duration_estimator)
        self.assertFalse(config.lookahead_for_full_batch)
        self.assertTrue(config.lookahead_for_partial_batch)
        self.assertEqual(config.partial_lookahead_max_wait_multiplier, 3.0)
        self.assertEqual(config.max_seed_lookahead, 64)
        self.assertEqual(config.candidate_pack_policy, "context")
        self.assertFalse(config.split_retry_on_memory_error)
        self.assertFalse(config.adaptive_memory_batch_cap)
        self.assertEqual(config.adaptive_memory_cap_recovery_successes, 7)
        self.assertEqual(config.max_generation_batches_before_control, 3)

    def test_stepwise_parser_exposes_context_admission_controls(self):
        args = get_stepwise_parser().parse_args(
            [
                "--test_list",
                "/tmp/test.jsonl",
                "--res_dir",
                "/tmp/out",
                "--max_total_context_tokens",
                "4096",
                "--max_context_ratio",
                "1.5",
                "--max_context_padding_ratio",
                "1.75",
            ]
        )

        self.assertEqual(args.max_total_context_tokens, 4096)
        self.assertEqual(args.max_context_ratio, 1.5)
        self.assertEqual(args.max_context_padding_ratio, 1.75)

    def test_stepwise_config_helpers_cover_supported_cli_surface(self):
        args = get_stepwise_parser().parse_args(
            [
                "--test_list",
                "/tmp/test.jsonl",
                "--res_dir",
                "/tmp/out",
                "--max_running_requests",
                "12",
                "--max_wait_ms",
                "25",
                "--partial_batch_floor",
                "3",
                "--max_total_target_tokens",
                "2048",
                "--max_total_context_tokens",
                "4096",
                "--max_cost_ratio",
                "1.5",
                "--max_context_ratio",
                "1.6",
                "--max_context_padding_ratio",
                "1.7",
                "--ready_queue_capacity",
                "96",
                "--control_queue_capacity",
                "5",
                "--prompt_cache_entries",
                "128",
                "--profile_cuda",
                "true",
                "--compile_static_shape",
                "true",
                "--seq_len_bucket_multiple",
                "32",
                "--target_len_bucket_multiple",
                "16",
                "--lookahead_for_full_batch",
                "false",
                "--max_seed_lookahead",
                "48",
                "--generation_mode",
                "official_compatible",
                "--num_step",
                "16",
                "--guidance_scale",
                "1.7",
                "--t_shift",
                "0.2",
                "--denoise",
                "false",
                "--preprocess_prompt",
                "false",
                "--postprocess_output",
                "false",
                "--layer_penalty_factor",
                "4",
                "--position_temperature",
                "3",
                "--class_temperature",
                "0.1",
                "--audio_chunk_duration",
                "8",
                "--audio_chunk_threshold",
                "16",
                "--enforce_output_duration",
                "true",
            ]
        )

        scheduler_config = _stepwise_scheduler_config_from_args(args)
        generation_config = _stepwise_generation_config_from_args(args)

        self.assertEqual(scheduler_config.max_running_requests, 12)
        self.assertEqual(scheduler_config.max_wait_ms, 25.0)
        self.assertEqual(scheduler_config.partial_batch_floor, 3)
        self.assertEqual(scheduler_config.max_total_target_tokens, 2048)
        self.assertEqual(scheduler_config.max_total_context_tokens, 4096)
        self.assertEqual(scheduler_config.max_cost_ratio, 1.5)
        self.assertEqual(scheduler_config.max_context_ratio, 1.6)
        self.assertEqual(scheduler_config.max_context_padding_ratio, 1.7)
        self.assertEqual(scheduler_config.ready_queue_capacity, 96)
        self.assertEqual(scheduler_config.control_queue_capacity, 5)
        self.assertEqual(scheduler_config.prompt_cache_entries, 128)
        self.assertTrue(scheduler_config.profile_cuda)
        self.assertTrue(scheduler_config.compile_static_shape)
        self.assertEqual(scheduler_config.seq_len_bucket_multiple, 32)
        self.assertEqual(scheduler_config.target_len_bucket_multiple, 16)
        self.assertFalse(scheduler_config.lookahead_for_full_batch)
        self.assertEqual(scheduler_config.max_seed_lookahead, 48)

        self.assertEqual(generation_config.generation_mode, "official_compatible")
        self.assertEqual(generation_config.num_step, 16)
        self.assertEqual(generation_config.guidance_scale, 1.7)
        self.assertEqual(generation_config.t_shift, 0.2)
        self.assertFalse(generation_config.denoise)
        self.assertFalse(generation_config.preprocess_prompt)
        self.assertFalse(generation_config.postprocess_output)
        self.assertEqual(generation_config.layer_penalty_factor, 4.0)
        self.assertEqual(generation_config.position_temperature, 3.0)
        self.assertEqual(generation_config.class_temperature, 0.1)
        self.assertEqual(generation_config.audio_chunk_duration, 8.0)
        self.assertEqual(generation_config.audio_chunk_threshold, 16.0)
        self.assertTrue(generation_config.enforce_output_duration)

    def test_scheduler_warmup_submits_batches_and_resets_metrics(self):
        class FakeScheduler:
            def __init__(self):
                self.submitted = []
                self.reset_called = False

            async def submit(self, request):
                self.submitted.append(request.request_id)

            def reset_metrics(self):
                self.reset_called = True

        scheduler = FakeScheduler()

        asyncio.run(
            _run_scheduler_warmup(
                scheduler,
                [
                    [
                        {"id": "a", "text": "A"},
                        {"id": "b", "text": "B"},
                    ],
                    [{"id": "c", "text": "C"}],
                ],
            )
        )

        self.assertEqual(scheduler.submitted, ["a", "b", "c"])
        self.assertTrue(scheduler.reset_called)

    def test_http_payload_uses_test_list_fields(self):
        payload = _payload_from_sample(
            {
                "id": "r1",
                "text": "你好",
                "language_id": "zh",
                "voice_id": "speaker-a",
                "ref_audio": "/tmp/ref.wav",
                "ref_audio_base64": "abc",
                "ref_text": "参考",
                "duration": 1.2,
                "speed": 1.1,
                "enforce_output_duration": True,
                "cost_tokens_hint": 88,
                "priority": "high",
            }
        )

        self.assertEqual(payload["request_id"], "r1")
        self.assertEqual(payload["language_id"], "zh")
        self.assertEqual(payload["voice_id"], "speaker-a")
        self.assertEqual(payload["ref_audio"], "/tmp/ref.wav")
        self.assertEqual(payload["ref_audio_base64"], "abc")
        self.assertEqual(payload["duration"], 1.2)
        self.assertTrue(payload["enforce_output_duration"])
        self.assertEqual(payload["cost_tokens_hint"], 88)
        self.assertEqual(payload["priority"], "high")
        self.assertNotIn("instruct", payload)

    def test_voice_pre_registration_helpers_group_and_rewrite_samples(self):
        first = {
            "id": "r1",
            "text": "你好",
            "ref_audio": "/tmp/ref.wav",
            "ref_text": "参考",
            "language_id": "zh",
            "duration": 1.2,
            "preprocess_prompt": False,
        }
        second = {
            "id": "r2",
            "text": "第二句",
            "ref_audio": "/tmp/ref.wav",
            "ref_text": "参考",
            "language_id": "zh",
        }
        key = _voice_registration_key(first)
        voice_id = _voice_id_for_registration_key(key)

        payload = _voice_registration_payload(first, voice_id=voice_id)
        rewritten = _sample_with_registered_voice_id(first, voice_id=voice_id)

        self.assertEqual(
            _voice_register_url_from_tts_url("http://127.0.0.1:8000/v1/tts"),
            "http://127.0.0.1:8000/v1/voices",
        )
        self.assertEqual(_voice_registration_key(second), key)
        self.assertTrue(voice_id.startswith("bench_voice_"))
        self.assertEqual(payload["voice_id"], voice_id)
        self.assertEqual(payload["ref_audio"], "/tmp/ref.wav")
        self.assertFalse(payload["preprocess_prompt"])
        self.assertEqual(rewritten["voice_id"], voice_id)
        self.assertNotIn("ref_audio", rewritten)
        self.assertNotIn("ref_text", rewritten)
        self.assertEqual(rewritten["duration"], 1.2)

    def test_pre_register_voices_posts_once_per_unique_voice(self):
        class FakeResponse:
            def __init__(self, payload):
                self.status_code = 200
                self._payload = payload
                self.text = json.dumps(payload)

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self):
                self.posts = []

            async def post(self, url, json):
                self.posts.append({"url": url, "json": json})
                return FakeResponse({"voice_id": json["voice_id"], "registered": True})

        samples = [
            {
                "id": "r1",
                "text": "你好",
                "ref_audio": "/tmp/ref.wav",
                "ref_text": "参考",
            },
            {
                "id": "r2",
                "text": "第二句",
                "ref_audio": "/tmp/ref.wav",
                "ref_text": "参考",
            },
            {
                "id": "r3",
                "text": "已有 voice id",
                "voice_id": "already-registered",
            },
        ]
        client = FakeClient()

        rewritten, report = asyncio.run(
            _pre_register_voices(
                client=client,
                voice_register_url="http://127.0.0.1:8000/v1/voices",
                samples=samples,
            )
        )

        self.assertEqual(len(client.posts), 1)
        self.assertEqual(report["num_registered_voices"], 1)
        self.assertEqual(report["num_rewritten_samples"], 2)
        self.assertEqual(rewritten[0]["voice_id"], rewritten[1]["voice_id"])
        self.assertEqual(rewritten[2]["voice_id"], "already-registered")
        self.assertNotIn("ref_audio", rewritten[0])

    def test_stepwise_local_voice_pre_registration_reuses_prompts(self):
        class FakeScheduler:
            def __init__(self):
                self.calls = []

            async def create_voice_clone_prompt(
                self,
                *,
                ref_audio,
                ref_text=None,
                preprocess_prompt=None,
            ):
                self.calls.append(
                    {
                        "ref_audio": ref_audio,
                        "ref_text": ref_text,
                        "preprocess_prompt": preprocess_prompt,
                    }
                )
                return f"prompt-{len(self.calls)}"

        samples = [
            {
                "id": "r1",
                "text": "你好",
                "ref_audio": "/tmp/ref.wav",
                "ref_text": "参考",
            },
            {
                "id": "r2",
                "text": "第二句",
                "ref_audio": "/tmp/ref.wav",
                "ref_text": "参考",
            },
        ]
        scheduler = FakeScheduler()

        key = _local_voice_registration_key(samples[0], preprocess_prompt=True)
        rewritten_sample = _sample_with_local_voice_prompt(
            samples[0],
            voice_id="local_voice_0000",
            voice_clone_prompt="prompt",
        )
        rewritten, report = asyncio.run(
            _pre_register_local_voice_prompts(
                scheduler=scheduler,
                samples=samples,
                preprocess_prompt=True,
            )
        )
        request = _stepwise_request_from_sample(rewritten[0])

        self.assertEqual(key, ("/tmp/ref.wav", "参考", True))
        self.assertEqual(rewritten_sample["voice_clone_prompt"], "prompt")
        self.assertNotIn("ref_audio", rewritten_sample)
        self.assertEqual(len(scheduler.calls), 1)
        self.assertEqual(report["num_registered_voices"], 1)
        self.assertEqual(report["num_rewritten_samples"], 2)
        self.assertEqual(rewritten[0]["voice_clone_prompt"], rewritten[1]["voice_clone_prompt"])
        self.assertEqual(request.voice_clone_prompt, "prompt-1")
        self.assertIsNone(request.ref_audio)

    def test_stepwise_request_from_sample_preserves_strict_duration_flag(self):
        request = _stepwise_request_from_sample(
            {
                "id": "r1",
                "text": "hello",
                "duration": 1.2,
                "enforce_output_duration": False,
            }
        )

        self.assertEqual(request.duration, 1.2)
        self.assertFalse(request.enforce_output_duration)

    def test_read_test_list_preserves_serving_fields(self):
        row = {
            "id": "r1",
            "text": "hello",
            "voice_id": "speaker-a",
            "ref_audio_base64": "abc",
            "priority": "high",
            "cost_tokens_hint": 88,
            "context_tokens_hint": 188,
            "preprocess_prompt": False,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            samples = read_test_list(path)

        self.assertEqual(samples[0]["voice_id"], "speaker-a")
        self.assertEqual(samples[0]["ref_audio_base64"], "abc")
        self.assertEqual(samples[0]["priority"], "high")
        self.assertEqual(samples[0]["cost_tokens_hint"], 88)
        self.assertEqual(samples[0]["context_tokens_hint"], 188)
        self.assertFalse(samples[0]["preprocess_prompt"])

    def test_http_result_parses_headers_and_wav_duration(self):
        audio = np.zeros(2400, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV")

        class FakeResponse:
            status_code = 200
            content = buf.getvalue()
            text = ""
            headers = {
                "X-OmniVoice-Batch-Size": "4",
                "X-OmniVoice-Queue-Wait-Ms": "12.5",
                "X-OmniVoice-Batch-Infer-S": "0.25",
                "X-OmniVoice-Batch-Reason": "full",
                "X-OmniVoice-Batch-Cost-Tokens": "360",
                "X-OmniVoice-Batch-Max-Cost-Tokens": "90",
                "X-OmniVoice-Batch-Context-Tokens": "720",
                "X-OmniVoice-Batch-Max-Context-Tokens": "180",
                "X-OmniVoice-Batch-Context-Padding-Ratio": "1.25",
                "X-OmniVoice-Generation-Profile": (
                    '{"timings_s":{"total_s":1.0}}'
                ),
                "X-OmniVoice-Total-S": "0.30",
            }

        row = _result_from_response(
            request_id="r1",
            response=FakeResponse(),
            request_wall_s=0.31,
        )

        self.assertEqual(row["batch_size"], 4)
        self.assertEqual(row["queue_wait_ms"], 12.5)
        self.assertEqual(row["batch_infer_s"], 0.25)
        self.assertEqual(row["batch_reason"], "full")
        self.assertEqual(row["batch_cost_tokens"], 360)
        self.assertEqual(row["batch_max_cost_tokens"], 90)
        self.assertEqual(row["batch_context_tokens"], 720)
        self.assertEqual(row["batch_max_context_tokens"], 180)
        self.assertEqual(row["batch_context_padding_ratio"], 1.25)
        self.assertEqual(row["generation_profile"]["timings_s"]["total_s"], 1.0)
        self.assertAlmostEqual(row["audio_s"], 0.1)
        self.assertAlmostEqual(_audio_seconds_from_wav(buf.getvalue()), 0.1)


if __name__ == "__main__":
    unittest.main()
