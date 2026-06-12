import unittest

from omnivoice.cli.benchmark_scheduler_packing import (
    _compare_to_baseline,
    simulate_packing,
)
from omnivoice.serving import BatchSchedulerConfig


class SchedulerPackingBenchmarkTests(unittest.TestCase):
    def _config(self, policy):
        return BatchSchedulerConfig(
            max_batch_size=4,
            max_wait_ms=10_000.0,
            partial_batch_floor=1,
            max_total_target_tokens=10_000,
            max_total_context_tokens=1000,
            max_cost_ratio=2.0,
            max_context_ratio=10.0,
            max_context_padding_ratio=0.0,
            ready_queue_capacity=16,
            use_model_duration_estimator=False,
            lookahead_for_full_batch=True,
            lookahead_for_partial_batch=True,
            max_seed_lookahead=1,
            candidate_pack_policy=policy,
            adaptive_memory_batch_cap=False,
        )

    def test_target_context_policy_reduces_context_padding_work(self):
        samples = [
            {
                "id": "seed",
                "text": "seed",
                "cost_tokens_hint": 100,
                "context_tokens_hint": 100,
            },
            {
                "id": "target-near-context-heavy",
                "text": "heavy",
                "cost_tokens_hint": 101,
                "context_tokens_hint": 800,
            },
            {
                "id": "fit-1",
                "text": "fit",
                "cost_tokens_hint": 130,
                "context_tokens_hint": 100,
            },
            {
                "id": "fit-2",
                "text": "fit",
                "cost_tokens_hint": 131,
                "context_tokens_hint": 100,
            },
            {
                "id": "fit-3",
                "text": "fit",
                "cost_tokens_hint": 132,
                "context_tokens_hint": 100,
            },
        ]

        target = simulate_packing(
            samples,
            config=self._config("target"),
            include_batches=True,
        )
        target_context = simulate_packing(
            samples,
            config=self._config("target_context"),
            include_batches=True,
        )
        comparisons = _compare_to_baseline([target, target_context])

        self.assertEqual(
            target["batches"][0]["request_ids"],
            ["seed", "target-near-context-heavy", "fit-1"],
        )
        self.assertEqual(
            target_context["batches"][0]["request_ids"],
            ["seed", "fit-1", "fit-2", "fit-3"],
        )
        self.assertEqual(target["full_batch_count"], 0)
        self.assertEqual(target_context["full_batch_count"], 1)
        self.assertLess(
            target_context["context_padding_work"],
            target["context_padding_work"],
        )
        self.assertLess(comparisons[1]["context_padding_work_pct_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
