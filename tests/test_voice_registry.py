import unittest

from omnivoice.serving.voice_registry import VoicePromptRegistry


class VoicePromptRegistryTests(unittest.TestCase):
    def test_get_refreshes_lru_order_before_eviction(self):
        registry = VoicePromptRegistry(max_entries=2)
        registry["a"] = object()
        registry["b"] = object()

        self.assertIsNotNone(registry.get("a"))
        registry["c"] = object()

        snapshot = registry.snapshot()
        self.assertEqual(snapshot.voice_ids, ["a", "c"])
        self.assertEqual(snapshot.evictions, 1)

    def test_unbounded_registry_never_evicts(self):
        registry = VoicePromptRegistry(max_entries=None)
        registry["a"] = object()
        registry["b"] = object()
        registry["c"] = object()

        snapshot = registry.snapshot()
        self.assertEqual(snapshot.size, 3)
        self.assertIsNone(snapshot.max_entries)
        self.assertEqual(snapshot.evictions, 0)

    def test_rejects_non_positive_capacity(self):
        with self.assertRaisesRegex(ValueError, "max_entries"):
            VoicePromptRegistry(max_entries=0)


if __name__ == "__main__":
    unittest.main()

