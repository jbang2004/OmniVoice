import types
import unittest

import numpy as np
import torch

from omnivoice.models.omnivoice import (
    GenerationTask,
    OmniVoice,
    OmniVoiceGenerationConfig as CompatOmniVoiceGenerationConfig,
    VoiceClonePrompt,
)
from omnivoice.models.generation import OmniVoiceGenerationConfig, fit_audio_to_duration


def _bare_model():
    model = object.__new__(OmniVoice)
    model.audio_tokenizer = types.SimpleNamespace(
        config=types.SimpleNamespace(frame_rate=25)
    )
    model._estimate_target_tokens = lambda *args, **kwargs: 16
    return model


class OmniVoicePreprocessTests(unittest.TestCase):
    def test_generation_config_is_still_reexported_from_omnivoice_module(self):
        self.assertIs(CompatOmniVoiceGenerationConfig, OmniVoiceGenerationConfig)

    def test_generation_mode_presets_are_applied(self):
        official = OmniVoiceGenerationConfig(
            generation_mode="official_compatible",
            batched_decode=True,
            reuse_static_input_embeds=True,
            split_guidance_forward="auto",
            seq_len_bucket_multiple=64,
            target_len_bucket_multiple=32,
        )

        self.assertEqual(official.generation_mode, "official_compatible")
        self.assertFalse(official.batched_decode)
        self.assertFalse(official.reuse_static_input_embeds)
        self.assertFalse(official.split_guidance_forward)
        self.assertEqual(official.seq_len_bucket_multiple, 1)
        self.assertEqual(official.target_len_bucket_multiple, 1)

        optimized = OmniVoiceGenerationConfig(generation_mode="throughput")

        self.assertEqual(optimized.generation_mode, "optimized")
        self.assertTrue(optimized.batched_decode)
        self.assertTrue(optimized.reuse_static_input_embeds)
        self.assertEqual(optimized.split_guidance_forward, "auto")

    def test_generation_mode_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "Unknown generation_mode"):
            OmniVoiceGenerationConfig(generation_mode="mystery")

    def test_fit_audio_to_duration_pads_and_crops_last_axis(self):
        mono = np.arange(4, dtype=np.float32)
        padded = fit_audio_to_duration(mono, 0.006, sample_rate=1000)
        np.testing.assert_allclose(padded, np.array([0, 1, 2, 3, 0, 0], dtype=np.float32))

        stereo = np.arange(12, dtype=np.float32).reshape(2, 6)
        cropped = fit_audio_to_duration(stereo, 0.004, sample_rate=1000)
        self.assertEqual(cropped.shape, (2, 4))
        np.testing.assert_allclose(cropped, stereo[:, :4])

    def test_preprocess_records_requested_durations_for_final_enforcement(self):
        model = _bare_model()

        task = model._preprocess_all(
            text=["first", "second"],
            duration=[2.0, None],
            speed=[1.0, 1.5],
        )

        self.assertEqual(task.target_lens, [50, 16])
        self.assertEqual(task.requested_durations, [2.0, None])
        self.assertEqual(task.speed, [16 / 50, 1.5])

        sliced = task.slice_task([1])
        self.assertIsNotNone(sliced)
        self.assertEqual(sliced.requested_durations, [None])

    def test_preprocess_broadcasts_single_item_duration_and_speed_lists(self):
        model = _bare_model()

        task = model._preprocess_all(
            text=["first", "second"],
            duration=[2.0],
            speed=[1.5],
        )

        self.assertEqual(task.target_lens, [50, 50])
        self.assertEqual(task.requested_durations, [2.0, 2.0])
        self.assertEqual(task.speed, [16 / 50, 16 / 50])

    def test_preprocess_rejects_misaligned_duration_and_speed_lists(self):
        model = _bare_model()

        with self.assertRaisesRegex(ValueError, "duration.*batch size 2"):
            model._preprocess_all(
                text=["first", "second"],
                duration=[1.0, 2.0, 3.0],
            )

        with self.assertRaisesRegex(ValueError, "speed.*batch size 2"):
            model._preprocess_all(
                text=["first", "second"],
                speed=[1.0, 1.2, 1.5],
            )

    def test_preprocess_rejects_non_positive_duration_and_speed(self):
        model = _bare_model()

        with self.assertRaisesRegex(ValueError, "duration values must be positive"):
            model._preprocess_all(text="hello", duration=0)

        with self.assertRaisesRegex(ValueError, "speed values must be positive"):
            model._preprocess_all(text="hello", speed=-1)

    def test_enforce_output_duration_flags_support_per_item_override(self):
        model = _bare_model()

        self.assertEqual(
            model._resolve_enforce_output_duration_flags(None, 2, default=True),
            [True, True],
        )
        self.assertEqual(
            model._resolve_enforce_output_duration_flags(
                [False, None],
                2,
                default=True,
            ),
            [False, True],
        )
        self.assertEqual(
            model._resolve_enforce_output_duration_flags([True], 2, default=False),
            [True, True],
        )

        with self.assertRaisesRegex(ValueError, "enforce_output_duration.*batch size 2"):
            model._resolve_enforce_output_duration_flags(
                [True, False, True],
                2,
                default=False,
            )
        with self.assertRaisesRegex(
            ValueError,
            "enforce_output_duration.*bool",
        ):
            model._resolve_enforce_output_duration_flags(
                ["false"],
                2,
                default=False,
            )
        with self.assertRaisesRegex(
            ValueError,
            "enforce_output_duration.*bool",
        ):
            model._resolve_enforce_output_duration_flags(
                "false",
                5,
                default=False,
            )

    def test_ref_text_none_expands_for_multiple_ref_audios(self):
        model = _bare_model()
        calls = []

        def create_voice_clone_prompt(ref_audio, ref_text, preprocess_prompt=True):
            calls.append((ref_audio, ref_text, preprocess_prompt))
            return VoiceClonePrompt(
                ref_audio_tokens=torch.zeros(8, len(calls), dtype=torch.long),
                ref_text=f"auto-{ref_audio}",
                ref_rms=0.1,
            )

        model.create_voice_clone_prompt = create_voice_clone_prompt

        task = model._preprocess_all(
            text=["first", "second"],
            language=["zh", "zh"],
            ref_audio=["ref-a.wav", "ref-b.wav"],
            ref_text=None,
        )

        self.assertEqual(
            calls,
            [
                ("ref-a.wav", None, True),
                ("ref-b.wav", None, True),
            ],
        )
        self.assertEqual(task.batch_size, 2)
        self.assertEqual(task.ref_texts, ["auto-ref-a.wav", "auto-ref-b.wav"])
        self.assertEqual([t.size(-1) for t in task.ref_audio_tokens], [1, 2])

    def test_duplicate_ref_audio_is_cached_within_batch(self):
        model = _bare_model()
        calls = []

        def create_voice_clone_prompt(ref_audio, ref_text, preprocess_prompt=True):
            calls.append((ref_audio, ref_text, preprocess_prompt))
            return VoiceClonePrompt(
                ref_audio_tokens=torch.zeros(8, 3, dtype=torch.long),
                ref_text=ref_text or "same",
                ref_rms=0.1,
            )

        model.create_voice_clone_prompt = create_voice_clone_prompt

        task = model._preprocess_all(
            text=["first", "second"],
            language=["zh", "zh"],
            ref_audio=["same-ref.wav", "same-ref.wav"],
            ref_text=["same text", "same text"],
        )

        self.assertEqual(calls, [("same-ref.wav", "same text", True)])
        self.assertEqual(task.batch_size, 2)
        self.assertEqual(task.ref_texts, ["same text", "same text"])

    def test_duplicate_tuple_ref_audio_is_cached_within_batch(self):
        model = _bare_model()
        calls = []
        ref_audio = (np.zeros(24000, dtype=np.float32), 24000)

        def create_voice_clone_prompt(ref_audio, ref_text, preprocess_prompt=True):
            calls.append((ref_audio, ref_text, preprocess_prompt))
            return VoiceClonePrompt(
                ref_audio_tokens=torch.zeros(8, 3, dtype=torch.long),
                ref_text=ref_text or "same",
                ref_rms=0.1,
            )

        model.create_voice_clone_prompt = create_voice_clone_prompt

        task = model._preprocess_all(
            text=["first", "second"],
            language=["zh", "zh"],
            ref_audio=[ref_audio, ref_audio],
            ref_text=["same text", "same text"],
        )

        self.assertEqual(calls, [(ref_audio, "same text", True)])
        self.assertEqual(task.batch_size, 2)
        self.assertEqual(task.ref_texts, ["same text", "same text"])

    def test_tuple_ref_audio_cache_key_changes_when_numpy_audio_changes(self):
        audio = np.zeros(8, dtype=np.float32)
        first = OmniVoice._voice_clone_prompt_cache_key(
            (audio, 24000),
            "same text",
            True,
        )
        audio[0] = 1.0
        second = OmniVoice._voice_clone_prompt_cache_key(
            (audio, 24000),
            "same text",
            True,
        )

        self.assertNotEqual(first, second)

    def test_equal_length_results_are_decoded_as_one_batch(self):
        model = _bare_model()
        model.sampling_rate = 24000
        tokenizer = _FakeAudioTokenizer()
        model.audio_tokenizer = tokenizer
        model._post_process_audio = lambda audio, postprocess_output, ref_rms: audio
        tokens = [
            torch.ones(2, 4, dtype=torch.long),
            torch.full((2, 4), 2, dtype=torch.long),
            torch.full((2, 3), 3, dtype=torch.long),
        ]

        audios = model._decode_batch_and_post_process(
            tokens,
            [1.0, 1.0, 1.0],
            OmniVoiceGenerationConfig(postprocess_output=False),
        )

        self.assertEqual(tokenizer.decode_shapes, [(2, 2, 4), (1, 2, 3)])
        self.assertEqual(len(audios), 3)
        np.testing.assert_allclose(audios[0], np.full(4, 2.0, dtype=np.float32))
        np.testing.assert_allclose(audios[1], np.full(4, 4.0, dtype=np.float32))
        np.testing.assert_allclose(audios[2], np.full(3, 6.0, dtype=np.float32))

    def test_chunked_results_are_decoded_by_equal_length_chunk_groups(self):
        model = _bare_model()
        model.sampling_rate = 24000
        tokenizer = _FakeAudioTokenizer()
        model.audio_tokenizer = tokenizer
        model._post_process_audio = lambda audio, postprocess_output, ref_rms: audio
        chunked = [
            torch.ones(2, 2, dtype=torch.long),
            torch.ones(2, 2, dtype=torch.long),
        ]
        regular = torch.full((2, 2), 2, dtype=torch.long)

        model._decode_batch_and_post_process(
            [chunked, regular],
            [1.0, 1.0],
            OmniVoiceGenerationConfig(postprocess_output=False),
        )

        self.assertEqual(tokenizer.decode_shapes, [(2, 2, 2), (1, 2, 2)])

    def test_forward_audio_logits_for_slices_gathers_and_zeros_padding(self):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(num_audio_codebook=1, audio_vocab_size=1)
        model.device = torch.device("cpu")
        hidden = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [10, 11, 12, 13, 14],
                [20, 21, 22, 23, 24],
            ],
            dtype=torch.float32,
        ).view(3, 5, 1)
        model.llm = _FakeLlm(hidden)
        model.audio_heads = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.audio_heads.weight.fill_(1.0)
        model._prepare_embed_inputs = lambda input_ids, audio_mask: torch.zeros(
            input_ids.size(0),
            input_ids.size(2),
            1,
        )
        model._forward_audio_logits_for_slices = types.MethodType(
            OmniVoice._forward_audio_logits_for_slices,
            model,
        )
        model._build_target_slice_gather_index = types.MethodType(
            OmniVoice._build_target_slice_gather_index,
            model,
        )
        model._build_iterative_update_groups = types.MethodType(
            OmniVoice._build_iterative_update_groups,
            model,
        )
        model._build_iterative_update_groups = types.MethodType(
            OmniVoice._build_iterative_update_groups,
            model,
        )

        logits = model._forward_audio_logits_for_slices(
            input_ids=torch.zeros((3, 1, 5), dtype=torch.long),
            audio_mask=torch.zeros((3, 5), dtype=torch.bool),
            attention_mask=None,
            target_slices=[(1, 4), (0, 2), (3, 5)],
            target_len_pad=4,
        )

        values = logits.squeeze(1).squeeze(-1)
        expected = torch.tensor(
            [
                [1, 2, 3, 0],
                [10, 11, 0, 0],
                [23, 24, 0, 0],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(values, expected)

    def test_refresh_target_audio_embeds_updates_only_valid_target_positions(self):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(num_audio_codebook=2, audio_mask_id=9)
        model.codebook_layer_offsets = torch.tensor([0, 100], dtype=torch.long)
        model.audio_embeddings = torch.nn.Embedding(128, 1)
        with torch.no_grad():
            model.audio_embeddings.weight.copy_(
                torch.arange(128, dtype=torch.float32).view(128, 1)
            )
        model._refresh_target_audio_embeds = types.MethodType(
            OmniVoice._refresh_target_audio_embeds,
            model,
        )
        inputs_embeds = torch.tensor([[[0.0], [10.0], [20.0], [30.0]]])
        input_ids = torch.tensor([[[1, 5, 6, 99], [2, 7, 8, 99]]])
        target_index = torch.tensor([[1, 2, 3]])
        target_valid_mask = torch.tensor([[True, True, False]])

        model._refresh_target_audio_embeds(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            target_index=target_index,
            target_valid_mask=target_valid_mask,
        )

        expected = torch.tensor([[[0.0], [112.0], [114.0], [30.0]]])
        torch.testing.assert_close(inputs_embeds, expected)

    def test_iterative_generation_reuses_static_input_embeds_between_steps(self):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(num_audio_codebook=1, audio_mask_id=99)
        model.device = torch.device("cpu")
        model.predict_calls = 0
        model.forward_calls = []
        model.prepare_embed_calls = 0
        model.refresh_calls = 0
        model._generate_iterative = types.MethodType(OmniVoice._generate_iterative, model)
        model._update_iterative_tokens_grouped = types.MethodType(
            OmniVoice._update_iterative_tokens_grouped,
            model,
        )
        model._build_target_slice_gather_index = types.MethodType(
            OmniVoice._build_target_slice_gather_index,
            model,
        )
        model._build_iterative_update_groups = types.MethodType(
            OmniVoice._build_iterative_update_groups,
            model,
        )

        def prepare(text, target_len, ref_text, ref_audio_tokens, lang, instruct, denoise):
            del text, ref_text, ref_audio_tokens, lang, instruct, denoise
            return {
                "input_ids": torch.full((1, 1, target_len + 2), 99, dtype=torch.long),
                "audio_mask": torch.ones((1, target_len + 2), dtype=torch.bool),
            }

        def prepare_embed_inputs(input_ids, audio_mask):
            del audio_mask
            model.prepare_embed_calls += 1
            return torch.zeros((input_ids.size(0), input_ids.size(2), 1))

        def refresh_target_audio_embeds(
            *,
            inputs_embeds,
            input_ids,
            target_index,
            target_valid_mask,
        ):
            del inputs_embeds, input_ids, target_index, target_valid_mask
            model.refresh_calls += 1

        def forward_audio_logits_for_slices(
            *,
            input_ids,
            audio_mask,
            attention_mask,
            target_slices,
            inputs_embeds=None,
            target_len_pad=None,
            target_index=None,
            target_valid_mask=None,
        ):
            del audio_mask, attention_mask, target_index, target_valid_mask
            model.forward_calls.append(
                {
                    "input_shape": tuple(input_ids.shape),
                    "has_inputs_embeds": inputs_embeds is not None,
                }
            )
            max_target_len = target_len_pad or max(end - start for start, end in target_slices)
            return torch.zeros(
                (len(target_slices), 1, max_target_len, 4),
                dtype=torch.float32,
            )

        def predict_tokens_with_scoring(c_logits, u_logits, gen_config):
            del u_logits, gen_config
            model.predict_calls += 1
            batch_size, codebooks, target_len, _ = c_logits.shape
            pred_tokens = torch.ones((batch_size, codebooks, target_len), dtype=torch.long)
            scores = torch.arange(target_len, dtype=torch.float32).view(1, 1, target_len)
            return pred_tokens, scores.expand(batch_size, codebooks, target_len).clone()

        model._prepare_inference_inputs = prepare
        model._prepare_embed_inputs = prepare_embed_inputs
        model._refresh_target_audio_embeds = refresh_target_audio_embeds
        model._forward_audio_logits_for_slices = forward_audio_logits_for_slices
        model._predict_tokens_with_scoring = predict_tokens_with_scoring

        task = GenerationTask(
            batch_size=2,
            texts=["a", "b"],
            target_lens=[3, 3],
            langs=[None, None],
            instructs=[None, None],
            ref_texts=[None, None],
            ref_audio_tokens=[None, None],
            ref_rms=[None, None],
        )

        model._generate_iterative(
            task,
            OmniVoiceGenerationConfig(
                num_step=2,
                position_temperature=0.0,
                layer_penalty_factor=0.0,
            ),
        )

        self.assertEqual(model.prepare_embed_calls, 1)
        self.assertEqual(model.refresh_calls, 1)
        self.assertEqual(len(model.forward_calls), 2)
        self.assertTrue(all(call["has_inputs_embeds"] for call in model.forward_calls))

    def test_split_guidance_update_matches_concatenated_logits(self):
        cond_logits = torch.zeros((2, 1, 2, 4), dtype=torch.float32)
        cond_logits[:, :, 0, 1] = 10.0
        cond_logits[:, :, 1, 2] = 5.0
        uncond_logits = torch.zeros_like(cond_logits)

        def run_update(*, split_logits: bool):
            model = types.SimpleNamespace()
            model.config = types.SimpleNamespace(num_audio_codebook=1, audio_mask_id=3)
            model.device = torch.device("cpu")
            model._update_iterative_tokens_grouped = types.MethodType(
                OmniVoice._update_iterative_tokens_grouped,
                model,
            )
            model._build_iterative_update_groups = types.MethodType(
                OmniVoice._build_iterative_update_groups,
                model,
            )
            model._predict_tokens_with_scoring = types.MethodType(
                OmniVoice._predict_tokens_with_scoring,
                model,
            )
            tokens = torch.full((2, 1, 2), 3, dtype=torch.long)
            batch_input_ids = torch.full((4, 1, 4), 3, dtype=torch.long)
            kwargs = {
                "batch_logits": None
                if split_logits
                else torch.cat([cond_logits, uncond_logits], dim=0),
                "batch_input_ids": batch_input_ids,
                "tokens": tokens,
                "c_lens": [4, 4],
                "target_lens": [2, 2],
                "schedules": [[1], [1]],
                "step": 0,
                "uncond_offset": 2,
                "layer_ids": torch.arange(1).view(1, -1, 1),
                "gen_config": OmniVoiceGenerationConfig(
                    num_step=1,
                    guidance_scale=0.0,
                    position_temperature=0.0,
                    class_temperature=0.0,
                    layer_penalty_factor=0.0,
                ),
            }
            if split_logits:
                kwargs["cond_logits"] = cond_logits
                kwargs["uncond_logits"] = uncond_logits
            model._update_iterative_tokens_grouped(**kwargs)
            return tokens, batch_input_ids

        cat_tokens, cat_input_ids = run_update(split_logits=False)
        split_tokens, split_input_ids = run_update(split_logits=True)

        torch.testing.assert_close(split_tokens, cat_tokens)
        torch.testing.assert_close(split_input_ids, cat_input_ids)

    def test_sparse_audio_embed_refresh_updates_only_selected_positions(self):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(num_audio_codebook=2, audio_mask_id=99)
        model.codebook_layer_offsets = torch.tensor([0, 10], dtype=torch.long)
        model.audio_embeddings = torch.nn.Embedding(20, 3)
        with torch.no_grad():
            model.audio_embeddings.weight.copy_(
                torch.arange(60, dtype=torch.float32).view(20, 3)
            )
        model._refresh_sparse_audio_embeds = types.MethodType(
            OmniVoice._refresh_sparse_audio_embeds,
            model,
        )

        input_ids = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[2, 3, 4], [5, 6, 7]],
            ],
            dtype=torch.long,
        )
        inputs_embeds = torch.zeros((2, 3, 3), dtype=torch.float32)

        model._refresh_sparse_audio_embeds(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            batch_rows=torch.tensor([0, 1], dtype=torch.long),
            seq_positions=torch.tensor([1, 2], dtype=torch.long),
        )

        expected_0_1 = (
            model.audio_embeddings(torch.tensor([2, 15], dtype=torch.long)).sum(dim=0)
        )
        expected_1_2 = (
            model.audio_embeddings(torch.tensor([4, 17], dtype=torch.long)).sum(dim=0)
        )
        torch.testing.assert_close(inputs_embeds[0, 1], expected_0_1)
        torch.testing.assert_close(inputs_embeds[1, 2], expected_1_2)
        torch.testing.assert_close(inputs_embeds[0, 0], torch.zeros(3))
        torch.testing.assert_close(inputs_embeds[1, 1], torch.zeros(3))

    def test_iterative_generation_uses_incremental_static_embed_refresh(self):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(num_audio_codebook=1, audio_mask_id=99)
        model.device = torch.device("cpu")
        model.predict_calls = 0
        model.forward_calls = []
        model.sparse_refresh_calls = []
        model._generate_iterative = types.MethodType(OmniVoice._generate_iterative, model)
        model._update_iterative_tokens_grouped = types.MethodType(
            OmniVoice._update_iterative_tokens_grouped,
            model,
        )
        model._build_target_slice_gather_index = types.MethodType(
            OmniVoice._build_target_slice_gather_index,
            model,
        )
        model._build_iterative_update_groups = types.MethodType(
            OmniVoice._build_iterative_update_groups,
            model,
        )

        def prepare(text, target_len, ref_text, ref_audio_tokens, lang, instruct, denoise):
            del text, ref_text, ref_audio_tokens, lang, instruct, denoise
            return {
                "input_ids": torch.full((1, 1, target_len + 2), 99, dtype=torch.long),
                "audio_mask": torch.ones((1, target_len + 2), dtype=torch.bool),
            }

        def prepare_embed_inputs(input_ids, audio_mask):
            del audio_mask
            return torch.zeros((input_ids.size(0), input_ids.size(-1), 4))

        def full_refresh(**kwargs):
            raise AssertionError("full target refresh should not run")

        def sparse_refresh(**kwargs):
            model.sparse_refresh_calls.append(
                {
                    "batch_rows": tuple(kwargs["batch_rows"].reshape(-1).tolist()),
                    "seq_positions": tuple(
                        kwargs["seq_positions"].reshape(-1).tolist()
                    ),
                }
            )

        def forward_audio_logits_for_slices(
            *,
            input_ids,
            audio_mask,
            attention_mask,
            target_slices,
            inputs_embeds=None,
            target_len_pad=None,
            target_index=None,
            target_valid_mask=None,
            profile=None,
        ):
            del audio_mask, attention_mask, target_index, target_valid_mask, profile
            model.forward_calls.append(
                {
                    "input_shape": tuple(input_ids.shape),
                    "has_inputs_embeds": inputs_embeds is not None,
                    "target_slices": target_slices,
                }
            )
            max_target_len = target_len_pad or max(
                end - start for start, end in target_slices
            )
            return torch.zeros(
                (len(target_slices), 1, max_target_len, 4),
                dtype=torch.float32,
            )

        def predict_tokens_with_scoring(c_logits, u_logits, gen_config):
            del u_logits, gen_config
            model.predict_calls += 1
            batch_size, codebooks, target_len, _ = c_logits.shape
            pred_tokens = torch.ones((batch_size, codebooks, target_len), dtype=torch.long)
            scores = torch.arange(target_len, dtype=torch.float32).view(1, 1, target_len)
            return pred_tokens, scores.expand(batch_size, codebooks, target_len).clone()

        model._prepare_inference_inputs = prepare
        model._prepare_embed_inputs = prepare_embed_inputs
        model._refresh_target_audio_embeds = full_refresh
        model._refresh_sparse_audio_embeds = sparse_refresh
        model._forward_audio_logits_for_slices = forward_audio_logits_for_slices
        model._predict_tokens_with_scoring = predict_tokens_with_scoring

        task = GenerationTask(
            batch_size=1,
            texts=["a"],
            target_lens=[2],
            langs=[None],
            instructs=[None],
            ref_texts=[None],
            ref_audio_tokens=[None],
            ref_rms=[None],
        )

        model._generate_iterative(
            task,
            OmniVoiceGenerationConfig(
                num_step=2,
                position_temperature=0.0,
                layer_penalty_factor=0.0,
                reuse_static_input_embeds=True,
            ),
        )

        self.assertEqual(len(model.forward_calls), 2)
        self.assertTrue(all(call["has_inputs_embeds"] for call in model.forward_calls))
        self.assertEqual(len(model.sparse_refresh_calls), 2)
        self.assertEqual(model.sparse_refresh_calls[0]["batch_rows"], (1,))
        self.assertEqual(model.sparse_refresh_calls[1]["batch_rows"], (0,))

    def test_iterative_generation_uses_static_padding_and_grouped_update(self):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(num_audio_codebook=1, audio_mask_id=99)
        model.device = torch.device("cpu")
        model.predict_calls = 0
        model.forward_calls = []
        model._generate_iterative = types.MethodType(OmniVoice._generate_iterative, model)
        model._update_iterative_tokens_grouped = types.MethodType(
            OmniVoice._update_iterative_tokens_grouped,
            model,
        )
        model._build_target_slice_gather_index = types.MethodType(
            OmniVoice._build_target_slice_gather_index,
            model,
        )
        model._build_iterative_update_groups = types.MethodType(
            OmniVoice._build_iterative_update_groups,
            model,
        )

        def prepare(text, target_len, ref_text, ref_audio_tokens, lang, instruct, denoise):
            del text, ref_text, ref_audio_tokens, lang, instruct, denoise
            return {
                "input_ids": torch.full((1, 1, target_len + 2), 99, dtype=torch.long),
                "audio_mask": torch.ones((1, target_len + 2), dtype=torch.bool),
            }

        def forward_audio_logits_for_slices(
            *,
            input_ids,
            audio_mask,
            attention_mask,
            target_slices,
            target_len_pad=None,
            target_index=None,
            target_valid_mask=None,
        ):
            del audio_mask, attention_mask
            model.forward_calls.append(
                {
                    "input_shape": tuple(input_ids.shape),
                    "target_slices": target_slices,
                    "target_len_pad": target_len_pad,
                    "target_index_shape": tuple(target_index.shape),
                    "target_valid_mask_shape": tuple(target_valid_mask.shape),
                    "target_valid_mask_count": int(target_valid_mask.sum().item()),
                }
            )
            max_target_len = target_len_pad or max(end - start for start, end in target_slices)
            return torch.zeros(
                (len(target_slices), 1, max_target_len, 4),
                dtype=torch.float32,
            )

        def predict_tokens_with_scoring(c_logits, u_logits, gen_config):
            del u_logits, gen_config
            model.predict_calls += 1
            batch_size, codebooks, target_len, _ = c_logits.shape
            pred_tokens = torch.ones((batch_size, codebooks, target_len), dtype=torch.long)
            scores = torch.arange(target_len, dtype=torch.float32).view(1, 1, target_len)
            return pred_tokens, scores.expand(batch_size, codebooks, target_len).clone()

        model._prepare_inference_inputs = prepare
        model._forward_audio_logits_for_slices = forward_audio_logits_for_slices
        model._predict_tokens_with_scoring = predict_tokens_with_scoring

        task = GenerationTask(
            batch_size=2,
            texts=["a", "b"],
            target_lens=[3, 3],
            langs=[None, None],
            instructs=[None, None],
            ref_texts=[None, None],
            ref_audio_tokens=[None, None],
            ref_rms=[None, None],
        )

        tokens = model._generate_iterative(
            task,
            OmniVoiceGenerationConfig(
                num_step=1,
                position_temperature=0.0,
                layer_penalty_factor=0.0,
                batch_size_pad=4,
                seq_len_bucket_multiple=8,
                target_len_bucket_multiple=4,
            ),
        )

        self.assertEqual(model.forward_calls[0]["input_shape"], (8, 1, 8))
        self.assertEqual(model.forward_calls[0]["target_len_pad"], 4)
        self.assertEqual(len(model.forward_calls[0]["target_slices"]), 8)
        self.assertEqual(model.forward_calls[0]["target_index_shape"], (8, 4))
        self.assertEqual(model.forward_calls[0]["target_valid_mask_shape"], (8, 4))
        self.assertEqual(model.forward_calls[0]["target_valid_mask_count"], 28)
        self.assertEqual(model.predict_calls, 1)
        self.assertEqual([tuple(token.shape) for token in tokens], [(1, 3), (1, 3)])

    def test_iterative_generation_auto_splits_guidance_for_large_context_savings(self):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(num_audio_codebook=1, audio_mask_id=99)
        model.device = torch.device("cpu")
        model.predict_calls = 0
        model.forward_calls = []
        model._generate_iterative = types.MethodType(OmniVoice._generate_iterative, model)
        model._update_iterative_tokens_grouped = types.MethodType(
            OmniVoice._update_iterative_tokens_grouped,
            model,
        )
        model._build_target_slice_gather_index = types.MethodType(
            OmniVoice._build_target_slice_gather_index,
            model,
        )
        model._build_iterative_update_groups = types.MethodType(
            OmniVoice._build_iterative_update_groups,
            model,
        )

        def prepare(text, target_len, ref_text, ref_audio_tokens, lang, instruct, denoise):
            del text, ref_text, ref_audio_tokens, lang, instruct, denoise
            return {
                "input_ids": torch.full((1, 1, target_len + 4), 99, dtype=torch.long),
                "audio_mask": torch.ones((1, target_len + 4), dtype=torch.bool),
            }

        def forward_audio_logits_for_slices(
            *,
            input_ids,
            audio_mask,
            attention_mask,
            target_slices,
            target_len_pad=None,
            target_index=None,
            target_valid_mask=None,
            bidirectional_no_mask=False,
        ):
            del audio_mask, attention_mask, target_index, target_valid_mask
            model.forward_calls.append(
                {
                    "input_shape": tuple(input_ids.shape),
                    "target_slices": target_slices,
                    "target_len_pad": target_len_pad,
                    "bidirectional_no_mask": bidirectional_no_mask,
                }
            )
            max_target_len = target_len_pad or max(end - start for start, end in target_slices)
            return torch.zeros(
                (len(target_slices), 1, max_target_len, 4),
                dtype=torch.float32,
            )

        def predict_tokens_with_scoring(c_logits, u_logits, gen_config):
            del u_logits, gen_config
            model.predict_calls += 1
            batch_size, codebooks, target_len, _ = c_logits.shape
            pred_tokens = torch.ones((batch_size, codebooks, target_len), dtype=torch.long)
            scores = torch.arange(target_len, dtype=torch.float32).view(1, 1, target_len)
            return pred_tokens, scores.expand(batch_size, codebooks, target_len).clone()

        model._prepare_inference_inputs = prepare
        model._forward_audio_logits_for_slices = forward_audio_logits_for_slices
        model._predict_tokens_with_scoring = predict_tokens_with_scoring

        batch_size = 24
        task = GenerationTask(
            batch_size=batch_size,
            texts=[str(i) for i in range(batch_size)],
            target_lens=[3 for _ in range(batch_size)],
            langs=[None for _ in range(batch_size)],
            instructs=[None for _ in range(batch_size)],
            ref_texts=[None for _ in range(batch_size)],
            ref_audio_tokens=[None for _ in range(batch_size)],
            ref_rms=[None for _ in range(batch_size)],
        )

        tokens = model._generate_iterative(
            task,
            OmniVoiceGenerationConfig(
                num_step=1,
                position_temperature=0.0,
                layer_penalty_factor=0.0,
                split_guidance_forward="auto",
                split_guidance_min_batch_size=24,
                split_guidance_min_saved_context_ratio=0.25,
            ),
        )

        self.assertEqual(
            [call["input_shape"] for call in model.forward_calls],
            [(24, 1, 7), (24, 1, 3)],
        )
        self.assertEqual(
            [call["bidirectional_no_mask"] for call in model.forward_calls],
            [True, True],
        )
        self.assertEqual(model.predict_calls, 1)
        self.assertEqual([tuple(token.shape) for token in tokens], [(1, 3)] * 24)

    def test_iterative_split_guidance_keeps_mask_when_padding_is_present(self):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(num_audio_codebook=1, audio_mask_id=99)
        model.device = torch.device("cpu")
        model.predict_calls = 0
        model.forward_calls = []
        model._generate_iterative = types.MethodType(OmniVoice._generate_iterative, model)
        model._update_iterative_tokens_grouped = types.MethodType(
            OmniVoice._update_iterative_tokens_grouped,
            model,
        )
        model._build_target_slice_gather_index = types.MethodType(
            OmniVoice._build_target_slice_gather_index,
            model,
        )
        model._build_iterative_update_groups = types.MethodType(
            OmniVoice._build_iterative_update_groups,
            model,
        )

        def prepare(text, target_len, ref_text, ref_audio_tokens, lang, instruct, denoise):
            del text, ref_text, ref_audio_tokens, lang, instruct, denoise
            return {
                "input_ids": torch.full((1, 1, target_len + 4), 99, dtype=torch.long),
                "audio_mask": torch.ones((1, target_len + 4), dtype=torch.bool),
            }

        def forward_audio_logits_for_slices(
            *,
            input_ids,
            audio_mask,
            attention_mask,
            target_slices,
            target_len_pad=None,
            target_index=None,
            target_valid_mask=None,
            bidirectional_no_mask=False,
        ):
            del audio_mask, attention_mask, target_index, target_valid_mask
            model.forward_calls.append(
                {
                    "input_shape": tuple(input_ids.shape),
                    "target_slices": target_slices,
                    "target_len_pad": target_len_pad,
                    "bidirectional_no_mask": bidirectional_no_mask,
                }
            )
            max_target_len = target_len_pad or max(end - start for start, end in target_slices)
            return torch.zeros(
                (len(target_slices), 1, max_target_len, 4),
                dtype=torch.float32,
            )

        def predict_tokens_with_scoring(c_logits, u_logits, gen_config):
            del u_logits, gen_config
            model.predict_calls += 1
            batch_size, codebooks, target_len, _ = c_logits.shape
            pred_tokens = torch.ones((batch_size, codebooks, target_len), dtype=torch.long)
            scores = torch.arange(target_len, dtype=torch.float32).view(1, 1, target_len)
            return pred_tokens, scores.expand(batch_size, codebooks, target_len).clone()

        model._prepare_inference_inputs = prepare
        model._forward_audio_logits_for_slices = forward_audio_logits_for_slices
        model._predict_tokens_with_scoring = predict_tokens_with_scoring

        task = GenerationTask(
            batch_size=24,
            texts=[str(i) for i in range(24)],
            target_lens=[3 for _ in range(24)],
            langs=[None for _ in range(24)],
            instructs=[None for _ in range(24)],
            ref_texts=[None for _ in range(24)],
            ref_audio_tokens=[None for _ in range(24)],
            ref_rms=[None for _ in range(24)],
        )

        model._generate_iterative(
            task,
            OmniVoiceGenerationConfig(
                num_step=1,
                position_temperature=0.0,
                layer_penalty_factor=0.0,
                split_guidance_forward="auto",
                split_guidance_min_batch_size=24,
                split_guidance_min_saved_context_ratio=0.25,
                batch_size_pad=32,
            ),
        )

        self.assertEqual(
            [call["input_shape"] for call in model.forward_calls],
            [(32, 1, 7), (32, 1, 3)],
        )
        self.assertEqual(
            [call["bidirectional_no_mask"] for call in model.forward_calls],
            [False, False],
        )

    def test_iterative_default_auto_splits_guidance_from_batch_eight(self):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(num_audio_codebook=1, audio_mask_id=99)
        model.device = torch.device("cpu")
        model.predict_calls = 0
        model.forward_calls = []
        model._generate_iterative = types.MethodType(OmniVoice._generate_iterative, model)
        model._update_iterative_tokens_grouped = types.MethodType(
            OmniVoice._update_iterative_tokens_grouped,
            model,
        )
        model._build_target_slice_gather_index = types.MethodType(
            OmniVoice._build_target_slice_gather_index,
            model,
        )
        model._build_iterative_update_groups = types.MethodType(
            OmniVoice._build_iterative_update_groups,
            model,
        )

        def prepare(text, target_len, ref_text, ref_audio_tokens, lang, instruct, denoise):
            del text, ref_text, ref_audio_tokens, lang, instruct, denoise
            return {
                "input_ids": torch.full((1, 1, target_len + 4), 99, dtype=torch.long),
                "audio_mask": torch.ones((1, target_len + 4), dtype=torch.bool),
            }

        def forward_audio_logits_for_slices(
            *,
            input_ids,
            audio_mask,
            attention_mask,
            target_slices,
            target_len_pad=None,
            target_index=None,
            target_valid_mask=None,
            bidirectional_no_mask=False,
        ):
            del audio_mask, attention_mask, target_index, target_valid_mask
            model.forward_calls.append(
                {
                    "input_shape": tuple(input_ids.shape),
                    "target_slices": target_slices,
                    "target_len_pad": target_len_pad,
                    "bidirectional_no_mask": bidirectional_no_mask,
                }
            )
            max_target_len = target_len_pad or max(end - start for start, end in target_slices)
            return torch.zeros(
                (len(target_slices), 1, max_target_len, 4),
                dtype=torch.float32,
            )

        def predict_tokens_with_scoring(c_logits, u_logits, gen_config):
            del u_logits, gen_config
            model.predict_calls += 1
            batch_size, codebooks, target_len, _ = c_logits.shape
            pred_tokens = torch.ones((batch_size, codebooks, target_len), dtype=torch.long)
            scores = torch.arange(target_len, dtype=torch.float32).view(1, 1, target_len)
            return pred_tokens, scores.expand(batch_size, codebooks, target_len).clone()

        model._prepare_inference_inputs = prepare
        model._forward_audio_logits_for_slices = forward_audio_logits_for_slices
        model._predict_tokens_with_scoring = predict_tokens_with_scoring

        task = GenerationTask(
            batch_size=8,
            texts=[str(i) for i in range(8)],
            target_lens=[3 for _ in range(8)],
            langs=[None for _ in range(8)],
            instructs=[None for _ in range(8)],
            ref_texts=[None for _ in range(8)],
            ref_audio_tokens=[None for _ in range(8)],
            ref_rms=[None for _ in range(8)],
        )

        model._generate_iterative(
            task,
            OmniVoiceGenerationConfig(
                num_step=1,
                position_temperature=0.0,
                layer_penalty_factor=0.0,
            ),
        )

        self.assertEqual(
            [call["input_shape"] for call in model.forward_calls],
            [(8, 1, 7), (8, 1, 3)],
        )
        self.assertEqual(
            [call["bidirectional_no_mask"] for call in model.forward_calls],
            [True, True],
        )


class _FakeAudioOutput:
    def __init__(self, audio_values):
        self.audio_values = audio_values


class _FakeAudioTokenizer:
    device = torch.device("cpu")

    def __init__(self):
        self.decode_shapes = []

    def decode(self, tokens):
        self.decode_shapes.append(tuple(tokens.shape))
        audio = tokens.to(torch.float32).sum(dim=1, keepdim=True)
        return _FakeAudioOutput(audio)


class _FakeLlm(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden

    def forward(self, inputs_embeds, attention_mask=None, return_dict=True, position_ids=None):
        del inputs_embeds, attention_mask, return_dict, position_ids
        return (self.hidden,)


if __name__ == "__main__":
    unittest.main()
