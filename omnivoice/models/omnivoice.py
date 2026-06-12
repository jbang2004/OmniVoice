#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core OmniVoice model implementation.

Defines the ``OmniVoice`` model class, generation config, and inference pipeline.
This is the main entry point for both inference and training:

- **Inference**: ``OmniVoice.from_pretrained()`` loads the model, then
  ``model.generate()`` supports voice cloning, voice design, and auto voice.
- **Training**: ``model.forward()`` computes the training loss; the model is
  built and used by ``omnivoice.training.builder`` and ``omnivoice.training.trainer``.

"""

import difflib
import hashlib
import logging
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Any, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

try:
    from torch.nn.attention.flex_attention import create_block_mask

    _flex_attention_available = True
except ImportError:
    _flex_attention_available = False
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HiggsAudioV2TokenizerModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING, AutoConfig

from omnivoice.utils.audio import (
    cross_fade_chunks,
    fade_and_pad_audio,
    load_audio,
    remove_silence,
    trim_long_audio,
)
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAMES
from omnivoice.utils.text import add_punctuation, chunk_text_punctuation
from omnivoice.utils.voice_design import (
    _INSTRUCT_ALL_VALID,
    _INSTRUCT_EN_TO_ZH,
    _INSTRUCT_MUTUALLY_EXCLUSIVE,
    _INSTRUCT_VALID_EN,
    _INSTRUCT_VALID_ZH,
    _INSTRUCT_ZH_TO_EN,
    _ZH_RE,
)

logger = logging.getLogger(__name__)


def _module_device(module: nn.Module) -> torch.device:
    try:
        return torch.device(module.device)
    except AttributeError:
        try:
            return next(module.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VoiceClonePrompt:
    ref_audio_tokens: torch.Tensor  # (C, T)
    ref_text: str
    ref_rms: float


@dataclass
class _GenerationProfiler:
    enabled: bool
    device: Union[torch.device, str]
    timings_s: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def _sync(self) -> None:
        if not self.enabled:
            return
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            synchronize = getattr(torch.mps, "synchronize", None)
            if synchronize is not None:
                synchronize()

    @contextmanager
    def section(self, name: str, *, sync: bool = False) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if sync:
            self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                self._sync()
            self.timings_s[name] = self.timings_s.get(name, 0.0) + (
                time.perf_counter() - started
            )
            self.counts[name] = self.counts.get(name, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        total_s = self.timings_s.get("total_s", 0.0)
        percentages = {
            name: (value / total_s * 100.0)
            for name, value in self.timings_s.items()
            if total_s > 0 and name != "total_s"
        }
        return {
            "timings_s": dict(sorted(self.timings_s.items())),
            "percentages": dict(sorted(percentages.items())),
            "counts": dict(sorted(self.counts.items())),
            "metadata": self.metadata,
            "sync_profile": True,
        }


@dataclass
class OmniVoiceGenerationConfig:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    batched_decode: bool = False
    batch_size_pad: Optional[int] = None
    seq_len_bucket_multiple: int = 1
    target_len_bucket_multiple: int = 1
    collect_profile: bool = False
    split_guidance_forward: Union[bool, str] = "auto"
    split_guidance_min_batch_size: int = 8
    split_guidance_min_saved_context_ratio: float = 0.25
    reuse_static_input_embeds: bool = True

    @classmethod
    def from_dict(cls, kwargs_dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys}
        return cls(**filtered)


def _ref_audio_tuple_cache_marker(ref_audio: Any) -> Optional[tuple[Any, ...]]:
    """Return a stable-ish cache marker for in-memory ``(waveform, sr)`` audio."""
    if not isinstance(ref_audio, tuple) or len(ref_audio) < 2:
        return None
    waveform, sample_rate = ref_audio[0], ref_audio[1]
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        return None

    marker = _waveform_cache_marker(waveform)
    if marker is None:
        return None
    return ("tuple", sample_rate, marker)


def _waveform_cache_marker(waveform: Any) -> Optional[tuple[Any, ...]]:
    shape = getattr(waveform, "shape", None)
    dtype = getattr(waveform, "dtype", None)
    if shape is None or dtype is None:
        return None
    shape_tuple = tuple(int(dim) for dim in shape)
    dtype_name = str(dtype)

    if isinstance(waveform, np.ndarray):
        array = np.ascontiguousarray(waveform)
        digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
        return ("numpy", shape_tuple, dtype_name, digest)

    if isinstance(waveform, torch.Tensor):
        tensor = waveform.detach()
        device = str(tensor.device)
        if tensor.device.type == "cpu":
            array = tensor.contiguous().numpy()
            digest = hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()
            return ("torch-cpu", shape_tuple, dtype_name, digest)
        data_ptr = int(tensor.data_ptr()) if tensor.numel() > 0 else 0
        version = int(getattr(tensor, "_version", 0))
        return (
            "torch-device",
            id(waveform),
            device,
            shape_tuple,
            dtype_name,
            data_ptr,
            version,
        )

    return None


@dataclass
class GenerationTask:
    batch_size: int
    texts: List[str]
    target_lens: List[int]
    langs: List[Optional[str]]
    instructs: List[Optional[str]]
    ref_texts: List[Optional[str]]
    ref_audio_tokens: List[Optional[torch.Tensor]]
    ref_rms: List[Optional[float]]
    speed: Optional[List[float]] = None

    def get_indices(self, config: OmniVoiceGenerationConfig, frame_rate: int):
        threshold = int(config.audio_chunk_threshold * frame_rate)
        short_idx = [i for i, l in enumerate(self.target_lens) if l <= threshold]
        long_idx = [i for i, l in enumerate(self.target_lens) if l > threshold]
        return short_idx, long_idx

    def slice_task(self, indices: List[int]):
        if not indices:
            return None
        return GenerationTask(
            batch_size=len(indices),
            texts=[self.texts[i] for i in indices],
            target_lens=[self.target_lens[i] for i in indices],
            langs=[self.langs[i] for i in indices],
            instructs=[self.instructs[i] for i in indices],
            ref_texts=[self.ref_texts[i] for i in indices],
            ref_audio_tokens=[self.ref_audio_tokens[i] for i in indices],
            ref_rms=[self.ref_rms[i] for i in indices],
            speed=[self.speed[i] for i in indices] if self.speed else None,
        )


@dataclass
class OmniVoiceModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Config & Model
# ---------------------------------------------------------------------------


class OmniVoiceConfig(PretrainedConfig):
    model_type = "omnivoice"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        audio_vocab_size: int = 1025,
        audio_mask_id: int = 1024,
        num_audio_codebook: int = 8,
        audio_codebook_weights: Optional[list[float]] = None,
        llm_config: Optional[Union[dict, PretrainedConfig]] = None,
        **kwargs,
    ):

        if isinstance(llm_config, dict):
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)

        self.llm_config = llm_config

        super().__init__(**kwargs)
        self.audio_vocab_size = audio_vocab_size
        self.audio_mask_id = audio_mask_id
        self.num_audio_codebook = num_audio_codebook
        if audio_codebook_weights is None:
            audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]
        self.audio_codebook_weights = audio_codebook_weights


def _resolve_model_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)


class OmniVoice(PreTrainedModel):
    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class = OmniVoiceConfig

    def __init__(self, config: OmniVoiceConfig, llm: Optional[PreTrainedModel] = None):
        super().__init__(config)

        if llm is not None:
            # If an LLM instance is provided, use it directly
            # (skipping config-based init).
            self.llm = llm
        else:
            # Otherwise, initialize the LLM from the config.
            self.llm = AutoModel.from_config(self.config.llm_config)

        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            self.config.llm_config.hidden_size,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
        )

        self.audio_heads = nn.Linear(
            self.config.llm_config.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )

        self.normalized_audio_codebook_weights = [
            w / sum(config.audio_codebook_weights)
            for w in config.audio_codebook_weights
        ]

        self.post_init()

        # Inference-only attributes (set by from_pretrained when not in train mode)
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.duration_estimator = None
        self.sampling_rate = None
        self._asr_pipe = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        train_mode = kwargs.pop("train", False)
        load_asr = kwargs.pop("load_asr", False)
        asr_model_name = kwargs.pop("asr_model_name", "openai/whisper-large-v3-turbo")

        # Suppress noisy INFO logs from transformers/huggingface_hub during loading
        _prev_disable = logging.root.manager.disable
        logging.disable(logging.INFO)

        try:
            # Resolve to local path first; download only if not already cached
            resolved_path = _resolve_model_path(pretrained_model_name_or_path)

            model = super().from_pretrained(resolved_path, *args, **kwargs)

            if not train_mode:
                model.text_tokenizer = AutoTokenizer.from_pretrained(resolved_path)

                audio_tokenizer_path = os.path.join(resolved_path, "audio_tokenizer")

                if not os.path.isdir(audio_tokenizer_path):
                    audio_tokenizer_path = _resolve_model_path(
                        "eustlb/higgs-audio-v2-tokenizer"
                    )

                # higgs-audio-v2-tokenizer does not support MPS
                # (output channels > 65536)
                tokenizer_device = (
                    "cpu" if str(model.device).startswith("mps") else model.device
                )
                model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                    audio_tokenizer_path, device_map=tokenizer_device
                )
                model.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    audio_tokenizer_path
                )

                model.sampling_rate = model.feature_extractor.sampling_rate

                model.duration_estimator = RuleDurationEstimator()

                if load_asr:
                    model.load_asr_model(model_name=asr_model_name)
        finally:
            logging.disable(_prev_disable)

        return model

    # -------------------------------------------------------------------
    # ASR support (optional, for auto-transcription)
    # -------------------------------------------------------------------

    def load_asr_model(self, model_name: str = "openai/whisper-large-v3-turbo"):
        """Load a Whisper ASR model for reference audio transcription.

        Args:
            model_name: HuggingFace model name or local path for the Whisper model.
        """
        from transformers import pipeline as hf_pipeline

        logger.info("Loading ASR model %s ...", model_name)
        asr_dtype = (
            torch.float16 if str(self.device).startswith(("cuda", "xpu")) else torch.float32
        )

        model_name = _resolve_model_path(model_name)

        self._asr_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            dtype=asr_dtype,
            device_map=self.device,
        )
        logger.info("ASR model loaded on %s.", self.device)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, tuple],
    ) -> str:
        """Transcribe audio using the loaded Whisper ASR model.

        Args:
            audio: File path or ``(waveform, sample_rate)`` tuple.
                Waveform can be a numpy array or torch.Tensor of shape
                ``(1, T)`` or ``(T,)``.

        Returns:
            Transcribed text.
        """
        if self._asr_pipe is None:
            raise RuntimeError(
                "ASR model is not loaded. Call model.load_asr_model() first."
            )

        if isinstance(audio, str):
            return self._asr_pipe(audio)["text"].strip()
        else:
            waveform, sr = audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            waveform = np.squeeze(waveform)  # (1, T) or (T,) → (T,)
            audio_input = {
                "array": waveform,
                "sampling_rate": sr,
            }
            return self._asr_pipe(audio_input)["text"].strip()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def _prepare_embed_inputs(
        self, input_ids: torch.Tensor, audio_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepares embeddings from input_ids of shape (batch_size, layers, seq_length).
        Embedding shape is (batch_size, seq_length, hidden_size).
        """
        text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])

        # Apply shift to audio IDs based on codebook layer
        # audio_ids: [Batch, 8, Seq]
        # codebook_layer_offsets: [1, 8, 1]
        # Result: Layer 0 ID Layer 1 ID + Layer 2 ID + 2050...
        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1)
        ) + self.codebook_layer_offsets.view(1, -1, 1)

        # input: [Batch, 8, Seq] -> output: [Batch, Seq, Hidden]
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):

        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)

        if attention_mask is None and document_ids is not None:
            if not _flex_attention_available:
                raise RuntimeError(
                    "flex_attention is not available in the current environment. "
                    "If you do not need flex_attention, set "
                    '"attn_implementation": "sdpa" in your training config.'
                )
            attention_mask = create_block_mask(
                _get_packed_mask(
                    document_ids[0].to(inputs_embeds.device),
                ),
                B=None,
                H=None,
                Q_LEN=input_ids.size(-1),
                KV_LEN=input_ids.size(-1),
                _compile=True,
                device=inputs_embeds.device,
            )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            position_ids=position_ids,
        )
        hidden_states = llm_outputs[0]

        loss = None

        # Shape: [B, S, C * Vocab]
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)
        # Shape: [B, S, C, Vocab] -> [B, C, S, Vocab]
        audio_logits = logits_flat.view(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)

        if labels is not None:

            # audio_logits.permute(0, 3, 1, 2):
            # [Batch, Layer, Seq, Vocab] -> [Batch, Vocab, Layer, Seq]
            # per_token_loss shape: [Batch, Layer, Seq]，ignore -100
            per_token_loss = torch.nn.functional.cross_entropy(
                audio_logits.permute(0, 3, 1, 2),
                labels,
                reduction="none",
                ignore_index=-100,
            )
            # valid_mask shape: [Batch, Layer, Seq]
            valid_mask = (labels != -100).float()

            # layer_means shape: [num_layers]
            layer_means = (per_token_loss * valid_mask).sum(
                dim=(0, 2)
            ) / valid_mask.sum(dim=(0, 2)).clamp(min=1.0)

            weights = torch.tensor(
                self.normalized_audio_codebook_weights, device=audio_logits.device
            )
            loss = (layer_means * weights).sum()

        return OmniVoiceModelOutput(
            loss=loss,
            logits=audio_logits,
        )

    def _forward_audio_logits_for_slices(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        target_slices: List[tuple[int, int]],
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        target_len_pad: Optional[int] = None,
        target_index: Optional[torch.LongTensor] = None,
        target_valid_mask: Optional[torch.Tensor] = None,
        bidirectional_no_mask: bool = False,
        profile: Optional[_GenerationProfiler] = None,
    ) -> torch.Tensor:
        """Run the LLM and project logits only for requested audio positions.

        Generation only consumes logits for the masked target-audio span, but
        ``forward()`` projects every hidden state through ``audio_heads``.  The
        projection is wide (num_codebooks * audio_vocab_size), so avoiding text
        and reference-audio positions removes avoidable work from every
        diffusion step while keeping the LLM context unchanged.
        """
        if profile is None:
            profile = _GenerationProfiler(enabled=False, device=_module_device(self))
        if len(target_slices) != input_ids.size(0):
            raise ValueError("target_slices must match batch size")

        with profile.section("forward_prepare_embeds_s", sync=True):
            if inputs_embeds is None:
                inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)
            elif (
                inputs_embeds.size(0) != input_ids.size(0)
                or inputs_embeds.size(1) != input_ids.size(-1)
            ):
                raise ValueError("inputs_embeds must match input_ids batch and length")
        with profile.section("forward_llm_s", sync=True):
            llm_attention_mask: Optional[Union[torch.Tensor, dict[str, None]]]
            llm_attention_mask = attention_mask
            llm_kwargs: dict[str, Any] = {}
            if bidirectional_no_mask:
                # OmniVoice iterative decoding intentionally uses bidirectional
                # attention over the diffusion target span. When a branch has
                # no padding, a dense all-true 4D mask is equivalent to no mask
                # plus is_causal=False, and avoids forcing SDPA down a custom
                # mask path.
                llm_attention_mask = {
                    "full_attention": None,
                    "sliding_attention": None,
                }
                llm_kwargs["is_causal"] = False
            llm_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=llm_attention_mask,
                return_dict=True,
                position_ids=position_ids,
                **llm_kwargs,
            )
            hidden_states = llm_outputs[0]

        with profile.section("forward_select_targets_s", sync=True):
            if target_index is None or target_valid_mask is None:
                target_index, target_valid_mask = self._build_target_slice_gather_index(
                    target_slices=target_slices,
                    source_len=hidden_states.size(1),
                    target_len_pad=target_len_pad,
                    device=hidden_states.device,
                )
            elif target_index.size(0) != hidden_states.size(0):
                raise ValueError("target_index must match batch size")

            selected = hidden_states.gather(
                1,
                target_index.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
            )
            selected = selected.masked_fill(~target_valid_mask.unsqueeze(-1), 0)

        with profile.section("forward_audio_heads_s", sync=True):
            logits_flat = self.audio_heads(selected)
            batch_size, seq_len, _ = logits_flat.shape
            return logits_flat.view(
                batch_size,
                seq_len,
                self.config.num_audio_codebook,
                self.config.audio_vocab_size,
            ).permute(0, 2, 1, 3)

    def _build_target_slice_gather_index(
        self,
        *,
        target_slices: List[tuple[int, int]],
        source_len: int,
        target_len_pad: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        lengths = [end - start for start, end in target_slices]
        max_len = max(lengths)
        if target_len_pad is not None:
            if target_len_pad < max_len:
                raise ValueError("target_len_pad must be >= max target slice length")
            max_len = target_len_pad

        starts = []
        for start, end in target_slices:
            if start < 0 or end < start or end > source_len:
                raise ValueError("target_slices must be within hidden-state length")
            starts.append(start)

        device = device or self.device
        starts_t = torch.tensor(starts, dtype=torch.long, device=device)
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        offsets = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0)
        target_index = starts_t.unsqueeze(1) + offsets
        target_valid_mask = offsets < lengths_t.unsqueeze(1)
        target_index = target_index.clamp(max=max(source_len - 1, 0))
        return target_index, target_valid_mask

    def _refresh_target_audio_embeds(
        self,
        *,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        target_index: torch.LongTensor,
        target_valid_mask: torch.Tensor,
    ) -> None:
        """Refresh cached input embeddings for target-audio positions only."""
        if target_index.size(0) != input_ids.size(0):
            raise ValueError("target_index must match input_ids batch size")
        if inputs_embeds.size(0) != input_ids.size(0):
            raise ValueError("inputs_embeds must match input_ids batch size")

        codebooks = input_ids.size(1)
        target_ids = input_ids.gather(
            2,
            target_index.unsqueeze(1).expand(-1, codebooks, -1),
        )
        safe_target_ids = target_ids.masked_fill(
            ~target_valid_mask.unsqueeze(1),
            self.config.audio_mask_id,
        )
        shifted_ids = safe_target_ids + self.codebook_layer_offsets.view(1, -1, 1)
        refreshed = self.audio_embeddings(shifted_ids).sum(dim=1)

        embed_index = target_index.unsqueeze(-1).expand(
            -1,
            -1,
            inputs_embeds.size(-1),
        )
        if not bool(target_valid_mask.all()):
            current = inputs_embeds.gather(1, embed_index)
            refreshed = torch.where(
                target_valid_mask.unsqueeze(-1),
                refreshed,
                current,
            )
        inputs_embeds.scatter_(1, embed_index, refreshed)

    def _refresh_sparse_audio_embeds(
        self,
        *,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        batch_rows: torch.LongTensor,
        seq_positions: torch.LongTensor,
    ) -> None:
        """Refresh cached input embeddings for selected audio positions only."""
        if batch_rows.numel() == 0:
            return
        if batch_rows.shape != seq_positions.shape:
            raise ValueError("batch_rows and seq_positions must have the same shape")
        if inputs_embeds.size(0) != input_ids.size(0):
            raise ValueError("inputs_embeds must match input_ids batch size")

        flat_rows = batch_rows.reshape(-1)
        flat_positions = seq_positions.reshape(-1)
        token_ids = input_ids[flat_rows, :, flat_positions]
        shifted_ids = token_ids + self.codebook_layer_offsets.view(1, -1)
        refreshed = self.audio_embeddings(shifted_ids).sum(dim=1)
        inputs_embeds[flat_rows, flat_positions, :] = refreshed

    def supported_language_ids(self) -> set[str]:
        """Return a list of supported language IDs."""
        return LANG_IDS

    def supported_language_names(self) -> set[str]:
        """Return a list of supported language names."""
        return LANG_NAMES

    # -------------------------------------------------------------------
    # Inference API
    # -------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        text: Union[str, list[str]],
        language: Union[str, list[str], None] = None,
        ref_text: Union[str, list[str], None] = None,
        ref_audio: Union[
            str,
            list[str],
            tuple[torch.Tensor, int],
            list[tuple[torch.Tensor, int]],
            None,
        ] = None,
        voice_clone_prompt: Union[
            VoiceClonePrompt, list[VoiceClonePrompt], None
        ] = None,
        instruct: Union[str, list[str], None] = None,
        duration: Union[float, list[Optional[float]], None] = None,
        speed: Union[float, list[Optional[float]], None] = None,
        generation_config: Optional[OmniVoiceGenerationConfig] = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Generate speech audio given text in various modes.

        Supports three modes:

        1. **Voice clone** — clone the voice style from the reference audio.
            Should provide ``voice_clone_prompt`` (from
           :meth:`create_voice_clone_prompt`) or ``ref_text`` + ``ref_audio``.
        2. **Voice design** — provide ``instruct`` text describing
           the desired voice style; no reference audio needed.
        3. **Auto** — provide neither; the model picks a voice itself.

        Args:
            text: Target text (single string or list for batch).
            language: Language name (e.g. ``"English"``) or code
                (e.g. ``"en"``). ``None`` for language-agnostic mode.
                Performance is slightly better if you specify the language.
            ref_text: Optional reference text for voice cloning mode.
            ref_audio: Optional reference audio for voice cloning mode.
                Can be a file path or a (waveform, sample_rate) tuple.
            voice_clone_prompt: Reusable prompt from :meth:`create_voice_clone_prompt`.
                If provided, it overrides ``ref_text`` and ``ref_audio``.
            instruct: Style instruction for voice design mode.
            duration: Fixed output duration in seconds. If a single float,
                applies to all items; if a list, one value per item.
                ``None`` (default) lets the model estimate duration from text.
                Overrides ``speed`` when both are provided.
            speed: Speaking speed factor. ``> 1.0`` for faster, ``< 1.0`` for
                slower. If a list, one value per item. ``None`` (default) uses
                the model's default estimation.
            generation_config: Explicit config object. If provided, takes
                precedence over ``**kwargs``.
            **kwargs: Generation config or its fields:
                denoise: Whether to prepend the ``<|denoise|>`` token.
                num_step: Number of iterative decoding steps.
                guidance_scale: Classifier-free guidance scale.
                t_shift: Time-step shift (smaller → emphasise low-SNR).
                postprocess_output: Post-process output (remove silence, fade-in/out, pad edges).
                layer_penalty_factor: Penalty encouraging earlier codebook
                    layers to unmask first.
                position_temperature: Temperature for position selection.
                class_temperature: Temperature for token sampling (0 = greedy).
                audio_chunk_duration: If > 0, split long text into chunks of
                    this duration (seconds) and generate chunk by chunk.
                audio_chunk_threshold: Only apply chunking if estimated audio
                    duration exceeds this threshold (seconds).
                batched_decode: Decode equal-length, non-chunked batch outputs
                    in one tokenizer call. This improves throughput but can
                    introduce tiny floating-point differences versus per-item
                    decode, so it is opt-in for the base ``generate`` API.
        Returns:
            ``audios`` a list of 1-D ``np.ndarray`` with shape ``(T,)`` and
            sampling rate consistent with the model's audio tokenizer
            (usually 24 000 Hz).  Can be saved directly with
            ``soundfile.write("out.wav", audios[0], model.sampling_rate)``.
        """

        if self.audio_tokenizer is None or self.text_tokenizer is None:
            raise RuntimeError(
                "Model is not loaded with audio/text tokenizers. Make sure you "
                "loaded the model with OmniVoice.from_pretrained()."
            )
        gen_config = (
            generation_config
            if generation_config is not None
            else OmniVoiceGenerationConfig.from_dict(kwargs)
        )
        profile = _GenerationProfiler(
            enabled=bool(gen_config.collect_profile),
            device=_module_device(self),
        )
        self.last_generation_profile = None

        with profile.section("total_s", sync=gen_config.collect_profile):
            self.eval()

            with profile.section("preprocess_s"):
                full_task = self._preprocess_all(
                    text=text,
                    language=language,
                    ref_text=ref_text,
                    ref_audio=ref_audio,
                    voice_clone_prompt=voice_clone_prompt,
                    instruct=instruct,
                    preprocess_prompt=gen_config.preprocess_prompt,
                    speed=speed,
                    duration=duration,
                )
            profile.metadata.update(
                {
                    "batch_size": full_task.batch_size,
                    "target_lens": list(full_task.target_lens),
                    "total_target_tokens": int(sum(full_task.target_lens)),
                    "max_target_tokens": int(max(full_task.target_lens, default=0)),
                    "num_step": gen_config.num_step,
                    "batched_decode": gen_config.batched_decode,
                    "postprocess_output": gen_config.postprocess_output,
                }
            )

            short_idx, long_idx = full_task.get_indices(
                gen_config, self.audio_tokenizer.config.frame_rate
            )
            profile.metadata["short_batch_size"] = len(short_idx)
            profile.metadata["long_batch_size"] = len(long_idx)

            results = [None] * full_task.batch_size

            if short_idx:
                short_task = full_task.slice_task(short_idx)
                with profile.section("short_iterative_s", sync=True):
                    short_results = self._generate_iterative(
                        short_task,
                        gen_config,
                        profile=profile,
                    )
                for idx, res in zip(short_idx, short_results):
                    results[idx] = res

            if long_idx:
                long_task = full_task.slice_task(long_idx)
                with profile.section("long_chunked_s", sync=True):
                    long_results = self._generate_chunked(long_task, gen_config)
                for idx, res in zip(long_idx, long_results):
                    results[idx] = res

            for i in range(full_task.batch_size):
                assert results[i] is not None, f"Result {i} was not generated"

            with profile.section("decode_and_postprocess_s", sync=True):
                if gen_config.batched_decode:
                    generated_audios = self._decode_batch_and_post_process(
                        results,  # type: ignore[arg-type]
                        full_task.ref_rms,
                        gen_config,
                        profile=profile,
                    )
                else:
                    generated_audios = []
                    for i in range(full_task.batch_size):
                        generated_audios.append(
                            self._decode_and_post_process(
                                results[i],  # type: ignore[arg-type]
                                full_task.ref_rms[i],
                                gen_config,
                                profile=profile,
                            )
                        )

        self.last_generation_profile = profile.to_dict()
        return generated_audios

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, tuple[torch.Tensor, int]],
        ref_text: Optional[str] = None,
        preprocess_prompt: bool = True,
    ) -> VoiceClonePrompt:
        """Create a reusable voice clone prompt from reference audio.

        Args:
            ref_audio: File path (str) or ``(waveform, sample_rate)`` tuple.
                waveform should be a 1-D or 2-D torch.Tensor (channels x samples).
            ref_text: Transcript of the reference audio. If ``None``, the
                ASR model will be used to auto-transcribe (must call
                :meth:`load_asr_model` first).
            preprocess_prompt: If ``True`` (default), apply silence removal and
                trimming to the reference audio, add punctuation in the end
                of reference text (if not already)

        Returns:
            A :class:`VoiceClonePrompt` that can be passed to :meth:`generate`.
        """
        if self.audio_tokenizer is None:
            raise RuntimeError(
                "Audio tokenizer is not loaded. Make sure you loaded the model "
                "with OmniVoice.from_pretrained()."
            )

        if isinstance(ref_audio, str):
            ref_wav = load_audio(ref_audio, self.sampling_rate)
        else:
            waveform, sr = ref_audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            if waveform.shape[0] > 1:
                waveform = np.mean(waveform, axis=0, keepdims=True)
            if sr != self.sampling_rate:
                waveform = torchaudio.functional.resample(
                    torch.from_numpy(waveform),
                    orig_freq=sr,
                    new_freq=self.sampling_rate,
                ).numpy()
            ref_wav = waveform

        ref_rms = float(np.sqrt(np.mean(ref_wav**2)))
        if 0 < ref_rms < 0.1:
            ref_wav = ref_wav * 0.1 / ref_rms

        if preprocess_prompt:
            # Trim long reference audio (>20s) by splitting at the largest silence gap.
            # Skip trimming when ref_text is user-provided, otherwise the
            # trimmed audio will no longer match the full transcript.
            if ref_text is None:
                ref_wav = trim_long_audio(
                    ref_wav, self.sampling_rate, trim_threshold=20.0
                )
            ref_wav = remove_silence(
                ref_wav,
                self.sampling_rate,
                mid_sil=200,
                lead_sil=100,
                trail_sil=200,
            )
            if ref_wav.shape[-1] == 0:
                raise ValueError(
                    "Reference audio is empty after silence removal. "
                    "Try setting preprocess_prompt=False."
                )

        ref_duration = ref_wav.shape[-1] / self.sampling_rate
        if ref_duration > 20.0:
            logger.warning(
                "Reference audio is %.1fs long (>20s). This may cause slower "
                "generation, higher memory usage, and degraded voice cloning "
                "quality. We recommend trimming it to 3-10s.",
                ref_duration,
            )

        # Auto-transcribe if ref_text not provided
        if ref_text is None:
            if self._asr_pipe is None:
                logger.info("ASR model not loaded yet, loading on-the-fly ...")
                self.load_asr_model()
            ref_text = self.transcribe((ref_wav, self.sampling_rate))
            logger.debug("Auto-transcribed ref_text: %s", ref_text)

        chunk_size = self.audio_tokenizer.config.hop_length
        clip_size = int(ref_wav.shape[-1] % chunk_size)
        ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav
        # numpy → torch at tokenizer boundary
        ref_wav_tensor = torch.from_numpy(ref_wav).to(self.audio_tokenizer.device)
        ref_audio_tokens = self.audio_tokenizer.encode(
            ref_wav_tensor.unsqueeze(0),
        ).audio_codes.squeeze(
            0
        )  # (C, T)

        if preprocess_prompt:
            ref_text = add_punctuation(ref_text)

        return VoiceClonePrompt(
            ref_audio_tokens=ref_audio_tokens,
            ref_text=ref_text,
            ref_rms=ref_rms,
        )

    def _decode_and_post_process(
        self,
        tokens: Union[torch.Tensor, List[torch.Tensor]],
        rms: Union[float, None],
        gen_config: OmniVoiceGenerationConfig,
        profile: Optional[_GenerationProfiler] = None,
    ) -> np.ndarray:
        """
        Args:
            tokens: Audio tokens — either a single tensor of shape
                (num_codebooks, seq_len) or a list of chunk tensors.
            rms: RMS of the reference audio for volume adjustment.
            gen_config: Generation config for post-processing options.
        Returns:
            Decoded and post-processed audio array of shape (T,).
        """
        if profile is None:
            profile = _GenerationProfiler(enabled=False, device=_module_device(self))
        tokenizer_device = self.audio_tokenizer.device
        if isinstance(tokens, list):
            chunk_audios = []
            for t in tokens:
                with profile.section("decode_tokenizer_s", sync=True):
                    chunk_audios.append(
                        self.audio_tokenizer.decode(t.to(tokenizer_device).unsqueeze(0))
                        .audio_values[0]
                        .cpu()
                        .numpy()
                    )
            with profile.section("decode_chunk_crossfade_s"):
                audio_waveform = cross_fade_chunks(chunk_audios, self.sampling_rate)
        else:
            with profile.section("decode_tokenizer_s", sync=True):
                audio_waveform = (
                    self.audio_tokenizer.decode(tokens.to(tokenizer_device).unsqueeze(0))
                    .audio_values[0]
                    .cpu()
                    .numpy()
                )

        with profile.section("postprocess_audio_s"):
            audio_waveform = self._post_process_audio(
                audio_waveform,
                postprocess_output=gen_config.postprocess_output,
                ref_rms=rms,
            )
        return audio_waveform.squeeze(0)

    def _decode_batch_and_post_process(
        self,
        results: List[Union[torch.Tensor, List[torch.Tensor]]],
        rms_list: List[Union[float, None]],
        gen_config: OmniVoiceGenerationConfig,
        profile: Optional[_GenerationProfiler] = None,
    ) -> List[np.ndarray]:
        """Decode generated token results, batching equal-length tensors.

        The audio tokenizer accepts batched code tensors, but padding unequal
        lengths can slightly change tail samples.  To keep the output contract
        close, this helper only batches tensors with the exact same token
        length. Chunked generations are decoded by equal-length chunk groups,
        then reassembled per item before the existing crossfade/postprocess
        path.
        """
        if profile is None:
            profile = _GenerationProfiler(enabled=False, device=_module_device(self))
        if len(results) != len(rms_list):
            raise ValueError("results and rms_list must have the same length")

        decoded: List[Optional[np.ndarray]] = [None] * len(results)
        groups: dict[int, list[int]] = {}
        chunk_groups: dict[int, list[tuple[int, int, torch.Tensor]]] = {}
        chunk_audios: dict[int, list[Optional[np.ndarray]]] = {}

        with profile.section("decode_grouping_s"):
            for i, tokens in enumerate(results):
                if isinstance(tokens, list):
                    chunk_audios[i] = [None] * len(tokens)
                    for chunk_idx, chunk_tokens in enumerate(tokens):
                        chunk_groups.setdefault(int(chunk_tokens.size(-1)), []).append(
                            (i, chunk_idx, chunk_tokens)
                        )
                    continue
                groups.setdefault(int(tokens.size(-1)), []).append(i)

        tokenizer_device = self.audio_tokenizer.device
        for entries in chunk_groups.values():
            with profile.section("decode_stack_tokens_s", sync=True):
                batch_tokens = torch.stack(
                    [chunk_tokens.to(tokenizer_device) for _, _, chunk_tokens in entries],
                    dim=0,
                )
            with profile.section("decode_tokenizer_s", sync=True):
                batch_audio = self.audio_tokenizer.decode(batch_tokens).audio_values
            for local_idx, (result_idx, chunk_idx, _) in enumerate(entries):
                chunk_audios[result_idx][chunk_idx] = (
                    batch_audio[local_idx].cpu().numpy()
                )

        for result_idx, chunks in chunk_audios.items():
            assert all(chunk is not None for chunk in chunks), (
                f"Decoded chunks for item {result_idx} are incomplete"
            )
            with profile.section("decode_chunk_crossfade_s"):
                audio_waveform = cross_fade_chunks(
                    chunks,  # type: ignore[arg-type]
                    self.sampling_rate,
                )
            with profile.section("postprocess_audio_s"):
                audio_waveform = self._post_process_audio(
                    audio_waveform,
                    postprocess_output=gen_config.postprocess_output,
                    ref_rms=rms_list[result_idx],
                )
            decoded[result_idx] = audio_waveform.squeeze(0)

        for indices in groups.values():
            if len(indices) == 1:
                i = indices[0]
                decoded[i] = self._decode_and_post_process(
                    results[i],  # type: ignore[arg-type]
                    rms_list[i],
                    gen_config,
                    profile=profile,
                )
                continue

            with profile.section("decode_stack_tokens_s", sync=True):
                batch_tokens = torch.stack(
                    [
                        results[i].to(tokenizer_device)  # type: ignore[union-attr]
                        for i in indices
                    ],
                    dim=0,
                )
            with profile.section("decode_tokenizer_s", sync=True):
                batch_audio = self.audio_tokenizer.decode(batch_tokens).audio_values
            for local_idx, result_idx in enumerate(indices):
                with profile.section("postprocess_audio_s"):
                    audio_waveform = batch_audio[local_idx].cpu().numpy()
                    audio_waveform = self._post_process_audio(
                        audio_waveform,
                        postprocess_output=gen_config.postprocess_output,
                        ref_rms=rms_list[result_idx],
                    )
                decoded[result_idx] = audio_waveform.squeeze(0)

        for i, audio in enumerate(decoded):
            assert audio is not None, f"Decoded audio {i} is missing"
        return decoded  # type: ignore[return-value]

    def _post_process_audio(
        self,
        generated_audio: np.ndarray,
        postprocess_output: bool,
        ref_rms: Union[float, None],
    ) -> np.ndarray:
        """Optionally remove long silences, adjust volume, and add edge padding.

        Args:
            generated_audio: Numpy array of shape (1, T).
            postprocess_output: If True, remove long silences and apply fade/pad.
            ref_rms: RMS of the reference audio for volume normalisation.
        Returns:
            Processed numpy array of shape (1, T).
        """
        if postprocess_output:
            generated_audio = remove_silence(
                generated_audio,
                self.sampling_rate,
                mid_sil=500,
                lead_sil=100,
                trail_sil=100,
            )

        if ref_rms is not None and ref_rms < 0.1:
            generated_audio = generated_audio * ref_rms / 0.1
        elif ref_rms is None:
            peak = np.abs(generated_audio).max()
            if peak > 1e-6:
                generated_audio = generated_audio / peak * 0.5

        generated_audio = fade_and_pad_audio(
            generated_audio,
            sample_rate=self.sampling_rate,
        )
        return generated_audio

    def _generate_chunked(
        self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig
    ) -> List[List[torch.Tensor]]:
        """Generate long audio by splitting text into chunks and batching.

        Each item in the returned list corresponds to one input and contains
        a list of audio token tensors — one per text chunk.

        Args:
            task: A :class:`GenerationTask` with one or more items whose
                estimated audio exceeds ``audio_chunk_threshold``.
            gen_config: Generation config (``audio_chunk_duration`` controls
                chunk size).
        Returns:
            Per-item list of chunk token-tensor lists.
        """
        # Chunk each item's text
        all_chunks = []
        for i in range(task.batch_size):
            avg_tokens_per_char = task.target_lens[i] / len(task.texts[i])
            text_chunk_len = int(
                gen_config.audio_chunk_duration
                * self.audio_tokenizer.config.frame_rate
                / avg_tokens_per_char
            )
            chunks = chunk_text_punctuation(
                text=task.texts[i],
                chunk_len=text_chunk_len,
                min_chunk_len=3,
            )
            logger.debug(f"Item {i} chunked into {len(chunks)} pieces: {chunks}")
            all_chunks.append(chunks)

        has_ref = [t is not None for t in task.ref_audio_tokens]
        assert all(has_ref) or not any(has_ref), (
            "Chunked inference requires all items to either have or not have "
            "ref_audio. Mixed ref/non-ref is not supported."
        )

        max_num_chunks = max(len(c) for c in all_chunks)

        # chunk_results[item_idx] = list of generated token tensors per chunk
        chunk_results = [[] for _ in range(task.batch_size)]

        def _run_batch(indices, texts, ref_audios, ref_texts):
            speed_list = task.speed
            target_lens = [
                self._estimate_target_tokens(
                    texts[j],
                    ref_texts[j],
                    ref_audios[j].size(-1) if ref_audios[j] is not None else None,
                    speed=speed_list[i] if speed_list else 1.0,
                )
                for j, i in enumerate(indices)
            ]
            sub_task = GenerationTask(
                batch_size=len(indices),
                texts=texts,
                target_lens=target_lens,
                langs=[task.langs[i] for i in indices],
                instructs=[task.instructs[i] for i in indices],
                ref_texts=ref_texts,
                ref_audio_tokens=ref_audios,
                ref_rms=[task.ref_rms[i] for i in indices],
                speed=[task.speed[i] for i in indices] if task.speed else None,
            )
            gen_tokens = self._generate_iterative(sub_task, gen_config)
            for j, idx in enumerate(indices):
                chunk_results[idx].append(gen_tokens[j])

        if all(has_ref):
            # All items have reference audio.
            # We still sequentially generate chunks within each item, but we
            # batch across items for the same chunk index. This allows to keep
            # the VRAM usage manageable while still benefiting from batching.
            for ci in range(max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[task.ref_audio_tokens[i] for i in indices],
                    ref_texts=[task.ref_texts[i] for i in indices],
                )
        else:
            # No reference audio — generate chunk 0 for all items first,
            # then use chunk 0 output as reference for all subsequent chunks.
            indices_0 = [i for i in range(task.batch_size) if len(all_chunks[i]) > 0]
            _run_batch(
                indices_0,
                texts=[all_chunks[i][0] for i in indices_0],
                ref_audios=[None] * len(indices_0),
                ref_texts=[None] * len(indices_0),
            )
            first_chunk_map = {idx: chunk_results[idx][0] for idx in indices_0}

            # Batch all remaining chunks, using chunk 0 as fixed reference
            for ci in range(1, max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[first_chunk_map[i] for i in indices],
                    ref_texts=[all_chunks[i][0] for i in indices],
                )

        return chunk_results

    def _preprocess_all(
        self,
        text: Union[str, list[str]],
        language: Union[str, list[str], None] = None,
        ref_text: Union[str, list[str], None] = None,
        ref_audio: Union[
            str,
            list[str],
            tuple[torch.Tensor, int],
            list[tuple[torch.Tensor, int]],
            None,
        ] = None,
        voice_clone_prompt: Union[
            VoiceClonePrompt, list[VoiceClonePrompt], None
        ] = None,
        instruct: Union[str, list[str], None] = None,
        preprocess_prompt: bool = True,
        speed: Union[float, list[Optional[float]], None] = None,
        duration: Union[float, list[Optional[float]], None] = None,
    ) -> GenerationTask:

        if isinstance(text, str):
            text_list = [text]
        else:
            assert isinstance(
                text, list
            ), "text should be a string or a list of strings"
            text_list = text
        batch_size = len(text_list)

        language_list = self._ensure_list(language, batch_size)
        language_list = [_resolve_language(lang) for lang in language_list]
        instruct_list = self._ensure_list(instruct, batch_size)
        for i, s in enumerate(instruct_list):
            if s is None:
                continue
            use_zh = bool(text_list[i] and _ZH_RE.search(text_list[i]))
            instruct_list[i] = _resolve_instruct(s, use_zh=use_zh)

        if voice_clone_prompt is not None and (
            ref_text is not None or ref_audio is not None
        ):
            logger.warning(
                "Both voice_clone_prompt and ref_text/ref_audio are provided. "
                "ref_text/ref_audio will be ignored."
            )
        if voice_clone_prompt is None and ref_audio is not None:
            # If voice_clone_prompt is not provided, create it from
            # ref_audio (ref_text will be auto-transcribed if not given).
            ref_text_list = self._ensure_list(ref_text, batch_size)
            ref_audio_list = self._ensure_list(ref_audio, batch_size)

            voice_clone_prompt = []
            prompt_cache = {}
            for i in range(batch_size):
                cache_key = self._voice_clone_prompt_cache_key(
                    ref_audio_list[i],
                    ref_text_list[i],
                    preprocess_prompt,
                )
                if cache_key is not None and cache_key in prompt_cache:
                    prompt = prompt_cache[cache_key]
                else:
                    prompt = self.create_voice_clone_prompt(
                        ref_audio=ref_audio_list[i],
                        ref_text=ref_text_list[i],
                        preprocess_prompt=preprocess_prompt,
                    )
                    if cache_key is not None:
                        prompt_cache[cache_key] = prompt
                voice_clone_prompt.append(prompt)

        voice_clone_prompt_list = self._ensure_list(voice_clone_prompt, batch_size)
        if voice_clone_prompt_list[0] is not None:
            ref_text_list = [vc.ref_text for vc in voice_clone_prompt_list]
            ref_audio_tokens_list = [
                vc.ref_audio_tokens for vc in voice_clone_prompt_list
            ]
            ref_rms_list = [vc.ref_rms for vc in voice_clone_prompt_list]
        else:
            ref_text_list = [None] * batch_size
            ref_audio_tokens_list = [None] * batch_size
            ref_rms_list = [None] * batch_size

        # Normalize speed/duration to per-item lists (may contain None).
        if speed is not None:
            if isinstance(speed, (int, float)):
                user_speed = [float(speed)] * batch_size
            else:
                user_speed = list(speed)
        else:
            user_speed = None

        if duration is not None:
            if isinstance(duration, (int, float)):
                durations = [float(duration)] * batch_size
            else:
                durations = list(duration)
        else:
            durations = None

        num_target_tokens_list = []
        for i in range(batch_size):
            # duration[i] overrides speed for estimation: use speed=1.0
            # to get the raw estimate, then override target_lens below.
            has_dur = durations is not None and durations[i] is not None
            item_speed = 1.0 if has_dur else (user_speed[i] if user_speed else 1.0)
            est = self._estimate_target_tokens(
                text_list[i],
                ref_text_list[i],
                ref_audio_tokens_list[i].size(-1)
                if ref_audio_tokens_list[i] is not None
                else None,
                speed=item_speed,
            )
            num_target_tokens_list.append(est)

        # Per-item duration overrides: set target_lens to exact frame count
        # and compute speed ratio so chunked generation scales proportionally.
        speed_list: Optional[List[float]] = None
        if durations is not None:
            frame_rate = self.audio_tokenizer.config.frame_rate
            speed_list = []
            for i in range(batch_size):
                if durations[i] is not None:
                    target_tokens = max(1, int(durations[i] * frame_rate))
                    est = num_target_tokens_list[i]
                    speed_list.append(est / target_tokens if target_tokens > 0 else 1.0)
                    num_target_tokens_list[i] = target_tokens
                else:
                    s = user_speed[i] if user_speed else None
                    speed_list.append(s if s is not None else 1.0)
        elif user_speed is not None:
            speed_list = [s if s is not None else 1.0 for s in user_speed]

        return GenerationTask(
            batch_size=batch_size,
            texts=text_list,
            target_lens=num_target_tokens_list,
            langs=language_list,
            instructs=instruct_list,
            ref_texts=ref_text_list,
            ref_audio_tokens=ref_audio_tokens_list,
            ref_rms=ref_rms_list,
            speed=speed_list,
        )

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        """Estimate number of target audio tokens."""
        if num_ref_audio_tokens is None or ref_text is None or len(ref_text) == 0:
            # Fall back to a simple heuristic
            ref_text = "Nice to meet you."
            num_ref_audio_tokens = 25

        est = self.duration_estimator.estimate_duration(
            text, ref_text, num_ref_audio_tokens
        )
        if speed > 0 and speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def _ensure_list(
        self, x: Union[Any, List[Any]], batch_size: int, auto_repeat: bool = True
    ) -> List[Any]:
        x_list = x if isinstance(x, list) else [x]
        if len(x_list) not in (
            1,
            batch_size,
        ):
            raise ValueError(
                f"should be either the number of the text or 1, but got {len(x_list)}"
            )
        if auto_repeat and len(x_list) == 1 and batch_size is not None:
            x_list = x_list * batch_size
        return x_list

    @staticmethod
    def _voice_clone_prompt_cache_key(
        ref_audio: Any,
        ref_text: Optional[str],
        preprocess_prompt: bool,
    ) -> Optional[tuple]:
        if isinstance(ref_audio, str):
            path = os.path.abspath(ref_audio)
            try:
                stat = os.stat(path)
                marker = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                marker = None
            return ("path", path, ref_text, preprocess_prompt, marker)
        tuple_marker = _ref_audio_tuple_cache_marker(ref_audio)
        if tuple_marker is not None:
            return ("memory", tuple_marker, ref_text, preprocess_prompt)
        return None

    def _prepare_inference_inputs(
        self,
        text: str,
        num_target_tokens: int,
        ref_text: Optional[str] = None,
        ref_audio_tokens: Optional[torch.Tensor] = None,
        lang: Optional[str] = None,
        instruct: Optional[str] = None,
        denoise: bool = True,
    ):
        """Prepare input_ids and audio masks for inference.
        Args:
            text: Target text to generate.
            num_target_tokens: Number of audio tokens to generate.
            ref_text: Optional reference text for voice cloning.
            ref_audio_tokens: Optional reference audio tokens for voice cloning.
                with shape (C, T).
            lang: Optional language ID.
            instruct: Optional style instruction for voice design.
            denoise: Whether to include the <|denoise|> token.
        """

        # Build style tokens: <|denoise|> + <|lang_start|>...<|lang_end|>
        #                      + <|instruct_start|>...<|instruct_end|>
        style_text = ""
        if denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        lang_str = lang if lang else "None"
        instruct_str = instruct if instruct else "None"
        style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"

        style_tokens = (
            self.text_tokenizer(style_text, return_tensors="pt")
            .input_ids.repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(
            self.device
        )  # [1, C, N1]

        # Build text tokens
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_tokens = (
            _tokenize_with_nonverbal_tags(wrapped_text, self.text_tokenizer)
            .repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(
            self.device
        )  # [1, C, N2]

        # Target: all MASK
        target_audio_tokens = torch.full(
            (1, self.config.num_audio_codebook, num_target_tokens),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        # Conditional input
        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).to(self.device))
        parts.append(target_audio_tokens)
        cond_input_ids = torch.cat(parts, dim=2)

        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)

        cond_audio_mask = torch.zeros(
            1, cond_total_length, dtype=torch.bool, device=self.device
        )
        cond_audio_mask[0, cond_audio_start_idx:] = True

        return {
            "input_ids": cond_input_ids,
            "audio_mask": cond_audio_mask,
        }

    def _generate_iterative(
        self,
        task: GenerationTask,
        gen_config: OmniVoiceGenerationConfig,
        profile: Optional[_GenerationProfiler] = None,
    ) -> List[torch.Tensor]:
        """N-step iterative unmasked decoding.

        Args:
            task: A :class:`GenerationTask` containing batch texts, target
                lengths, languages, instructions, and optional reference data.
            gen_config: A :class:`OmniVoiceGenerationConfig` controlling
                decoding steps, guidance, temperatures, etc.
        Returns:
            List of generated audio token tensors of shape (C, T) (one per
            input text).
        """

        if profile is None:
            profile = _GenerationProfiler(enabled=False, device=_module_device(self))

        B = task.batch_size

        for i in range(B):
            logger.debug(
                "Item %d — text: %s | ref_text: %s | instruct: %s | lang: %s | target_tokens: %d",
                i,
                task.texts[i],
                task.ref_texts[i],
                task.instructs[i],
                task.langs[i],
                task.target_lens[i],
            )

        with profile.section("iterative_prepare_inputs_s", sync=False):
            inputs_list = [
                self._prepare_inference_inputs(
                    task.texts[i],
                    task.target_lens[i],
                    task.ref_texts[i],
                    task.ref_audio_tokens[i],
                    task.langs[i],
                    task.instructs[i],
                    gen_config.denoise,
                )
                for i in range(B)
            ]

        with profile.section("iterative_pack_inputs_s", sync=True):
            c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
            real_max_c_len = max(c_lens)
            real_max_target_len = max(task.target_lens)
            target_len_pad = _ceil_to_multiple(
                real_max_target_len,
                gen_config.target_len_bucket_multiple,
            )
            max_c_len = _ceil_to_multiple(
                max(real_max_c_len, target_len_pad),
                gen_config.seq_len_bucket_multiple,
            )
            uncond_max_c_len = _ceil_to_multiple(
                target_len_pad,
                gen_config.seq_len_bucket_multiple,
            )
            packed_b = gen_config.batch_size_pad or B
            if packed_b < B:
                raise ValueError("batch_size_pad must be >= batch size")
            pad_id = self.config.audio_mask_id  # Or any other tokens

            batch_input_ids = torch.full(
                (2 * packed_b, self.config.num_audio_codebook, max_c_len),
                pad_id,
                dtype=torch.long,
                device=self.device,
            )
            batch_audio_mask = torch.zeros(
                (2 * packed_b, max_c_len), dtype=torch.bool, device=self.device
            )
            batch_attention_mask = torch.zeros(
                (2 * packed_b, 1, max_c_len, max_c_len),
                dtype=torch.bool,
                device=self.device,
            )

            for i, inp in enumerate(inputs_list):
                c_len, u_len = c_lens[i], task.target_lens[i]

                # Cond (0 ~ B-1)
                batch_input_ids[i, :, :c_len] = inp["input_ids"]
                batch_audio_mask[i, :c_len] = inp["audio_mask"]
                batch_attention_mask[i, :, :c_len, :c_len] = True

                # Uncond (packed_b ~ 2*packed_b-1)
                batch_input_ids[packed_b + i, :, :u_len] = inp["input_ids"][
                    ...,
                    -u_len:,
                ]
                batch_audio_mask[packed_b + i, :u_len] = inp["audio_mask"][
                    ...,
                    -u_len:,
                ]
                batch_attention_mask[packed_b + i, :, :u_len, :u_len] = True
                if max_c_len > u_len:
                    pad_diag = torch.arange(u_len, max_c_len, device=self.device)
                    batch_attention_mask[packed_b + i, :, pad_diag, pad_diag] = True

            if packed_b > B:
                pad_diag = torch.arange(max_c_len, device=self.device)
                for i in range(B, packed_b):
                    batch_attention_mask[i, :, pad_diag, pad_diag] = True
                    batch_attention_mask[packed_b + i, :, pad_diag, pad_diag] = True

            target_slices = [
                (c_lens[i] - task.target_lens[i], c_lens[i]) for i in range(B)
            ]
            target_slices.extend((0, target_len_pad) for _ in range(B, packed_b))
            target_slices.extend((0, task.target_lens[i]) for i in range(B))
            target_slices.extend((0, target_len_pad) for _ in range(B, packed_b))
            target_index, target_valid_mask = self._build_target_slice_gather_index(
                target_slices=target_slices,
                source_len=max_c_len,
                target_len_pad=target_len_pad,
                device=self.device,
            )

            tokens = torch.full(
                (B, self.config.num_audio_codebook, real_max_target_len),
                self.config.audio_mask_id,
                dtype=torch.long,
                device=self.device,
            )

            can_reuse_static_input_embeds = (
                gen_config.reuse_static_input_embeds
                and hasattr(self, "_prepare_embed_inputs")
                and hasattr(self, "_refresh_target_audio_embeds")
            )
            batch_inputs_embeds = None
            if can_reuse_static_input_embeds:
                with profile.section("iterative_prepare_static_embeds_s", sync=True):
                    batch_inputs_embeds = self._prepare_embed_inputs(
                        batch_input_ids,
                        batch_audio_mask,
                    )
            incremental_embed_refresh = batch_inputs_embeds is not None and hasattr(
                self,
                "_refresh_sparse_audio_embeds",
            )

        split_saved_context_tokens = max(0, max_c_len - uncond_max_c_len)
        split_saved_context_ratio = (
            split_saved_context_tokens / max_c_len if max_c_len > 0 else 0.0
        )
        (
            effective_split_guidance_forward,
            split_guidance_decision,
        ) = _resolve_split_guidance_forward(
            mode=gen_config.split_guidance_forward,
            batch_size=B,
            max_context_len=max_c_len,
            uncond_context_len=uncond_max_c_len,
            min_batch_size=gen_config.split_guidance_min_batch_size,
            min_saved_context_ratio=gen_config.split_guidance_min_saved_context_ratio,
        )

        profile.metadata.update(
            {
                "iterative_batch_size": B,
                "packed_batch_size": packed_b,
                "real_max_context_tokens": real_max_c_len,
                "padded_context_tokens": max_c_len,
                "padded_uncond_context_tokens": uncond_max_c_len,
                "split_guidance_saved_context_tokens": split_saved_context_tokens,
                "split_guidance_saved_context_ratio": split_saved_context_ratio,
                "real_max_target_tokens": real_max_target_len,
                "padded_target_tokens": target_len_pad,
                "split_guidance_forward": effective_split_guidance_forward,
                "split_guidance_forward_mode": gen_config.split_guidance_forward,
                "split_guidance_decision": split_guidance_decision,
                "split_guidance_min_batch_size": gen_config.split_guidance_min_batch_size,
                "split_guidance_min_saved_context_ratio": (
                    gen_config.split_guidance_min_saved_context_ratio
                ),
                "reuse_static_input_embeds": bool(batch_inputs_embeds is not None),
                "incremental_static_input_embed_refresh": bool(
                    incremental_embed_refresh
                ),
            }
        )
        split_cond_bidirectional_no_mask = (
            effective_split_guidance_forward
            and packed_b == B
            and all(c_len == max_c_len for c_len in c_lens)
        )
        split_uncond_bidirectional_no_mask = (
            effective_split_guidance_forward
            and packed_b == B
            and all(t_len == target_len_pad for t_len in task.target_lens)
        )
        profile.metadata.update(
            {
                "split_cond_bidirectional_no_mask": bool(
                    split_cond_bidirectional_no_mask
                ),
                "split_uncond_bidirectional_no_mask": bool(
                    split_uncond_bidirectional_no_mask
                ),
            }
        )

        def _call_forward_audio_logits(forward_kwargs):
            call_kwargs = forward_kwargs
            while True:
                try:
                    return self._forward_audio_logits_for_slices(**call_kwargs)
                except TypeError as exc:
                    message = str(exc)
                    if (
                        "unexpected keyword argument 'profile'" in message
                        and "profile" in call_kwargs
                    ):
                        call_kwargs = dict(call_kwargs)
                        call_kwargs.pop("profile", None)
                        continue
                    if (
                        "unexpected keyword argument 'inputs_embeds'" in message
                        and "inputs_embeds" in call_kwargs
                    ):
                        call_kwargs = dict(call_kwargs)
                        call_kwargs.pop("inputs_embeds", None)
                        continue
                    if (
                        "unexpected keyword argument 'bidirectional_no_mask'" in message
                        and "bidirectional_no_mask" in call_kwargs
                    ):
                        call_kwargs = dict(call_kwargs)
                        call_kwargs.pop("bidirectional_no_mask", None)
                        continue
                    raise

        with profile.section("iterative_schedule_s", sync=True):
            timesteps = _get_time_steps(
                t_start=0.0,
                t_end=1.0,
                num_step=gen_config.num_step,
                t_shift=gen_config.t_shift,
            ).tolist()
            schedules = []
            for t_len in task.target_lens:
                total_mask = t_len * self.config.num_audio_codebook
                rem = total_mask
                sched = []
                for step in range(gen_config.num_step):
                    num = (
                        rem
                        if step == gen_config.num_step - 1
                        else min(
                            math.ceil(
                                total_mask * (timesteps[step + 1] - timesteps[step])
                            ),
                            rem,
                        )
                    )
                    sched.append(int(num))
                    rem -= int(num)
                schedules.append(sched)
            update_groups_by_step = self._build_iterative_update_groups(
                target_lens=task.target_lens,
                schedules=schedules,
                device=self.device,
            )

            layer_ids = torch.arange(
                self.config.num_audio_codebook, device=self.device
            ).view(1, -1, 1)

        for step in range(gen_config.num_step):
            if (
                batch_inputs_embeds is not None
                and step > 0
                and not incremental_embed_refresh
            ):
                with profile.section("iterative_refresh_target_embeds_s", sync=True):
                    self._refresh_target_audio_embeds(
                        inputs_embeds=batch_inputs_embeds,
                        input_ids=batch_input_ids,
                        target_index=target_index,
                        target_valid_mask=target_valid_mask,
                    )

            with profile.section("iterative_forward_s", sync=True):
                if effective_split_guidance_forward:
                    cond_forward_kwargs = {
                        "input_ids": batch_input_ids[:packed_b],
                        "audio_mask": batch_audio_mask[:packed_b],
                        "attention_mask": batch_attention_mask[:packed_b],
                        "target_slices": target_slices[:packed_b],
                        "target_len_pad": target_len_pad,
                        "target_index": target_index[:packed_b],
                        "target_valid_mask": target_valid_mask[:packed_b],
                        "bidirectional_no_mask": split_cond_bidirectional_no_mask,
                        "profile": profile,
                    }
                    if batch_inputs_embeds is not None:
                        cond_forward_kwargs["inputs_embeds"] = batch_inputs_embeds[
                            :packed_b
                        ]
                    uncond_forward_kwargs = {
                        "input_ids": batch_input_ids[
                            packed_b:,
                            :,
                            :uncond_max_c_len,
                        ],
                        "audio_mask": batch_audio_mask[
                            packed_b:,
                            :uncond_max_c_len,
                        ],
                        "attention_mask": batch_attention_mask[
                            packed_b:,
                            :,
                            :uncond_max_c_len,
                            :uncond_max_c_len,
                        ],
                        "target_slices": target_slices[packed_b:],
                        "target_len_pad": target_len_pad,
                        "target_index": target_index[packed_b:],
                        "target_valid_mask": target_valid_mask[packed_b:],
                        "bidirectional_no_mask": split_uncond_bidirectional_no_mask,
                        "profile": profile,
                    }
                    if batch_inputs_embeds is not None:
                        uncond_forward_kwargs["inputs_embeds"] = batch_inputs_embeds[
                            packed_b:,
                            :uncond_max_c_len,
                            :,
                        ]
                    with profile.section("iterative_forward_cond_s", sync=True):
                        cond_logits = _call_forward_audio_logits(cond_forward_kwargs)
                    with profile.section("iterative_forward_uncond_s", sync=True):
                        uncond_logits = _call_forward_audio_logits(
                            uncond_forward_kwargs
                        )
                    batch_logits = None
                    split_cond_logits = cond_logits.to(torch.float32)
                    split_uncond_logits = uncond_logits.to(torch.float32)
                else:
                    forward_kwargs = {
                        "input_ids": batch_input_ids,
                        "audio_mask": batch_audio_mask,
                        "attention_mask": batch_attention_mask,
                        "target_slices": target_slices,
                        "target_len_pad": target_len_pad,
                        "target_index": target_index,
                        "target_valid_mask": target_valid_mask,
                        "profile": profile,
                    }
                    if batch_inputs_embeds is not None:
                        forward_kwargs["inputs_embeds"] = batch_inputs_embeds
                    batch_logits = _call_forward_audio_logits(forward_kwargs).to(
                        torch.float32
                    )
                    split_cond_logits = None
                    split_uncond_logits = None

            with profile.section("iterative_update_s", sync=True):
                self._update_iterative_tokens_grouped(
                    batch_logits=batch_logits,
                    cond_logits=split_cond_logits,
                    uncond_logits=split_uncond_logits,
                    batch_input_ids=batch_input_ids,
                    tokens=tokens,
                    c_lens=c_lens,
                    target_lens=task.target_lens,
                    schedules=schedules,
                    step=step,
                    update_groups=update_groups_by_step[step],
                    uncond_offset=packed_b,
                    layer_ids=layer_ids,
                    gen_config=gen_config,
                    inputs_embeds=batch_inputs_embeds
                    if incremental_embed_refresh and step < gen_config.num_step - 1
                    else None,
                )

        with profile.section("iterative_slice_results_s", sync=True):
            return [tokens[i, :, : task.target_lens[i]] for i in range(B)]

    def _update_iterative_tokens_grouped(
        self,
        *,
        batch_logits: Optional[torch.Tensor],
        batch_input_ids: torch.Tensor,
        tokens: torch.Tensor,
        c_lens: list[int],
        target_lens: list[int],
        schedules: list[list[int]],
        step: int,
        uncond_offset: int,
        layer_ids: torch.Tensor,
        gen_config: OmniVoiceGenerationConfig,
        update_groups: Optional[
            list[
                tuple[
                    int,
                    int,
                    torch.LongTensor,
                    torch.LongTensor,
                    tuple[int, ...],
                ]
            ]
        ] = None,
        cond_logits: Optional[torch.Tensor] = None,
        uncond_logits: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        if update_groups is None:
            update_groups = self._build_iterative_update_groups(
                target_lens=target_lens,
                schedules=schedules,
                device=self.device,
            )[step]

        for t_len, k, index_tensor, rows, indices in update_groups:
            if cond_logits is not None and uncond_logits is not None:
                c_logits = cond_logits[index_tensor, :, :t_len, :]
                u_logits = uncond_logits[index_tensor, :, :t_len, :]
            else:
                if batch_logits is None:
                    raise ValueError("batch_logits or cond/uncond logits are required")
                c_logits = batch_logits[index_tensor, :, :t_len, :]
                u_logits = batch_logits[uncond_offset + index_tensor, :, :t_len, :]

            pred_tokens, scores = self._predict_tokens_with_scoring(
                c_logits, u_logits, gen_config
            )
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)

            if gen_config.position_temperature > 0.0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)

            sample_tokens = tokens[index_tensor, :, :t_len]
            scores.masked_fill_(
                sample_tokens != self.config.audio_mask_id, -float("inf")
            )

            _, topk_idx = torch.topk(scores.flatten(start_dim=1), k, dim=1)
            flat_tokens = sample_tokens.flatten(start_dim=1)
            flat_pred_tokens = pred_tokens.flatten(start_dim=1)
            flat_tokens[rows, topk_idx] = flat_pred_tokens[rows, topk_idx]
            updated_tokens = flat_tokens.view_as(sample_tokens)

            tokens[index_tensor, :, :t_len] = updated_tokens
            batch_input_ids[uncond_offset + index_tensor, :, :t_len] = updated_tokens
            for row, i in enumerate(indices):
                c_len = c_lens[i]
                batch_input_ids[i : i + 1, :, c_len - t_len : c_len] = (
                    updated_tokens[row : row + 1]
                )

            if inputs_embeds is not None:
                updated_time_positions = topk_idx.remainder(t_len)
                uncond_rows = (uncond_offset + index_tensor).unsqueeze(1).expand_as(
                    updated_time_positions
                )
                self._refresh_sparse_audio_embeds(
                    inputs_embeds=inputs_embeds,
                    input_ids=batch_input_ids,
                    batch_rows=uncond_rows,
                    seq_positions=updated_time_positions,
                )
                cond_rows = index_tensor.unsqueeze(1).expand_as(
                    updated_time_positions
                )
                cond_offsets = torch.tensor(
                    [c_lens[i] - t_len for i in indices],
                    dtype=updated_time_positions.dtype,
                    device=updated_time_positions.device,
                ).unsqueeze(1)
                self._refresh_sparse_audio_embeds(
                    inputs_embeds=inputs_embeds,
                    input_ids=batch_input_ids,
                    batch_rows=cond_rows,
                    seq_positions=cond_offsets + updated_time_positions,
                )

    def _build_iterative_update_groups(
        self,
        *,
        target_lens: list[int],
        schedules: list[list[int]],
        device: torch.device,
    ) -> list[
        list[tuple[int, int, torch.LongTensor, torch.LongTensor, tuple[int, ...]]]
    ]:
        num_steps = len(schedules[0]) if schedules else 0
        groups_by_step = []
        for step in range(num_steps):
            groups: dict[tuple[int, int], list[int]] = {}
            for i, t_len in enumerate(target_lens):
                k = schedules[i][step]
                if k <= 0:
                    continue
                groups.setdefault((t_len, k), []).append(i)

            step_groups = []
            for (t_len, k), indices in groups.items():
                index_tuple = tuple(indices)
                index_tensor = torch.tensor(index_tuple, dtype=torch.long, device=device)
                rows = torch.arange(len(index_tuple), device=device).unsqueeze(1)
                step_groups.append((t_len, k, index_tensor, rows, index_tuple))
            groups_by_step.append(step_groups)
        return groups_by_step

    def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = torch.log_softmax(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)

        log_probs[..., self.config.audio_mask_id] = -float("inf")

        confidence_scores, greedy_tokens = log_probs.max(dim=-1)

        if gen_config.class_temperature > 0.0:
            filtered_probs = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(
                filtered_probs, gen_config.class_temperature
            ).argmax(dim=-1)
        else:
            pred_tokens = greedy_tokens

        return pred_tokens, confidence_scores


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _resolve_split_guidance_forward(
    *,
    mode: Union[bool, str],
    batch_size: int,
    max_context_len: int,
    uncond_context_len: int,
    min_batch_size: int,
    min_saved_context_ratio: float,
) -> tuple[bool, str]:
    if isinstance(mode, str):
        normalized = mode.strip().lower()
    else:
        normalized = mode

    if normalized in (True, "true", "1", "yes", "y", "on", "force", "forced"):
        return True, "forced_on"
    if normalized in (False, "false", "0", "no", "n", "off", "none"):
        return False, "forced_off"
    if normalized not in ("auto", "adaptive"):
        raise ValueError(
            "split_guidance_forward must be true, false, or auto; "
            f"got {mode!r}"
        )

    saved_context_tokens = max(0, max_context_len - uncond_context_len)
    saved_context_ratio = (
        saved_context_tokens / max_context_len if max_context_len > 0 else 0.0
    )

    if batch_size < min_batch_size:
        return False, "auto_batch_too_small"
    if saved_context_ratio < min_saved_context_ratio:
        return False, "auto_context_savings_too_small"
    return True, "auto_enabled"


def _get_packed_mask(document_ids):
    return partial(_mask_mod_packed, document_ids)


def _mask_mod_packed(document_ids, b, h, q_idx, kv_idx):
    # 1. Sequence Packing Logic: Tokens must belong to the same document.
    # Note: The doc_id for padding tokens is -1, which will automatically not match
    # (if handled correctly) or be ignored.
    same_doc = document_ids[q_idx] == document_ids[kv_idx]
    return same_doc


def _resolve_language(language: Optional[str]) -> Union[str, None]:
    from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID

    if language is None or language.lower() == "none":
        return None
    if language in LANG_IDS:
        return language
    key = language.lower()
    if key in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[key]
    logger.warning(
        f"Language '{language}' is not recognized. "
        f"Please use a valid language ID (e.g., 'en', 'zh', 'ja', 'de') "
        f"or a full language name (e.g., 'English', 'Chinese', 'Japanese'). "
        f"See supported_language_ids() or supported_language_names() for details. "
        f"Falling back to None (language-agnostic mode)."
    )
    return None


def _resolve_instruct(
    instruct: Optional[str], use_zh: bool = False
) -> Union[str, None]:
    """Validate and normalise a voice-design instruct string.

    Supported instruct items (case-insensitive for English):

    English (comma + space separated):
        gender: male, female
        age: child, teenager, young adult, middle-aged, elderly
        pitch: very low pitch, low pitch, moderate pitch,
               high pitch, very high pitch
        style: whisper
        accent: american accent, british accent, australian accent, ...

    Chinese (full-width comma separated):
        gender: 男, 女
        age: 儿童, 少年, 青年, 中年, 老年
        pitch: 极低音调, 低音调, 中音调, 高音调, 极高音调
        style: 耳语
        dialect: 河南话, 陕西话, 四川话, 贵州话, 云南话,
                 桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话,
                 青岛话, 东北话

    Minor issues (auto-fixed):
      - Wrong separator (half-width comma in Chinese instruct or
        full-width comma in English instruct)
      - Leading / trailing commas

    Major issues (raise ``ValueError``):
      - Unsupported or misspelled instruct items
      - Suggestions are offered for close matches

    Args:
        instruct: Raw instruct string, or ``None``.
        use_zh: If True, normalise all items to Chinese (used when the
            synthesis text contains Chinese and no accent is specified).

    Returns:
        Normalised instruct string, or ``None``.

    Raises:
        ValueError: if any instruct item is unsupported or misspelled.
    """
    if instruct is None:
        return None

    instruct_str = instruct.strip()
    if not instruct_str:
        return None

    # Split on both half-width and full-width commas
    raw_items = re.split(r"\s*[,，]\s*", instruct_str)
    raw_items = [x for x in raw_items if x]

    # Validate each item
    unknown = []
    normalised = []
    for raw in raw_items:
        n = raw.strip().lower()
        if n in _INSTRUCT_ALL_VALID:
            normalised.append(n)
        else:
            sug = difflib.get_close_matches(n, _INSTRUCT_ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, n, sug[0] if sug else None))

    if unknown:
        lines = []
        for raw, n, sug in unknown:
            if sug:
                lines.append(f"  '{raw}' -> '{n}' (unsupported; did you mean '{sug}'?)")
            else:
                lines.append(f"  '{raw}' -> '{n}' (unsupported)")
        err = (
            f"Unsupported instruct items found in {instruct_str}:\n"
            + "\n".join(lines)
            + "\n\nValid English items: "
            + ", ".join(sorted(_INSTRUCT_VALID_EN))
            + "\nValid Chinese items: "
            + "，".join(sorted(_INSTRUCT_VALID_ZH))
            + "\n\nTip: Use only English or only Chinese instructs. "
            "English instructs should use comma + space (e.g. "
            "'male, indian accent'),\nChinese instructs should use full-width "
            "comma (e.g. '男，河南话')."
        )
        raise ValueError(err)

    # --- Language consistency: dialect forces Chinese, accent forces English ---
    has_dialect = any(n.endswith("话") for n in normalised)
    has_accent = any(" accent" in n for n in normalised)

    if has_dialect and has_accent:
        raise ValueError(
            "Cannot mix Chinese dialect and English accent in a single instruct. "
            "Dialects are for Chinese speech, accents for English speech."
        )

    if has_dialect:
        use_zh = True
    elif has_accent:
        use_zh = False

    # --- Unify to single language ---
    if use_zh:
        normalised = [_INSTRUCT_EN_TO_ZH.get(n, n) for n in normalised]
    else:
        normalised = [_INSTRUCT_ZH_TO_EN.get(n, n) for n in normalised]

    # --- Category conflict check ---
    conflicts = []
    for cat in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [n for n in normalised if n in cat]
        if len(hits) > 1:
            conflicts.append(hits)
    if conflicts:
        parts = []
        for group in conflicts:
            parts.append(" vs ".join(f"'{x}'" for x in group))
        raise ValueError(
            "Conflicting instruct items within the same category: "
            + "; ".join(parts)
            + ". Each category (gender, age, pitch, style, accent, dialect) "
            "allows at most one item."
        )

    # Determine separator based on language
    has_zh = any(any("\u4e00" <= c <= "\u9fff" for c in n) for n in normalised)
    separator = "，" if has_zh else ", "

    return separator.join(normalised)


def _filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps


_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """Tokenize text containing non-verbal tags, handling each tag independently.

    Non-verbal tags are tokenized standalone to guarantee consistent token
    IDs regardless of surrounding language context (Chinese, English, etc.).

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: HuggingFace text tokenizer instance.
    Returns:
        Token IDs tensor of shape (1, seq_len).
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)

    if not parts:
        result = tokenizer(text, return_tensors="pt").input_ids
    else:
        combined = []
        for p in parts:
            combined.extend(p)
        result = torch.tensor([combined], dtype=torch.long)
    return result


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _combine_text(text, ref_text: Optional[str] = None) -> str:

    # combine with reference text if not None
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()

    # filter out newline / carriage-return characters
    full_text = re.sub(r"[\r\n]+", "", full_text)

    # replace Chinese parentheses with English ones
    full_text = full_text.replace("\uff08", "(").replace("\uff09", ")")

    # collapse consecutive spaces / tabs into a single space
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # remove spaces around chinese characters
    chinese_range = r"[\u4e00-\u9fff]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)

    return full_text


# ---------------------------------------------------------------------------
# Register with HuggingFace Auto classes
# ---------------------------------------------------------------------------

AutoConfig.register("omnivoice", OmniVoiceConfig)
AutoModel.register(OmniVoiceConfig, OmniVoice)
