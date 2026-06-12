"""Generation configuration and output-duration helpers for OmniVoice."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Optional, Union

import numpy as np


@dataclass
class OmniVoiceGenerationConfig:
    generation_mode: str = "custom"
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
    enforce_output_duration: bool = False

    def __post_init__(self):
        mode = normalize_generation_mode(self.generation_mode)
        self.generation_mode = mode
        for key, value in GENERATION_MODE_PRESETS[mode].items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, kwargs_dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys}
        return cls(**filtered)


GENERATION_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "custom": {},
    "official_compatible": {
        "batched_decode": False,
        "batch_size_pad": None,
        "seq_len_bucket_multiple": 1,
        "target_len_bucket_multiple": 1,
        "split_guidance_forward": False,
        "reuse_static_input_embeds": False,
    },
    "optimized": {
        "batched_decode": True,
        "reuse_static_input_embeds": True,
        "split_guidance_forward": "auto",
    },
}

GENERATION_MODE_ALIASES = {
    "official": "official_compatible",
    "compat": "official_compatible",
    "compatible": "official_compatible",
    "throughput": "optimized",
}


def normalize_generation_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    normalized = GENERATION_MODE_ALIASES.get(normalized, normalized)
    if normalized not in GENERATION_MODE_PRESETS:
        choices = ", ".join(sorted(GENERATION_MODE_PRESETS))
        raise ValueError(
            f"Unknown generation_mode {mode!r}. Expected one of: {choices}."
        )
    return normalized


def resolve_generation_config(
    config: OmniVoiceGenerationConfig,
) -> OmniVoiceGenerationConfig:
    mode = normalize_generation_mode(config.generation_mode)
    preset = GENERATION_MODE_PRESETS[mode]
    if not preset and config.generation_mode == mode:
        return config
    return replace(config, generation_mode=mode, **preset)


def ensure_optional_bool_list(
    value: Union[bool, list[Optional[bool]], None],
    batch_size: int,
    name: str,
) -> Optional[list[Optional[bool]]]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return [bool(value)] * batch_size
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"{name} should be a bool or a list with length 1 or batch "
            f"size {batch_size}, but got {type(value).__name__}"
        )
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} should be a bool or a list with length 1 or batch "
            f"size {batch_size}"
        ) from exc
    if len(values) not in (1, batch_size):
        raise ValueError(
            f"{name} should be a bool or a list with length 1 or batch "
            f"size {batch_size}, but got {len(values)}"
        )
    if len(values) == 1:
        values = values * batch_size

    normalized: list[Optional[bool]] = []
    for item in values:
        if item is None:
            normalized.append(None)
        elif isinstance(item, (bool, np.bool_)):
            normalized.append(bool(item))
        else:
            raise ValueError(f"{name} values must be bool or None")
    return normalized


def resolve_optional_bool_flags(
    value: Union[bool, list[Optional[bool]], None],
    batch_size: int,
    name: str,
    *,
    default: bool,
) -> list[bool]:
    if not isinstance(default, (bool, np.bool_)):
        raise ValueError(f"{name} default must be bool")
    values = ensure_optional_bool_list(value, batch_size, name)
    if values is None:
        return [bool(default)] * batch_size
    return [bool(default) if item is None else bool(item) for item in values]


def fit_audio_to_duration(
    audio: np.ndarray,
    target_duration_s: Optional[float],
    sample_rate: int,
) -> np.ndarray:
    if target_duration_s is None:
        return audio
    target_samples = max(1, int(round(float(target_duration_s) * sample_rate)))
    current_samples = int(audio.shape[-1])
    if current_samples == target_samples:
        return audio
    if current_samples > target_samples:
        return audio[..., :target_samples].copy()

    pad_shape = list(audio.shape)
    pad_shape[-1] = target_samples - current_samples
    padding = np.zeros(pad_shape, dtype=audio.dtype)
    return np.concatenate([audio, padding], axis=-1)
