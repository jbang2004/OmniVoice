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
        for field_name in STRICT_BOOL_CONFIG_FIELDS:
            setattr(self, field_name, ensure_bool(getattr(self, field_name), field_name))
        self.split_guidance_forward = ensure_split_guidance_forward(
            self.split_guidance_forward
        )
        self.num_step = ensure_positive_int(self.num_step, "num_step")
        self.batch_size_pad = ensure_optional_positive_int(
            self.batch_size_pad,
            "batch_size_pad",
        )
        self.seq_len_bucket_multiple = ensure_positive_int(
            self.seq_len_bucket_multiple,
            "seq_len_bucket_multiple",
        )
        self.target_len_bucket_multiple = ensure_positive_int(
            self.target_len_bucket_multiple,
            "target_len_bucket_multiple",
        )
        self.split_guidance_min_batch_size = ensure_positive_int(
            self.split_guidance_min_batch_size,
            "split_guidance_min_batch_size",
        )
        self.t_shift = ensure_positive_float(self.t_shift, "t_shift")
        self.audio_chunk_duration = ensure_positive_float(
            self.audio_chunk_duration,
            "audio_chunk_duration",
        )
        self.audio_chunk_threshold = ensure_positive_float(
            self.audio_chunk_threshold,
            "audio_chunk_threshold",
        )
        self.position_temperature = ensure_non_negative_float(
            self.position_temperature,
            "position_temperature",
        )
        self.class_temperature = ensure_non_negative_float(
            self.class_temperature,
            "class_temperature",
        )
        self.layer_penalty_factor = ensure_non_negative_float(
            self.layer_penalty_factor,
            "layer_penalty_factor",
        )
        self.split_guidance_min_saved_context_ratio = ensure_ratio(
            self.split_guidance_min_saved_context_ratio,
            "split_guidance_min_saved_context_ratio",
        )

    @classmethod
    def from_dict(cls, kwargs_dict):
        valid_keys = {f.name for f in fields(cls)}
        unknown_keys = sorted(set(kwargs_dict) - valid_keys)
        if unknown_keys:
            choices = ", ".join(sorted(valid_keys))
            unknown = ", ".join(unknown_keys)
            raise ValueError(
                f"Unknown generation config field(s): {unknown}. "
                f"Expected one of: {choices}."
            )
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

STRICT_BOOL_CONFIG_FIELDS = (
    "denoise",
    "preprocess_prompt",
    "postprocess_output",
    "batched_decode",
    "collect_profile",
    "reuse_static_input_embeds",
    "enforce_output_duration",
)

SPLIT_GUIDANCE_FORCE_ON_VALUES = frozenset(
    ("true", "1", "yes", "y", "on", "force", "forced")
)
SPLIT_GUIDANCE_FORCE_OFF_VALUES = frozenset(
    ("false", "0", "no", "n", "off", "none")
)
SPLIT_GUIDANCE_AUTO_VALUES = frozenset(("auto", "adaptive"))
SPLIT_GUIDANCE_STRING_VALUES = (
    SPLIT_GUIDANCE_FORCE_ON_VALUES
    | SPLIT_GUIDANCE_FORCE_OFF_VALUES
    | SPLIT_GUIDANCE_AUTO_VALUES
)


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


def ensure_bool(value: bool, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be bool")


def ensure_split_guidance_forward(value: Union[bool, str]) -> Union[bool, str]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SPLIT_GUIDANCE_STRING_VALUES:
            return normalized
    raise ValueError(
        "split_guidance_forward must be true, false, or auto; "
        f"got {value!r}"
    )


def ensure_positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def ensure_optional_positive_int(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    return ensure_positive_int(value, name)


def ensure_positive_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return normalized


def ensure_non_negative_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a non-negative number")
    return normalized


def ensure_ratio(value: float, name: str) -> float:
    normalized = ensure_non_negative_float(value, name)
    if normalized > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return normalized


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
    default_value = ensure_bool(default, f"{name} default")
    values = ensure_optional_bool_list(value, batch_size, name)
    if values is None:
        return [default_value] * batch_size
    return [default_value if item is None else bool(item) for item in values]


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
