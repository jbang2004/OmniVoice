"""Step-wise OmniVoice token generation primitives.

This module exposes the inner loop of ``OmniVoice._generate_iterative`` as a
stateful, one-step-at-a-time API. It is the foundation needed for continuous
batching: callers can keep each request's token state, choose any compatible
active set, run one batched model forward, then re-pack the next step with newly
arrived requests.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

from omnivoice.models.generation import OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import GenerationTask
from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample


@dataclass
class StepwiseGenerationState:
    request_id: str
    cond_input_ids: torch.Tensor
    cond_audio_mask: torch.Tensor
    target_len: int
    schedule: list[int]
    step_index: int
    tokens: torch.Tensor
    completed: bool = False

    @property
    def cond_len(self) -> int:
        return int(self.cond_input_ids.size(2))


@dataclass(frozen=True)
class StepwiseStepTimings:
    pack_s: float
    forward_s: float
    update_s: float


def create_stepwise_states(
    model: Any,
    task: GenerationTask,
    gen_config: OmniVoiceGenerationConfig,
    *,
    request_ids: Optional[Iterable[str]] = None,
) -> list[StepwiseGenerationState]:
    ids = list(request_ids) if request_ids is not None else [
        str(i) for i in range(task.batch_size)
    ]
    if len(ids) != task.batch_size:
        raise ValueError("request_ids must match task.batch_size")

    timesteps = _get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=gen_config.num_step,
        t_shift=gen_config.t_shift,
    ).tolist()

    states: list[StepwiseGenerationState] = []
    for i in range(task.batch_size):
        prepared = model._prepare_inference_inputs(
            task.texts[i],
            task.target_lens[i],
            task.ref_texts[i],
            task.ref_audio_tokens[i],
            task.langs[i],
            task.instructs[i],
            gen_config.denoise,
        )
        target_len = task.target_lens[i]
        states.append(
            StepwiseGenerationState(
                request_id=ids[i],
                cond_input_ids=prepared["input_ids"],
                cond_audio_mask=prepared["audio_mask"],
                target_len=target_len,
                schedule=_build_unmask_schedule(
                    target_len=target_len,
                    num_codebooks=model.config.num_audio_codebook,
                    num_step=gen_config.num_step,
                    timesteps=timesteps,
                ),
                step_index=0,
                tokens=torch.full(
                    (model.config.num_audio_codebook, target_len),
                    model.config.audio_mask_id,
                    dtype=torch.long,
                    device=model.device,
                ),
            )
        )
    return states


def run_stepwise_to_completion(
    model: Any,
    states: list[StepwiseGenerationState],
    gen_config: OmniVoiceGenerationConfig,
) -> list[torch.Tensor]:
    while any(not state.completed for state in states):
        run_generation_step(model, states, gen_config)
    return [state.tokens for state in states]


def run_generation_step(
    model: Any,
    states: list[StepwiseGenerationState],
    gen_config: OmniVoiceGenerationConfig,
    *,
    profile_cuda: bool = False,
    batch_size_pad: Optional[int] = None,
    seq_len_pad: Optional[int] = None,
    target_len_pad: Optional[int] = None,
) -> StepwiseStepTimings:
    active = [state for state in states if not state.completed]
    if not active:
        return StepwiseStepTimings(pack_s=0.0, forward_s=0.0, update_s=0.0)

    pack_start = time.monotonic()
    (
        batch_input_ids,
        batch_audio_mask,
        batch_attention_mask,
        target_slices,
        uncond_offset,
    ) = _pack_active_states(
        model,
        active,
        batch_size_pad=batch_size_pad,
        seq_len_pad=seq_len_pad,
        target_len_pad=target_len_pad,
    )
    _sync_if_requested(model, profile_cuda)
    pack_s = time.monotonic() - pack_start

    forward_start = time.monotonic()
    batch_logits = model._forward_audio_logits_for_slices(
        input_ids=batch_input_ids,
        audio_mask=batch_audio_mask,
        attention_mask=batch_attention_mask,
        target_slices=target_slices,
        target_len_pad=target_len_pad,
    ).to(torch.float32)
    _sync_if_requested(model, profile_cuda)
    forward_s = time.monotonic() - forward_start

    layer_ids = torch.arange(
        model.config.num_audio_codebook,
        device=model.device,
    ).view(1, -1, 1)

    update_start = time.monotonic()
    if _try_vectorized_update_same_shape(
        model,
        active,
        batch_logits,
        uncond_offset,
        gen_config,
        layer_ids,
    ):
        pass
    else:
        for i, state in enumerate(active):
            _update_one_state_from_logits(
                model,
                state,
                batch_logits,
                i,
                uncond_offset,
                gen_config,
                layer_ids,
            )

    for state in active:
        state.step_index += 1
        if state.step_index >= gen_config.num_step:
            state.completed = True
    _sync_if_requested(model, profile_cuda)
    update_s = time.monotonic() - update_start
    return StepwiseStepTimings(
        pack_s=pack_s,
        forward_s=forward_s,
        update_s=update_s,
    )


def _update_one_state_from_logits(
    model: Any,
    state: StepwiseGenerationState,
    batch_logits: torch.Tensor,
    active_index: int,
    uncond_offset: int,
    gen_config: OmniVoiceGenerationConfig,
    layer_ids: torch.Tensor,
) -> None:
    k = state.schedule[state.step_index]
    if k <= 0:
        return
    t_len = state.target_len
    c_logits = batch_logits[active_index : active_index + 1, :, :t_len, :]
    u_logits = batch_logits[
        uncond_offset + active_index : uncond_offset + active_index + 1,
        :,
        :t_len,
        :,
    ]
    pred_tokens, scores = model._predict_tokens_with_scoring(
        c_logits,
        u_logits,
        gen_config,
    )
    scores = scores - (layer_ids * gen_config.layer_penalty_factor)
    if gen_config.position_temperature > 0.0:
        scores = _gumbel_sample(scores, gen_config.position_temperature)

    sample_tokens = state.tokens.unsqueeze(0)
    scores.masked_fill_(
        sample_tokens != model.config.audio_mask_id,
        -float("inf"),
    )
    _, topk_idx = torch.topk(scores.flatten(), k)
    flat_tokens = sample_tokens.flatten()
    flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
    state.tokens = flat_tokens.view_as(sample_tokens).squeeze(0)
    state.cond_input_ids[
        ...,
        state.cond_len - state.target_len : state.cond_len,
    ] = state.tokens.unsqueeze(0)


def _try_vectorized_update_same_shape(
    model: Any,
    active: list[StepwiseGenerationState],
    batch_logits: torch.Tensor,
    uncond_offset: int,
    gen_config: OmniVoiceGenerationConfig,
    layer_ids: torch.Tensor,
) -> bool:
    if not active:
        return True
    target_len = active[0].target_len
    step_index = active[0].step_index
    k = active[0].schedule[step_index]
    if any(
        state.target_len != target_len
        or state.step_index >= len(state.schedule)
        or state.schedule[state.step_index] != k
        for state in active
    ):
        return False
    if k <= 0:
        return True

    batch_size = len(active)
    c_logits = batch_logits[:batch_size, :, :target_len, :]
    u_logits = batch_logits[
        uncond_offset : uncond_offset + batch_size,
        :,
        :target_len,
        :,
    ]
    pred_tokens, scores = model._predict_tokens_with_scoring(
        c_logits,
        u_logits,
        gen_config,
    )
    scores = scores - (layer_ids * gen_config.layer_penalty_factor)
    if gen_config.position_temperature > 0.0:
        scores = _gumbel_sample(scores, gen_config.position_temperature)

    sample_tokens = torch.stack([state.tokens for state in active], dim=0)
    scores.masked_fill_(
        sample_tokens != model.config.audio_mask_id,
        -float("inf"),
    )
    _, topk_idx = torch.topk(scores.flatten(start_dim=1), k, dim=1)
    flat_tokens = sample_tokens.flatten(start_dim=1)
    flat_pred_tokens = pred_tokens.flatten(start_dim=1)
    rows = torch.arange(batch_size, device=flat_tokens.device).unsqueeze(1)
    flat_tokens[rows, topk_idx] = flat_pred_tokens[rows, topk_idx]
    updated_tokens = flat_tokens.view_as(sample_tokens)

    for state, tokens in zip(active, updated_tokens):
        state.tokens = tokens
        state.cond_input_ids[
            ...,
            state.cond_len - state.target_len : state.cond_len,
        ] = tokens.unsqueeze(0)
    return True


def _sync_if_requested(model: Any, enabled: bool) -> None:
    if not enabled:
        return
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        is_cuda = device.type == "cuda"
    else:
        is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize(device)


def _build_unmask_schedule(
    *,
    target_len: int,
    num_codebooks: int,
    num_step: int,
    timesteps: list[float],
) -> list[int]:
    total_mask = target_len * num_codebooks
    remaining = total_mask
    schedule = []
    for step in range(num_step):
        if step == num_step - 1:
            num = remaining
        else:
            num = min(
                math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                remaining,
            )
        schedule.append(int(num))
        remaining -= int(num)
    return schedule


def _pack_active_states(
    model: Any,
    active: list[StepwiseGenerationState],
    *,
    batch_size_pad: Optional[int] = None,
    seq_len_pad: Optional[int] = None,
    target_len_pad: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]], int]:
    real_b = len(active)
    packed_b = batch_size_pad or real_b
    if packed_b < real_b:
        raise ValueError("batch_size_pad must be >= number of active states")

    real_max_c_len = max(state.cond_len for state in active)
    real_max_target_len = max(state.target_len for state in active)
    max_c_len = seq_len_pad or real_max_c_len
    max_target_len = target_len_pad or real_max_target_len
    if max_c_len < real_max_c_len:
        raise ValueError("seq_len_pad must be >= max active context length")
    if max_target_len < real_max_target_len:
        raise ValueError("target_len_pad must be >= max active target length")
    if max_c_len < max_target_len:
        raise ValueError("seq_len_pad must be >= target_len_pad")

    pad_id = model.config.audio_mask_id
    batch_input_ids = torch.full(
        (2 * packed_b, model.config.num_audio_codebook, max_c_len),
        pad_id,
        dtype=torch.long,
        device=model.device,
    )
    batch_audio_mask = torch.zeros(
        (2 * packed_b, max_c_len),
        dtype=torch.bool,
        device=model.device,
    )
    batch_attention_mask = torch.zeros(
        (2 * packed_b, 1, max_c_len, max_c_len),
        dtype=torch.bool,
        device=model.device,
    )
    target_slices: list[tuple[int, int]] = []

    for i, state in enumerate(active):
        c_len = state.cond_len
        u_len = state.target_len

        batch_input_ids[i, :, :c_len] = state.cond_input_ids
        batch_audio_mask[i, :c_len] = state.cond_audio_mask
        batch_attention_mask[i, :, :c_len, :c_len] = True
        target_slices.append((c_len - u_len, c_len))

        batch_input_ids[packed_b + i, :, :u_len] = state.tokens
        batch_audio_mask[packed_b + i, :u_len] = state.cond_audio_mask[..., -u_len:]
        batch_attention_mask[packed_b + i, :, :u_len, :u_len] = True
        if max_c_len > u_len:
            pad_diag = torch.arange(u_len, max_c_len, device=model.device)
            batch_attention_mask[packed_b + i, :, pad_diag, pad_diag] = True

    if packed_b > real_b:
        pad_diag = torch.arange(max_c_len, device=model.device)
        for i in range(real_b, packed_b):
            batch_attention_mask[i, :, pad_diag, pad_diag] = True
            batch_attention_mask[packed_b + i, :, pad_diag, pad_diag] = True
            target_slices.append((0, max_target_len))
    target_slices.extend((0, state.target_len) for state in active)
    target_slices.extend((0, max_target_len) for _ in range(real_b, packed_b))

    return batch_input_ids, batch_audio_mask, batch_attention_mask, target_slices, packed_b
