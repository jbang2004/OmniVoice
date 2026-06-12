"""FastAPI app factory for online micro-batched OmniVoice serving."""

import base64
import asyncio
import hashlib
import io
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional

import numpy as np
import soundfile as sf

from omnivoice.serving.batcher import (
    OmniVoiceBatchRequest,
    OmniVoiceBatchScheduler,
)
from omnivoice.serving.voice_registry import VoicePromptRegistry
from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.audio import load_audio_bytes


@dataclass
class StartupWarmupStatus:
    enabled: bool = False
    status: str = "disabled"
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None

    def configure(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.status = "pending" if enabled else "disabled"
        self.started_at = None
        self.completed_at = None
        self.duration_s = None
        self.error = None

    def mark_running(self) -> float:
        self.status = "running"
        self.started_at = time.time()
        self.completed_at = None
        self.duration_s = None
        self.error = None
        return time.monotonic()

    def mark_completed(self, monotonic_started: float) -> None:
        self.status = "completed"
        self.completed_at = time.time()
        self.duration_s = time.monotonic() - monotonic_started
        self.error = None

    def mark_failed(self, monotonic_started: float, exc: BaseException) -> None:
        self.status = "failed"
        self.completed_at = time.time()
        self.duration_s = time.monotonic() - monotonic_started
        message = str(exc)
        self.error = f"{type(exc).__name__}: {message}" if message else type(exc).__name__

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "error": self.error,
        }


@dataclass(frozen=True)
class OnlineBatchServerState:
    scheduler: OmniVoiceBatchScheduler
    sample_rate: int = 24000
    max_request_text_chars: int = 2000
    max_voice_prompts: Optional[int] = 256
    voice_prompts: Optional[
        VoicePromptRegistry | Mapping[str, VoiceClonePrompt]
    ] = None
    startup_warmup: StartupWarmupStatus = field(default_factory=StartupWarmupStatus)

    def __post_init__(self):
        if isinstance(self.voice_prompts, VoicePromptRegistry):
            return
        object.__setattr__(
            self,
            "voice_prompts",
            VoicePromptRegistry(
                max_entries=self.max_voice_prompts,
                initial=self.voice_prompts or {},
            ),
        )


def _require_fastapi():
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import Response
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "FastAPI serving dependencies are not installed. Install with "
            "`pip install omnivoice[serve]` or install fastapi and uvicorn."
        ) from exc
    return Body, FastAPI, HTTPException, Response, BaseModel, Field


def _decode_ref_audio_bytes(raw: bytes, sample_rate: int) -> tuple[np.ndarray, int]:
    audio = load_audio_bytes(raw, sample_rate)
    return audio, sample_rate


def _decode_ref_audio_base64(raw_base64: str, sample_rate: int) -> tuple[np.ndarray, int]:
    try:
        raw = base64.b64decode(raw_base64, validate=True)
    except Exception as exc:
        raise ValueError("ref_audio_base64 is not valid base64") from exc
    return _decode_ref_audio_bytes(raw, sample_rate)


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


def create_online_batch_app(
    state: OnlineBatchServerState,
    startup_warmup: Optional[Callable[[], Awaitable[None]]] = None,
):
    """Create a FastAPI app around a single resident batch scheduler."""

    Body, FastAPI, HTTPException, Response, BaseModel, Field = _require_fastapi()
    state.startup_warmup.configure(enabled=startup_warmup is not None)

    class TTSRequest(BaseModel):
        request_id: Optional[str] = None
        text: str = Field(min_length=1)
        language: Optional[str] = None
        language_id: Optional[str] = None
        voice_id: Optional[str] = None
        ref_audio: Optional[str] = None
        ref_audio_base64: Optional[str] = None
        ref_text: Optional[str] = None
        instruct: Optional[str] = None
        duration: Optional[float] = Field(default=None, gt=0)
        speed: Optional[float] = Field(default=None, gt=0)
        cost_tokens_hint: Optional[int] = Field(default=None, gt=0)
        priority: Literal["normal", "high"] = "normal"

    class TTSBatchRequest(BaseModel):
        requests: list[TTSRequest] = Field(min_length=1)
        response_format: Literal["json_base64"] = "json_base64"

    class VoicePromptRequest(BaseModel):
        voice_id: Optional[str] = None
        ref_audio: Optional[str] = None
        ref_audio_base64: Optional[str] = None
        ref_text: Optional[str] = None
        preprocess_prompt: Optional[bool] = None

    class ResetMetricsRequest(BaseModel):
        reset_prompt_cache_stats: bool = True

    @asynccontextmanager
    async def lifespan(app):
        await state.scheduler.start()
        try:
            if startup_warmup is not None:
                monotonic_started = state.startup_warmup.mark_running()
                try:
                    await startup_warmup()
                except BaseException as exc:
                    state.startup_warmup.mark_failed(monotonic_started, exc)
                    raise
                state.startup_warmup.mark_completed(monotonic_started)
            yield
        finally:
            await state.scheduler.stop()

    app = FastAPI(
        title="OmniVoice Online Batch Server",
        version="0.1",
        lifespan=lifespan,
    )
    app.state.omnivoice = state

    def build_request(payload: TTSRequest) -> OmniVoiceBatchRequest:
        if len(payload.text) > state.max_request_text_chars:
            raise HTTPException(
                status_code=413,
                detail=f"text exceeds {state.max_request_text_chars} characters",
            )
        if payload.ref_audio and payload.ref_audio_base64:
            raise HTTPException(
                status_code=400,
                detail="Provide only one of ref_audio or ref_audio_base64",
            )
        if payload.voice_id and (
            payload.ref_audio or payload.ref_audio_base64 or payload.instruct
        ):
            raise HTTPException(
                status_code=400,
                detail="Provide either voice_id, ref_audio/ref_audio_base64, or instruct",
            )
        if payload.ref_audio is not None and payload.instruct is not None:
            raise HTTPException(
                status_code=400,
                detail="Provide either ref_audio/ref_audio_base64 or instruct, not both",
            )
        if payload.ref_audio_base64 is not None and payload.instruct is not None:
            raise HTTPException(
                status_code=400,
                detail="Provide either ref_audio/ref_audio_base64 or instruct, not both",
            )

        voice_clone_prompt = None
        if payload.voice_id:
            voice_clone_prompt = state.voice_prompts.get(payload.voice_id)
            if voice_clone_prompt is None:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "code": "voice_not_found",
                        "message": f"voice_id {payload.voice_id!r} is not registered",
                    },
                )

        ref_audio: Any = payload.ref_audio
        ref_audio_cache_key = None
        if payload.ref_audio_base64 is not None:
            raw_hash = hashlib.sha256(payload.ref_audio_base64.encode("utf-8")).hexdigest()
            try:
                ref_audio = _decode_ref_audio_base64(
                    payload.ref_audio_base64,
                    state.sample_rate,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            ref_audio_cache_key = ("base64-sha256", raw_hash)

        return OmniVoiceBatchRequest(
            request_id=payload.request_id or uuid.uuid4().hex,
            text=payload.text,
            language=payload.language_id or payload.language,
            ref_audio=ref_audio,
            ref_audio_cache_key=ref_audio_cache_key,
            ref_text=payload.ref_text,
            voice_clone_prompt=voice_clone_prompt,
            instruct=payload.instruct,
            duration=payload.duration,
            speed=payload.speed,
            cost_tokens_hint=payload.cost_tokens_hint,
            priority=payload.priority,
        )

    def build_voice_prompt_audio(payload: VoicePromptRequest) -> tuple[Any, Optional[tuple[Any, ...]]]:
        if payload.ref_audio and payload.ref_audio_base64:
            raise HTTPException(
                status_code=400,
                detail="Provide only one of ref_audio or ref_audio_base64",
            )
        if not payload.ref_audio and not payload.ref_audio_base64:
            raise HTTPException(
                status_code=400,
                detail="Provide ref_audio or ref_audio_base64",
            )
        if payload.ref_audio_base64 is None:
            return payload.ref_audio, None

        raw_hash = hashlib.sha256(payload.ref_audio_base64.encode("utf-8")).hexdigest()
        try:
            ref_audio = _decode_ref_audio_base64(
                payload.ref_audio_base64,
                state.sample_rate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ref_audio, ("base64-sha256", raw_hash)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        snapshot = state.scheduler.snapshot()
        warmup_status = state.startup_warmup.to_dict()
        return {
            "ok": warmup_status["status"] != "failed",
            "scheduler": snapshot.__dict__,
            "voices": state.voice_prompts.snapshot().__dict__,
            "startup_warmup": warmup_status,
        }

    @app.get("/v1/scheduler")
    async def scheduler_snapshot() -> dict[str, Any]:
        return state.scheduler.snapshot().__dict__

    @app.post("/v1/scheduler/reset_metrics")
    async def reset_scheduler_metrics(
        payload: Optional[ResetMetricsRequest] = Body(default=None),
    ) -> dict[str, Any]:
        payload = payload or ResetMetricsRequest()
        state.scheduler.reset_metrics(
            reset_prompt_cache_stats=payload.reset_prompt_cache_stats
        )
        return {
            "reset": True,
            "scheduler": state.scheduler.snapshot().__dict__,
        }

    def map_scheduler_runtime_error(exc: RuntimeError) -> HTTPException:
        message = str(exc)
        if "queue is full" in message:
            return HTTPException(
                status_code=429,
                detail={
                    "code": "queue_full",
                    "message": message,
                },
                headers={"Retry-After": "1"},
            )
        if "scheduler stopped" in message:
            return HTTPException(
                status_code=503,
                detail={
                    "code": "scheduler_stopped",
                    "message": "OmniVoice scheduler is not accepting requests",
                },
            )
        return HTTPException(
            status_code=500,
            detail={
                "code": "generation_failed",
                "message": message or "OmniVoice generation failed",
            },
        )

    @app.post("/v1/voices")
    async def register_voice(payload: VoicePromptRequest = Body(...)) -> dict[str, Any]:
        voice_id = payload.voice_id or uuid.uuid4().hex
        ref_audio, cache_key = build_voice_prompt_audio(payload)
        try:
            prompt = await state.scheduler.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=payload.ref_text,
                preprocess_prompt=payload.preprocess_prompt,
                cache_key=cache_key,
            )
        except RuntimeError as exc:
            raise map_scheduler_runtime_error(exc) from exc
        state.voice_prompts.put(voice_id, prompt)
        return {
            "voice_id": voice_id,
            "ref_text": prompt.ref_text,
            "registered": True,
            "voices": state.voice_prompts.snapshot().__dict__,
        }

    @app.get("/v1/voices")
    async def list_voices() -> dict[str, Any]:
        return state.voice_prompts.snapshot().__dict__

    @app.delete("/v1/voices/{voice_id}")
    async def delete_voice(voice_id: str) -> dict[str, Any]:
        deleted = voice_id in state.voice_prompts
        if deleted:
            del state.voice_prompts[voice_id]
        return {
            "voice_id": voice_id,
            "deleted": deleted,
            "voices": state.voice_prompts.snapshot().__dict__,
        }

    async def submit_request(request: OmniVoiceBatchRequest):
        try:
            return await state.scheduler.submit(request)
        except RuntimeError as exc:
            raise map_scheduler_runtime_error(exc) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "generation_failed",
                    "message": str(exc) or "OmniVoice generation failed",
                },
            ) from exc

    @app.post("/v1/tts")
    async def synthesize(payload: TTSRequest = Body(...)):
        started = time.monotonic()
        result = await submit_request(build_request(payload))
        audio = _wav_bytes(result.audio, result.sample_rate)
        headers = {
            "X-OmniVoice-Request-Id": result.request_id,
            "X-OmniVoice-Batch-Size": str(result.batch_size),
            "X-OmniVoice-Queue-Wait-Ms": f"{result.queue_wait_ms:.3f}",
            "X-OmniVoice-Batch-Infer-S": f"{result.batch_infer_s:.6f}",
            "X-OmniVoice-Batch-Reason": result.batch_reason,
            "X-OmniVoice-Batch-Cost-Tokens": str(result.batch_cost_tokens),
            "X-OmniVoice-Batch-Max-Cost-Tokens": str(
                result.batch_max_cost_tokens
            ),
            "X-OmniVoice-Batch-Context-Tokens": str(result.batch_context_tokens),
            "X-OmniVoice-Batch-Max-Context-Tokens": str(
                result.batch_max_context_tokens
            ),
            "X-OmniVoice-Batch-Context-Padding-Ratio": (
                f"{result.batch_context_padding_ratio:.6f}"
            ),
            "X-OmniVoice-Total-S": f"{time.monotonic() - started:.6f}",
        }
        generation_profile = getattr(result, "generation_profile", {}) or {}
        if generation_profile:
            headers["X-OmniVoice-Generation-Profile"] = json.dumps(
                generation_profile,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return Response(content=audio, media_type="audio/wav", headers=headers)

    async def scheduler_submit_many(requests: list[OmniVoiceBatchRequest]):
        return await asyncio.gather(*[submit_request(request) for request in requests])

    @app.post("/v1/tts_batch")
    async def synthesize_batch(payload: TTSBatchRequest = Body(...)) -> dict[str, Any]:
        started = time.monotonic()
        results = await scheduler_submit_many(
            [build_request(item) for item in payload.requests]
        )
        return {
            "total_s": time.monotonic() - started,
            "scheduler": state.scheduler.snapshot().__dict__,
            "results": [
                {
                    "request_id": result.request_id,
                    "sample_rate": result.sample_rate,
                    "audio_s": len(result.audio) / result.sample_rate,
                    "audio_base64": base64.b64encode(
                        _wav_bytes(result.audio, result.sample_rate)
                    ).decode("ascii"),
                    "batch_size": result.batch_size,
                    "queue_wait_ms": result.queue_wait_ms,
                    "batch_infer_s": result.batch_infer_s,
                    "batch_reason": result.batch_reason,
                    "batch_cost_tokens": result.batch_cost_tokens,
                    "batch_max_cost_tokens": result.batch_max_cost_tokens,
                    "batch_context_tokens": result.batch_context_tokens,
                    "batch_max_context_tokens": result.batch_max_context_tokens,
                    "batch_context_padding_ratio": (
                        result.batch_context_padding_ratio
                    ),
                    "generation_profile": (
                        getattr(result, "generation_profile", {}) or {}
                    ),
                }
                for result in results
            ],
        }
    return app


__all__ = [
    "OnlineBatchServerState",
    "StartupWarmupStatus",
    "create_online_batch_app",
]
