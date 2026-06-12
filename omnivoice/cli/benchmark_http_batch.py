#!/usr/bin/env python3
"""Benchmark an OmniVoice online batch HTTP server with concurrent requests."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Optional

import soundfile as sf

from omnivoice.cli.benchmark_utils import summarize_request_results
from omnivoice.models.generation import ensure_bool
from omnivoice.utils.common import str2bool
from omnivoice.utils.data_utils import read_test_list, require_sample_id


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark /v1/tts on an OmniVoice online batch server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/tts")
    parser.add_argument("--scheduler_url", default=None)
    parser.add_argument("--test_list", required=True)
    parser.add_argument("--res_dir", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--arrival_gap_ms", type=float, default=0.0)
    parser.add_argument(
        "--request_repeats",
        type=int,
        default=1,
        help="Repeat the input test list this many times for one load profile.",
    )
    parser.add_argument("--timeout_s", type=float, default=120.0)
    parser.add_argument("--save_wavs", type=str2bool, default=True)
    parser.add_argument("--pre_register_voices", type=str2bool, default=False)
    parser.add_argument("--voice_register_url", default=None)
    return parser


def _payload_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": require_sample_id(sample),
        "text": sample["text"],
    }
    for source_key, target_key in (
        ("language_id", "language_id"),
        ("language", "language"),
        ("voice_id", "voice_id"),
        ("ref_audio", "ref_audio"),
        ("ref_audio_base64", "ref_audio_base64"),
        ("ref_text", "ref_text"),
        ("instruct", "instruct"),
        ("duration", "duration"),
        ("speed", "speed"),
        ("enforce_output_duration", "enforce_output_duration"),
        ("cost_tokens_hint", "cost_tokens_hint"),
        ("priority", "priority"),
    ):
        value = sample.get(source_key)
        if value is not None:
            payload[target_key] = value
    return payload


def _voice_register_url_from_tts_url(url: str) -> str:
    suffix = "/v1/tts"
    if url.endswith(suffix):
        return url[: -len(suffix)] + "/v1/voices"
    return url.rstrip("/") + "/voices"


def _voice_registration_key(sample: dict[str, Any]) -> Optional[tuple[Any, ...]]:
    if sample.get("voice_id"):
        return None
    ref_audio = sample.get("ref_audio")
    ref_audio_base64 = sample.get("ref_audio_base64")
    if not ref_audio and not ref_audio_base64:
        return None
    if ref_audio and ref_audio_base64:
        raise ValueError("sample provides both ref_audio and ref_audio_base64")
    preprocess_prompt = _optional_preprocess_prompt(sample)
    return (
        "base64" if ref_audio_base64 else "path",
        ref_audio_base64 or ref_audio,
        sample.get("ref_text"),
        preprocess_prompt,
    )


def _voice_id_for_registration_key(key: tuple[Any, ...]) -> str:
    raw = json.dumps(key, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"bench_voice_{digest}"


def _voice_registration_payload(
    sample: dict[str, Any],
    *,
    voice_id: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"voice_id": voice_id}
    for key in ("ref_audio", "ref_audio_base64", "ref_text"):
        value = sample.get(key)
        if value is not None:
            payload[key] = value
    preprocess_prompt = _optional_preprocess_prompt(sample)
    if preprocess_prompt is not None:
        payload["preprocess_prompt"] = preprocess_prompt
    return payload


def _optional_preprocess_prompt(sample: dict[str, Any]) -> Optional[bool]:
    value = sample.get("preprocess_prompt")
    if value is None:
        return None
    return ensure_bool(value, "preprocess_prompt")


def _sample_with_registered_voice_id(
    sample: dict[str, Any],
    *,
    voice_id: str,
) -> dict[str, Any]:
    rewritten = dict(sample)
    rewritten["voice_id"] = voice_id
    rewritten.pop("ref_audio", None)
    rewritten.pop("ref_audio_base64", None)
    rewritten.pop("ref_text", None)
    rewritten.pop("preprocess_prompt", None)
    return rewritten


def _expand_samples_for_request_repeats(
    samples: list[dict[str, Any]],
    *,
    request_repeats: int,
) -> list[dict[str, Any]]:
    if request_repeats <= 0:
        raise ValueError("request_repeats must be positive")
    if request_repeats == 1:
        return samples

    expanded: list[dict[str, Any]] = []
    for repeat_index in range(request_repeats):
        for sample_index, sample in enumerate(samples):
            repeated = dict(sample)
            base_id = (
                sample.get("id")
                or sample.get("save_name")
                or f"sample_{sample_index + 1:04d}"
            )
            repeated["id"] = f"{base_id}_rep{repeat_index + 1}"
            expanded.append(repeated)
    return expanded


async def _pre_register_voices(
    *,
    client: Any,
    voice_register_url: str,
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    voice_ids_by_key: dict[tuple[Any, ...], str] = {}
    source_sample_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for sample in samples:
        key = _voice_registration_key(sample)
        if key is None:
            continue
        if key not in voice_ids_by_key:
            voice_ids_by_key[key] = _voice_id_for_registration_key(key)
            source_sample_by_key[key] = sample

    registrations = []
    for key, voice_id in voice_ids_by_key.items():
        response = await client.post(
            voice_register_url,
            json=_voice_registration_payload(
                source_sample_by_key[key],
                voice_id=voice_id,
            ),
        )
        registration = {
            "voice_id": voice_id,
            "status_code": response.status_code,
        }
        if response.status_code != 200:
            registration["error"] = response.text
            registrations.append(registration)
            raise RuntimeError(
                f"voice registration failed for {voice_id}: "
                f"{response.status_code} {response.text}"
            )
        registration["response"] = response.json()
        registrations.append(registration)

    rewritten = []
    rewritten_count = 0
    for sample in samples:
        key = _voice_registration_key(sample)
        if key is None:
            rewritten.append(sample)
            continue
        rewritten.append(
            _sample_with_registered_voice_id(
                sample,
                voice_id=voice_ids_by_key[key],
            )
        )
        rewritten_count += 1

    return rewritten, {
        "enabled": True,
        "voice_register_url": voice_register_url,
        "num_registered_voices": len(registrations),
        "num_rewritten_samples": rewritten_count,
        "registrations": registrations,
    }


def _float_header(headers: dict[str, str], name: str) -> Optional[float]:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_header(headers: dict[str, str], name: str) -> Optional[int]:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _json_header(headers: dict[str, str], name: str) -> Optional[Any]:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _audio_seconds_from_wav(raw: bytes) -> Optional[float]:
    try:
        with sf.SoundFile(io.BytesIO(raw)) as audio_file:
            return len(audio_file) / float(audio_file.samplerate)
    except Exception:
        return None


def _response_json(response: Any) -> Optional[Any]:
    try:
        return response.json()
    except Exception:
        return None


def _response_error_message(response: Any) -> str:
    payload = _response_json(response)
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        if isinstance(detail, dict):
            code = detail.get("code")
            message = detail.get("message") or detail.get("error")
            if code and message:
                return f"{code}: {message}"
            if message:
                return str(message)
            if code:
                return str(code)
        if isinstance(detail, str):
            return detail
        error = payload.get("error")
        if error is not None:
            return str(error)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    text = getattr(response, "text", "")
    return str(text)


def _result_from_exception(
    *,
    request_id: str,
    exc: BaseException,
    request_wall_s: float,
) -> dict[str, Any]:
    message = str(exc) or exc.__class__.__name__
    return {
        "id": request_id,
        "status_code": None,
        "success": False,
        "request_wall_s": request_wall_s,
        "error": f"{exc.__class__.__name__}: {message}",
    }


def _result_from_response(
    *,
    request_id: str,
    response: Any,
    request_wall_s: float,
    wav_path: Optional[Path] = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": request_id,
        "status_code": response.status_code,
        "success": False,
        "request_wall_s": request_wall_s,
    }
    if response.status_code != 200:
        row["error"] = _response_error_message(response)
        return row

    headers = {key.lower(): value for key, value in response.headers.items()}
    content = getattr(response, "content", b"")
    audio_s = _audio_seconds_from_wav(content)
    row.update(
        {
            "batch_size": _int_header(headers, "x-omnivoice-batch-size"),
            "queue_wait_ms": _float_header(headers, "x-omnivoice-queue-wait-ms"),
            "batch_infer_s": _float_header(headers, "x-omnivoice-batch-infer-s"),
            "batch_reason": headers.get("x-omnivoice-batch-reason"),
            "batch_cost_tokens": _int_header(
                headers,
                "x-omnivoice-batch-cost-tokens",
            ),
            "batch_max_cost_tokens": _int_header(
                headers,
                "x-omnivoice-batch-max-cost-tokens",
            ),
            "batch_context_tokens": _int_header(
                headers,
                "x-omnivoice-batch-context-tokens",
            ),
            "batch_max_context_tokens": _int_header(
                headers,
                "x-omnivoice-batch-max-context-tokens",
            ),
            "batch_context_padding_ratio": _float_header(
                headers,
                "x-omnivoice-batch-context-padding-ratio",
            ),
            "generation_profile": _json_header(
                headers,
                "x-omnivoice-generation-profile",
            ),
            "server_total_s": _float_header(headers, "x-omnivoice-total-s"),
            "audio_s": audio_s,
            "bytes": len(content),
        }
    )
    if audio_s is None or audio_s <= 0.0:
        row["error"] = "invalid_wav_response"
        content_type = headers.get("content-type")
        if content_type:
            row["response_content_type"] = content_type
        error_detail = _response_error_message(response)
        if error_detail:
            row["error_detail"] = error_detail
        return row

    row["success"] = True
    if wav_path is not None:
        wav_path.write_bytes(content)
        row["wav"] = str(wav_path)
    return row


async def _fetch_scheduler_snapshot(client: Any, scheduler_url: Optional[str]):
    if not scheduler_url:
        return None
    try:
        response = await client.get(scheduler_url)
        if response.status_code != 200:
            return {"status_code": response.status_code, "text": response.text}
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


async def _run(args) -> dict[str, Any]:
    import httpx

    res_dir = Path(args.res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    samples = read_test_list(args.test_list)
    samples = _expand_samples_for_request_repeats(
        samples,
        request_repeats=getattr(args, "request_repeats", 1),
    )
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=args.timeout_s) as client:
        voice_registration = {"enabled": False}
        if args.pre_register_voices:
            samples, voice_registration = await _pre_register_voices(
                client=client,
                voice_register_url=args.voice_register_url
                or _voice_register_url_from_tts_url(args.url),
                samples=samples,
            )
        scheduler_before = await _fetch_scheduler_snapshot(client, args.scheduler_url)
        started = time.monotonic()

        async def submit_one(index: int, sample: dict[str, Any]) -> dict[str, Any]:
            if args.arrival_gap_ms > 0:
                await asyncio.sleep(index * args.arrival_gap_ms / 1000.0)
            payload = _payload_from_sample(sample)
            request_id = payload["request_id"]
            async with sem:
                request_started = time.monotonic()
                try:
                    response = await client.post(args.url, json=payload)
                except Exception as exc:
                    request_wall_s = time.monotonic() - request_started
                    return _result_from_exception(
                        request_id=request_id,
                        exc=exc,
                        request_wall_s=request_wall_s,
                    )
                request_wall_s = time.monotonic() - request_started
            wav_path = res_dir / f"{request_id}.wav" if args.save_wavs else None
            return _result_from_response(
                request_id=request_id,
                response=response,
                request_wall_s=request_wall_s,
                wav_path=wav_path,
            )

        tasks = [asyncio.create_task(submit_one(i, s)) for i, s in enumerate(samples)]
        for item in await asyncio.gather(*tasks):
            results.append(item)
        wall_s = time.monotonic() - started
        scheduler_after = await _fetch_scheduler_snapshot(client, args.scheduler_url)

    successful = [row for row in results if row.get("success") is True]
    audio_s = sum(row["audio_s"] or 0.0 for row in successful)
    summary = {
        "num_requests": len(results),
        "num_successful": len(successful),
        "num_failed": len(results) - len(successful),
        "wall_s": wall_s,
        "audio_s": audio_s,
        "rtf_wall": wall_s / audio_s if audio_s > 0 else None,
        "request_repeats": getattr(args, "request_repeats", 1),
        "scheduler_before": scheduler_before,
        "scheduler_after": scheduler_after,
        "voice_registration": voice_registration,
        "request_metrics": summarize_request_results(
            successful,
            batch_size_key="batch_size",
            infer_s_key="batch_infer_s",
        ),
        "results": results,
    }
    summary_path = res_dir / "http_batch_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = get_parser().parse_args()
    summary = asyncio.run(_run(args))
    print(
        json.dumps(
            {
                "num_requests": summary["num_requests"],
                "num_successful": summary["num_successful"],
                "num_failed": summary["num_failed"],
                "wall_s": summary["wall_s"],
                "audio_s": summary["audio_s"],
                "rtf_wall": summary["rtf_wall"],
                "request_metrics": summary["request_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
