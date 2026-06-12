#!/usr/bin/env python3
"""Create JSONL traffic mixes for OmniVoice batching benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LONG_TEXTS = [
    "这是一条较长的中文语音请求，用来模拟真实服务中偶尔出现的长句子和长时长任务。",
    "请继续生成较长的测试语音，帮助我们观察异构请求进入批处理队列之后的调度效果。",
    "这个样本故意比其他请求更长，用来检查队头请求是否会阻塞后续可以凑满批次的短请求。",
]

SHORT_TEXTS = [
    "短请求一。",
    "短请求二。",
    "短请求三。",
    "短请求四。",
    "短请求五。",
    "短请求六。",
    "短请求七。",
    "短请求八。",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate heterogeneous JSONL inputs for OmniVoice batching tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--shorts_per_group", type=int, default=4)
    parser.add_argument("--long_duration", type=float, default=2.4)
    parser.add_argument("--short_duration", type=float, default=0.8)
    parser.add_argument(
        "--long_text_repeat",
        type=int,
        default=1,
        help="Repeat long-sample text this many times to create context outliers.",
    )
    parser.add_argument(
        "--long_ref_text_repeat",
        type=int,
        default=1,
        help="Repeat long-sample ref_text this many times to create prompt-context outliers.",
    )
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--language_id", default="zh")
    parser.add_argument("--id_prefix", default="hetero")
    return parser


def build_samples(args) -> list[dict]:
    samples = []
    for group in range(args.groups):
        samples.append(
            _sample(
                sample_id=f"{args.id_prefix}_g{group + 1:02d}_long",
                text=_repeat_text(
                    LONG_TEXTS[group % len(LONG_TEXTS)],
                    args.long_text_repeat,
                ),
                ref_text=_repeat_text(args.ref_text, args.long_ref_text_repeat),
                duration=args.long_duration,
                args=args,
            )
        )
        for short_idx in range(args.shorts_per_group):
            text = SHORT_TEXTS[(group * args.shorts_per_group + short_idx) % len(SHORT_TEXTS)]
            samples.append(
                _sample(
                    sample_id=(
                        f"{args.id_prefix}_g{group + 1:02d}_"
                        f"short{short_idx + 1:02d}"
                    ),
                    text=text,
                    ref_text=args.ref_text,
                    duration=args.short_duration,
                    args=args,
                )
            )
    return samples


def _repeat_text(text: str, repeat: int) -> str:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    return " ".join(text for _ in range(repeat))


def _sample(
    *,
    sample_id: str,
    text: str,
    ref_text: str,
    duration: float,
    args,
) -> dict:
    return {
        "id": sample_id,
        "text": text,
        "ref_audio": args.ref_audio,
        "ref_text": ref_text,
        "language_id": args.language_id,
        "duration": duration,
    }


def main() -> None:
    args = get_parser().parse_args()
    samples = build_samples(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} samples to {output}")


if __name__ == "__main__":
    main()
