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

"""Data utilities for batch inference and evaluation.

Provides ``read_test_list()`` to parse JSONL test list files used by
``omnivoice.cli.infer_batch`` and evaluation scripts.
"""

import json
import logging
from pathlib import Path


def require_sample_id(sample):
    """Return a stable sample identifier or fail with a clear error."""
    sample_id = sample.get("id")
    if sample_id is not None and str(sample_id).strip():
        return str(sample_id)
    save_name = sample.get("save_name")
    if save_name is not None and str(save_name).strip():
        return str(save_name)
    raise ValueError("Each test-list sample requires id or save_name")


def read_test_list(path):
    """Read a JSONL test list file.

    Each line should be a JSON object.  ``text`` and either ``id`` or
    ``save_name`` are required by inference CLIs; all other fields are optional
    (default to ``None``):
        id, save_name, text, ref_audio, ref_audio_base64, ref_text, voice_id,
        instruct, language_id, language, language_name, duration, speed,
        enforce_output_duration, cost_tokens_hint, context_tokens_hint,
        priority, preprocess_prompt

    Note: ``language_name`` is only used by evaluation scripts (under
    ``omnivoice/eval/``) for grouping and reporting results.  The model
    itself only consumes ``language_id``.

    Returns a list of dicts.
    """
    path = Path(path)
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON at line {line_no}: {line}")
                continue

            sample = {
                "id": obj.get("id"),
                "save_name": obj.get("save_name"),
                "text": obj.get("text"),
                "ref_audio": obj.get("ref_audio"),
                "ref_audio_base64": obj.get("ref_audio_base64"),
                "ref_text": obj.get("ref_text"),
                "voice_id": obj.get("voice_id"),
                "language_id": obj.get("language_id"),
                "language": obj.get("language"),
                "language_name": obj.get("language_name"),
                "duration": obj.get("duration"),
                "speed": obj.get("speed"),
                "enforce_output_duration": obj.get("enforce_output_duration"),
                "cost_tokens_hint": obj.get("cost_tokens_hint"),
                "context_tokens_hint": obj.get("context_tokens_hint"),
                "instruct": obj.get("instruct"),
                "priority": obj.get("priority"),
                "preprocess_prompt": obj.get("preprocess_prompt"),
            }
            samples.append(sample)
    return samples
