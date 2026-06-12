"""Bounded in-memory voice prompt registry for serving."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class VoicePromptRegistrySnapshot:
    size: int
    max_entries: Optional[int]
    evictions: int
    voice_ids: list[str]


class VoicePromptRegistry(MutableMapping[str, Any]):
    """Small LRU registry for reusable voice-clone prompts.

    The registry deliberately behaves like a mutable mapping so older callers
    that used ``state.voice_prompts[voice_id] = prompt`` keep working.
    """

    def __init__(
        self,
        *,
        max_entries: Optional[int] = 256,
        initial: Optional[Mapping[str, Any]] = None,
    ):
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive or None")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self.evictions = 0
        if initial:
            for key, value in initial.items():
                self[key] = value

    def __getitem__(self, key: str) -> Any:
        value = self._entries[key]
        self._entries.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        self._evict_if_needed()

    def __delitem__(self, key: str) -> None:
        del self._entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._entries:
            return default
        return self[key]

    def put(self, key: str, value: Any) -> None:
        self[key] = value

    def snapshot(self) -> VoicePromptRegistrySnapshot:
        return VoicePromptRegistrySnapshot(
            size=len(self._entries),
            max_entries=self.max_entries,
            evictions=self.evictions,
            voice_ids=list(self._entries.keys()),
        )

    def _evict_if_needed(self) -> None:
        if self.max_entries is None:
            return
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1


__all__ = ["VoicePromptRegistry", "VoicePromptRegistrySnapshot"]

