"""Generic file-based cache for LLM judge outputs.

Keys are versioned, so stale results are invalidated automatically when a judge
prompt changes: each eval sets its own version constant and bumps it when its judge
prompt is modified.

Cache directory structure: {cache_dir}/{md5_key}.json
Key = MD5 of json({"version": str, "judge_model": str, "inputs": dict})
Value = arbitrary JSON-serializable judge output (dict, str, list, etc.)
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class JudgeCache:
    """File-based judge output cache with version-gated invalidation.

    Args:
        cache_dir: Directory to store cache files.
        version: Cache version string. Bump to invalidate all entries
                 (e.g., when the judge prompt changes).
    """

    def __init__(self, cache_dir: Path, version: str):
        self.cache_dir = Path(cache_dir)
        self.version = version
        self._cache: dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self):
        """Load all valid cached entries into memory."""
        if not self.cache_dir.exists():
            return
        n_corrupt = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text())
                if data.get("_version") == self.version:
                    self._cache[cache_file.stem] = data["result"]
            except (json.JSONDecodeError, KeyError):
                n_corrupt += 1
        if n_corrupt > 0:
            print(f"Warning: Skipped {n_corrupt} corrupt judge cache file(s) in {self.cache_dir}")

    def key(self, judge_model: str, **inputs) -> str:
        """Generate cache key from judge model and all input parameters.

        All keyword arguments are included in the key. Pass every input
        that affects the judge output (prompt text, response text, etc.).
        """
        content = json.dumps(
            {
                "version": self.version,
                "judge_model": judge_model,
                "inputs": inputs,
            },
            sort_keys=True,
        )
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached judge output, or None on miss."""
        return self._cache.get(cache_key)

    def set(self, cache_key: str, result: Any):
        """Store judge output in cache."""
        self._cache[cache_key] = result
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "_version": self.version,
                        "result": result,
                    }
                )
            )
        except IOError as e:
            print(f"Warning: Failed to write judge cache file {cache_file}: {e}")

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache
