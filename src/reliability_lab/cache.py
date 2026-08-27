from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Shared utilities — use these in both ResponseCache and SharedRedisCache
# ---------------------------------------------------------------------------

PRIVACY_PATTERNS = re.compile(
    r"\b(balance|password|credit.card|ssn|social.security|user.\d+|account.\d+)\b",
    re.IGNORECASE,
)


def _is_uncacheable(query: str) -> bool:
    """Return True if query contains privacy-sensitive keywords."""
    return bool(PRIVACY_PATTERNS.search(query))


def _looks_like_false_hit(query: str, cached_key: str) -> bool:
    """Return True if query and cached key contain different 4-digit numbers (years, IDs)."""
    nums_q = set(re.findall(r"\b\d{4}\b", query))
    nums_c = set(re.findall(r"\b\d{4}\b", cached_key))
    return bool(nums_q and nums_c and nums_q != nums_c)


# ---------------------------------------------------------------------------
# In-memory cache (existing)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CacheEntry:
    key: str
    value: str
    created_at: float
    metadata: dict[str, str]


class ResponseCache:
    """Simple in-memory cache skeleton.

    Add a better semantic similarity function and false-hit guardrails.
    Use the module-level _is_uncacheable() and _looks_like_false_hit() helpers in your
    get() and set() methods. For production, replace with SharedRedisCache.
    """

    def __init__(self, ttl_seconds: int, similarity_threshold: float):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._entries: list[CacheEntry] = []
        self.false_hit_log: list[dict[str, object]] = []

    def get(self, query: str) -> tuple[str | None, float]:
        """Look up a cached response by semantic similarity.

        1. Return (None, 0.0) if _is_uncacheable(query) — privacy check
        2. Evict expired entries (compare time.time() - created_at vs ttl_seconds)
        3. Find best matching entry using self.similarity(query, entry.key)
        4. If best_score >= similarity_threshold:
           a. Check _looks_like_false_hit(query, best_key) — if true, log to
              self.false_hit_log and return (None, best_score)
           b. Otherwise return (best_value, best_score)
        5. Return (None, best_score) if no match above threshold
        """
        if _is_uncacheable(query):
            return None, 0.0

        now = time.time()
        self._entries = [e for e in self._entries if (now - e.created_at) <= self.ttl_seconds]
        if not self._entries:
            return None, 0.0

        best_score = 0.0
        best_entry: CacheEntry | None = None
        for entry in self._entries:
            score = self.similarity(query, entry.key)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            if _looks_like_false_hit(query, best_entry.key):
                self.false_hit_log.append(
                    {"query": query, "cached_key": best_entry.key, "reason": "date_or_number_mismatch"}
                )
                return None, best_score
            return best_entry.value, best_score

        return None, best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response in cache.

        1. Return immediately if _is_uncacheable(query)
        2. Append a CacheEntry to self._entries
        """
        if _is_uncacheable(query):
            return
        self._entries.append(
            CacheEntry(
                key=query,
                value=value,
                created_at=time.time(),
                metadata=metadata or {},
            )
        )

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Compute semantic similarity between two strings.

        Cosine similarity over character n-grams + word tokens.
        1. If a == b, return 1.0
        2. Tokenize both strings: split into words + character n-grams (n=3)
        3. Build Counter vectors from these tokens
        4. Compute cosine similarity: dot(a,b) / (|a| * |b|)
        """
        if a == b:
            return 1.0

        def extract_tokens(text: str) -> list[str]:
            words = re.findall(r"\w+", text.lower())
            tokens: list[str] = list(words)
            for word in words:
                if len(word) >= 3:
                    for i in range(len(word) - 2):
                        tokens.append(word[i : i + 3])
            return tokens

        tokens_a = extract_tokens(a)
        tokens_b = extract_tokens(b)
        if not tokens_a or not tokens_b:
            return 0.0

        vec_a = Counter(tokens_a)
        vec_b = Counter(tokens_b)

        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Redis shared cache (new)
# ---------------------------------------------------------------------------


class SharedRedisCache:
    """Redis-backed shared cache for multi-instance deployments."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        similarity_threshold: float,
        prefix: str = "rl:cache:",
    ):
        import redis as redis_lib

        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.prefix = prefix
        self.false_hit_log: list[dict[str, object]] = []
        self._redis: Any = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return bool(self._redis.ping())
        except Exception:  # noqa: BLE001
            return False

    def get(self, query: str) -> tuple[str | None, float]:
        """Look up a cached response from Redis."""
        if _is_uncacheable(query):
            return None, 0.0

        exact_key = f"{self.prefix}{self._query_hash(query)}"
        exact_response = self._redis.hget(exact_key, "response")
        exact_query = self._redis.hget(exact_key, "query")
        if exact_response is not None and exact_query == query:
            return exact_response, 1.0

        best_score = 0.0
        best_response: str | None = None
        best_cached_query: str | None = None

        for key in self._redis.scan_iter(f"{self.prefix}*"):
            data = self._redis.hgetall(key)
            cached_query = data.get("query")
            cached_response = data.get("response")
            if cached_query and cached_response:
                score = ResponseCache.similarity(query, cached_query)
                if score > best_score:
                    best_score = score
                    best_response = cached_response
                    best_cached_query = cached_query

        if best_response is not None and best_score >= self.similarity_threshold:
            if best_cached_query is not None and _looks_like_false_hit(query, best_cached_query):
                self.false_hit_log.append(
                    {"query": query, "cached_key": best_cached_query, "reason": "date_or_number_mismatch"}
                )
                return None, best_score
            return best_response, best_score

        return None, best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response in Redis with TTL."""
        if _is_uncacheable(query):
            return
        key = f"{self.prefix}{self._query_hash(query)}"
        self._redis.hset(key, mapping={"query": query, "response": value})
        self._redis.expire(key, self.ttl_seconds)

    def flush(self) -> None:
        """Remove all entries with this cache prefix (for testing)."""
        for key in self._redis.scan_iter(f"{self.prefix}*"):
            self._redis.delete(key)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            self._redis.close()

    @staticmethod
    def _query_hash(query: str) -> str:
        """Deterministic short hash for a query string."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]
