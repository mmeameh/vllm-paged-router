import hashlib
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

@dataclass
class GPUWorkerState:
    worker_id: str
    endpoint: str
    active_requests: int = 0
    total_tokens_queued: int = 0
    kv_cache_usage_pct: float = 0.0
    cached_prefixes: Set[str] = field(default_factory=set)

class PrefixCacheRouter:
    """Intelligently dispatches prompt requests to GPU workers that hold hot PagedAttention KV-cache blocks."""

    def __init__(self, prefix_token_len: int = 64):
        self.prefix_token_len = prefix_token_len
        self.workers: Dict[str, GPUWorkerState] = {}

    def register_worker(self, worker_id: str, endpoint: str):
        self.workers[worker_id] = GPUWorkerState(worker_id=worker_id, endpoint=endpoint)

    def remove_worker(self, worker_id: str):
        if worker_id in self.workers:
            del self.workers[worker_id]

    def compute_prefix_hash(self, prompt: str) -> str:
        """Hashes the initial token slice (system prompt / context) to index KV cache locality."""
        # Simple tokenization proxy using whitespace words for portability
        words = prompt.strip().split()
        prefix = " ".join(words[:self.prefix_token_len])
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]

    def select_worker_for_prompt(self, prompt: str) -> GPUWorkerState:
        """Finds worker with cache hit; otherwise selects worker with minimum load."""
        if not self.workers:
            raise RuntimeError("No GPU workers registered in cluster pool")

        prefix_hash = self.compute_prefix_hash(prompt)

        # 1. Check for cache hit among available workers
        cache_hit_candidates = [
            w for w in self.workers.values()
            if prefix_hash in w.cached_prefixes and w.kv_cache_usage_pct < 0.90
        ]

        if cache_hit_candidates:
            # Pick worker with least active load among cache hits
            best_worker = min(cache_hit_candidates, key=lambda w: w.active_requests)
            return best_worker

        # 2. Cache miss: pick healthiest worker with lowest KV-cache utilization and load
        best_worker = min(
            self.workers.values(),
            key=lambda w: (w.kv_cache_usage_pct, w.active_requests)
        )
        # Register prefix for future cache hits
        best_worker.cached_prefixes.add(prefix_hash)
        return best_worker
