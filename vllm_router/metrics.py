import time
from typing import List, Dict

class TokenLatencyTracker:
    """Tracks LLM token generation metrics: TTFT (Time to First Token) and ITL (Inter-Token Latency)."""

    def __init__(self):
        self.ttft_samples_ms: List[float] = []
        self.itl_samples_ms: List[float] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def record_ttft(self, ttft_ms: float, cache_hit: bool):
        self.ttft_samples_ms.append(ttft_ms)
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_itl(self, itl_ms: float):
        self.itl_samples_ms.append(itl_ms)

    def summary(self) -> Dict[str, float]:
        n_ttft = len(self.ttft_samples_ms)
        avg_ttft = sum(self.ttft_samples_ms) / n_ttft if n_ttft > 0 else 0.0

        n_itl = len(self.itl_samples_ms)
        avg_itl = sum(self.itl_samples_ms) / n_itl if n_itl > 0 else 0.0

        total_reqs = self.cache_hits + self.cache_misses
        hit_ratio = (self.cache_hits / total_reqs * 100.0) if total_reqs > 0 else 0.0

        return {
            "avg_ttft_ms": avg_ttft,
            "avg_itl_ms": avg_itl,
            "cache_hit_ratio_pct": hit_ratio,
            "total_requests": float(total_reqs),
        }
