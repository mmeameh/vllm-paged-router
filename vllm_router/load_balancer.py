from dataclasses import dataclass
from typing import Dict, List, Optional
from .prefix_cache import PrefixCacheRouter, GPUWorkerState

@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@dataclass
class WorkerMetrics:
    worker_id: str
    active_requests: int
    kv_cache_usage_pct: float
    avg_ttft_ms: float
    tokens_per_sec: float

class PagedAttentionLoadBalancer:
    """Enterprise load balancer orchestrating vLLM / TensorRT-LLM GPU instances."""

    def __init__(self, router: PrefixCacheRouter):
        self.router = router

    def route_request(self, req: GenerationRequest) -> GPUWorkerState:
        worker = self.router.select_worker_for_prompt(req.prompt)
        worker.active_requests += 1
        worker.total_tokens_queued += req.max_tokens
        # Estimate incremental KV-cache pressure
        worker.kv_cache_usage_pct = min(1.0, worker.kv_cache_usage_pct + 0.02)
        return worker

    def complete_request(self, worker_id: str, tokens_generated: int):
        if worker_id in self.router.workers:
            w = self.router.workers[worker_id]
            w.active_requests = max(0, w.active_requests - 1)
            w.total_tokens_queued = max(0, w.total_tokens_queued - tokens_generated)
            w.kv_cache_usage_pct = max(0.0, w.kv_cache_usage_pct - 0.02)
