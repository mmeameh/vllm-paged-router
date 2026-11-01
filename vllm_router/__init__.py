"""
vllm-paged-router: High-Throughput LLM Gateway & PagedAttention KV-Cache Router
Architected by Dr. Ameh Moses Monday (Oxford Ph.D., Senior SWE @ Flutterwave)
"""

from .prefix_cache import PrefixCacheRouter, GPUWorkerState
from .load_balancer import PagedAttentionLoadBalancer, GenerationRequest, WorkerMetrics
from .metrics import TokenLatencyTracker

__version__ = "0.3.0"
__all__ = [
    "PrefixCacheRouter",
    "GPUWorkerState",
    "PagedAttentionLoadBalancer",
    "GenerationRequest",
    "WorkerMetrics",
    "TokenLatencyTracker",
]
