import unittest
from vllm_router.prefix_cache import PrefixCacheRouter
from vllm_router.load_balancer import PagedAttentionLoadBalancer, GenerationRequest
from vllm_router.metrics import TokenLatencyTracker

class TestVLLMPagedRouter(unittest.TestCase):

    def setUp(self):
        self.router = PrefixCacheRouter(prefix_token_len=10)
        self.router.register_worker("gpu-node-01", "http://10.0.1.10:8000")
        self.router.register_worker("gpu-node-02", "http://10.0.1.11:8000")
        self.lb = PagedAttentionLoadBalancer(self.router)
        self.tracker = TokenLatencyTracker()

    def test_prefix_cache_affinity(self):
        prompt_a1 = "You are a helpful banking financial intelligence assistant for Flutterwave. Analyze transaction #1"
        prompt_a2 = "You are a helpful banking financial intelligence assistant for Flutterwave. Analyze transaction #2"
        prompt_b = "Translate the following French text into English please."

        # First request for prompt A routes to one worker and warms prefix cache
        w1 = self.router.select_worker_for_prompt(prompt_a1)
        self.assertIn(w1.worker_id, ["gpu-node-01", "gpu-node-02"])

        # Second request sharing identical 10-word system prefix MUST hit the same worker
        w2 = self.router.select_worker_for_prompt(prompt_a2)
        self.assertEqual(w1.worker_id, w2.worker_id, "Expected KV cache affinity routing to same GPU worker")

    def test_load_balancer_lifecycle(self):
        req = GenerationRequest(
            request_id="req-999",
            prompt="Summarize recent transaction activity",
            max_tokens=128
        )
        worker = self.lb.route_request(req)
        self.assertEqual(worker.active_requests, 1)

        # Complete request
        self.lb.complete_request(worker.worker_id, tokens_generated=100)
        self.assertEqual(worker.active_requests, 0)

    def test_latency_metrics(self):
        self.tracker.record_ttft(18.5, cache_hit=True)
        self.tracker.record_ttft(35.0, cache_hit=False)
        self.tracker.record_itl(8.2)

        summary = self.tracker.summary()
        self.assertEqual(summary["total_requests"], 2.0)
        self.assertEqual(summary["cache_hit_ratio_pct"], 50.0)
        self.assertAlmostEqual(summary["avg_itl_ms"], 8.2)

if __name__ == "__main__":
    unittest.main()
