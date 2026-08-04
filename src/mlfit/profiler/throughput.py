import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_BENCHMARK_PROMPT = "Explain neural networks briefly:"
_BENCHMARK_MAX_TOKENS = 64


@dataclass
class ThroughputResult:
    """Aggregate throughput measurement at a specific concurrency level."""

    concurrency: int
    tps: float
    total_tokens: int
    elapsed_s: float


def measure_tps(
    base_url: str,
    model_id: str,
    concurrency: int,
    num_requests: int = 50,
) -> ThroughputResult:
    """
    Measure aggregate tokens-per-second by sending concurrent requests.

    Dispatches `num_requests` total requests with at most `concurrency`
    in-flight simultaneously, then divides total tokens by wall-clock time.

    Uses the OpenAI-compatible /v1/completions endpoint.

    Args:
        base_url: Server base URL, e.g. "http://localhost:8000".
        model_id: Model identifier passed in each request body.
        concurrency: Maximum simultaneous in-flight requests.
        num_requests: Total requests to dispatch in the measurement window.

    Returns:
        ThroughputResult with aggregate TPS, total token count, and elapsed time.
    """
    return asyncio.run(
        _async_measure_tps(base_url, model_id, concurrency, num_requests)
    )


async def _async_measure_tps(
    base_url: str,
    model_id: str,
    concurrency: int,
    num_requests: int,
) -> ThroughputResult:
    url = f"{base_url}/v1/completions"
    semaphore = asyncio.Semaphore(concurrency)

    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            _send_one_request(client, url, model_id, semaphore)
            for _ in range(num_requests)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - start

    total_tokens = sum(r for r in results if isinstance(r, int))
    failed = sum(1 for r in results if not isinstance(r, int))

    if failed:
        logger.warning(
            "Throughput benchmark at concurrency=%d: %d/%d requests failed",
            concurrency, failed, num_requests,
        )

    tps = total_tokens / max(elapsed, 0.001)
    logger.info(
        "Throughput at concurrency=%d: %d tokens / %.1fs = %.1f t/s",
        concurrency, total_tokens, elapsed, tps,
    )

    return ThroughputResult(
        concurrency=concurrency,
        tps=tps,
        total_tokens=total_tokens,
        elapsed_s=elapsed,
    )


async def _send_one_request(
    client: httpx.AsyncClient,
    url: str,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> int:
    payload = {
        "model": model_id,
        "prompt": _BENCHMARK_PROMPT,
        "max_tokens": _BENCHMARK_MAX_TOKENS,
    }
    async with semaphore:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("usage", {}).get("completion_tokens", _BENCHMARK_MAX_TOKENS)
