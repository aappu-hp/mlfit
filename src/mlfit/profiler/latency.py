import logging
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = "Tell me about machine learning in one sentence."


@dataclass
class LatencyResult:
    """
    TTFT percentile measurements derived from multiple streaming requests.

    All time values are in milliseconds.
    """

    samples_ms: list

    @property
    def p50_ms(self) -> float:
        """Median time-to-first-token in milliseconds."""
        return statistics.median(self.samples_ms) if self.samples_ms else 0.0

    @property
    def p95_ms(self) -> float:
        """95th-percentile time-to-first-token in milliseconds."""
        if not self.samples_ms:
            return 0.0
        sorted_samples = sorted(self.samples_ms)
        idx = min(int(len(sorted_samples) * 0.95), len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def p99_ms(self) -> float:
        """99th-percentile time-to-first-token in milliseconds."""
        if not self.samples_ms:
            return 0.0
        sorted_samples = sorted(self.samples_ms)
        idx = min(int(len(sorted_samples) * 0.99), len(sorted_samples) - 1)
        return sorted_samples[idx]


def measure_ttft(
    base_url: str,
    model_id: str,
    num_samples: int = 20,
    prompt: str = _DEFAULT_PROMPT,
) -> LatencyResult:
    """
    Measure time-to-first-token by sending streaming requests to the server.

    Uses the OpenAI-compatible /v1/completions endpoint with stream=True.
    The timer starts just before the HTTP request is sent and stops when the
    first non-empty chunk arrives — that delta is the TTFT for one sample.

    Args:
        base_url: Server base URL, e.g. "http://localhost:8000".
        model_id: Model identifier passed in the request body.
        num_samples: Number of requests to send for percentile calculation.
        prompt: Prompt sent in every request (kept constant for comparability).

    Returns:
        LatencyResult containing all raw samples and computed percentiles.

    Raises:
        RuntimeError: If every request fails (server unresponsive).
    """
    url = f"{base_url}/v1/completions"
    samples: list[float] = []

    with httpx.Client(timeout=30.0) as client:
        for i in range(num_samples):
            ttft = _measure_one_ttft(client, url, model_id, prompt)
            if ttft is not None:
                samples.append(ttft)
                logger.debug("TTFT sample %d/%d: %.1f ms", i + 1, num_samples, ttft)
            else:
                logger.warning("TTFT sample %d/%d failed", i + 1, num_samples)

    if not samples:
        raise RuntimeError(
            f"All {num_samples} TTFT requests to {base_url} failed — "
            "the server may be unhealthy or the model not loaded."
        )

    logger.info(
        "TTFT measurement complete: %d samples, p50=%.1f ms, p95=%.1f ms",
        len(samples),
        statistics.median(samples),
        sorted(samples)[min(int(len(samples) * 0.95), len(samples) - 1)],
    )
    return LatencyResult(samples_ms=samples)


def _measure_one_ttft(
    client: httpx.Client, url: str, model_id: str, prompt: str
) -> Optional[float]:
    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": 1,
        "stream": True,
    }
    start = time.perf_counter()
    try:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for chunk in response.iter_text():
                if chunk.strip():
                    return (time.perf_counter() - start) * 1000.0
    except httpx.RequestError as exc:
        logger.debug("TTFT request error: %s", exc)
    except httpx.HTTPStatusError as exc:
        logger.debug("TTFT HTTP error %d: %s", exc.response.status_code, exc)
    return None
