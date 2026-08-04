import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """A single point-in-time reading of GPU and system RAM usage."""

    vram_gb: float
    ram_gb: float


class MemoryTracker:
    """
    Polls GPU and RAM usage in a background thread to capture the peak
    memory consumed during a profiling run.

    Supports use as a context manager:

        with MemoryTracker() as tracker:
            load_model()
        peak = tracker.peak_vram_gb()
    """

    def __init__(self, poll_interval_s: float = 0.5):
        """
        Initialise the tracker.

        Args:
            poll_interval_s: Seconds between memory samples.
        """
        self._interval = poll_interval_s
        self._snapshots: list[MemorySnapshot] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Launch the background polling thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.debug("Memory tracker started (interval=%.2fs)", self._interval)

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to finish."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.debug("Memory tracker stopped after %d samples", len(self._snapshots))

    def peak_vram_gb(self) -> float:
        """
        Return the highest VRAM reading seen since start() was called.

        Returns:
            Peak VRAM in GB, or 0.0 if no samples were collected.
        """
        return max((s.vram_gb for s in self._snapshots), default=0.0)

    def peak_ram_gb(self) -> float:
        """
        Return the highest system RAM reading seen since start() was called.

        Returns:
            Peak RAM used in GB, or 0.0 if no samples were collected.
        """
        return max((s.ram_gb for s in self._snapshots), default=0.0)

    def __enter__(self) -> "MemoryTracker":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._snapshots.append(self._take_snapshot())
            except Exception:
                logger.debug("Memory sample failed", exc_info=True)

    def _take_snapshot(self) -> MemorySnapshot:
        import psutil
        used = psutil.virtual_memory().total - psutil.virtual_memory().available
        ram_gb = used / 1e9
        return MemorySnapshot(vram_gb=self._read_vram_gb(), ram_gb=ram_gb)

    def _read_vram_gb(self) -> float:
        # Prefer pynvml for NVIDIA: most accurate GPU-only measurement
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1e9
        except Exception:
            pass

        # Fall back to PyTorch when available on CUDA
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / 1e9
        except Exception:
            pass

        # On Apple Silicon the GPU and CPU share memory — use system RAM as proxy
        import psutil
        used = psutil.virtual_memory().total - psutil.virtual_memory().available
        return used / 1e9
