import logging
import shlex
import subprocess
import time
from typing import Optional

import httpx

from mlfit.strategies.base import BackendConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0


class ServerManager:
    """
    Manages the lifecycle of a backend inference server subprocess.

    Starts the server process, polls its health endpoint until ready,
    and terminates it on exit. Supports use as a context manager:

        with ServerManager(config) as server:
            if server.wait_until_ready():
                benchmark(server.base_url)
    """

    def __init__(self, config: BackendConfig, timeout_s: int = 120):
        """
        Initialise the manager without starting the server.

        Args:
            config: BackendConfig supplying the command and server URL.
            timeout_s: Maximum seconds to wait for the health check to pass.
        """
        self._config = config
        self._timeout_s = timeout_s
        self._process: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        """Base URL where the server listens, e.g. 'http://localhost:8000'."""
        return self._config.server_url

    def start(self) -> None:
        """
        Launch the server subprocess.

        Uses server_command when set (e.g. 'ollama serve'), otherwise falls
        back to the user-facing command in BackendConfig.command.

        Raises:
            FileNotFoundError: If the backend executable is not on PATH.
        """
        raw_cmd = self._config.server_command or self._config.command
        cmd = shlex.split(raw_cmd)

        logger.info(
            "Starting %s server: %s", self._config.backend, " ".join(cmd[:4])
        )
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.debug(
            "%s server process started (PID=%d)", self._config.backend, self._process.pid
        )

    def wait_until_ready(self) -> bool:
        """
        Poll the health endpoint until the server responds 200 or times out.

        Returns:
            True if the server became ready within timeout_s, False otherwise.
            Also returns False if the process exits before becoming ready (OOM).
        """
        url = f"{self.base_url}{self._config.health_path}"
        deadline = time.monotonic() + self._timeout_s

        while time.monotonic() < deadline:
            if self._has_crashed():
                logger.warning(
                    "%s process exited with code %d before becoming ready",
                    self._config.backend, self._process.returncode,
                )
                return False

            try:
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    logger.info(
                        "%s server ready at %s", self._config.backend, self.base_url
                    )
                    return True
            except httpx.RequestError:
                pass

            time.sleep(_POLL_INTERVAL_S)

        logger.warning(
            "%s server did not become ready within %ds",
            self._config.backend, self._timeout_s,
        )
        return False

    def is_running(self) -> bool:
        """Return True if the server process is alive."""
        return self._process is not None and self._process.poll() is None

    def stop(self) -> None:
        """Terminate the server process, forcefully killing it if necessary."""
        if self._process and self._process.poll() is None:
            logger.info(
                "Stopping %s server (PID=%d)",
                self._config.backend, self._process.pid,
            )
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate cleanly — sending SIGKILL")
                self._process.kill()
        self._process = None

    def __enter__(self) -> "ServerManager":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def _has_crashed(self) -> bool:
        return self._process is not None and self._process.poll() is not None
