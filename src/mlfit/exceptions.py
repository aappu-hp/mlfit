class MLFitError(Exception):
    """Base exception for mlfit."""
    pass


class ModelNotFoundError(MLFitError):
    """Model not found on HuggingFace or local filesystem."""
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class BackendNotInstalledError(MLFitError):
    """Required backend is not installed."""
    def __init__(self, backend: str, install_command: str = ""):
        self.backend = backend
        self.install_command = install_command or f"pip install {backend}"
        super().__init__(f"{backend} is not installed")


class InsufficientVRAMError(MLFitError):
    """Model too large for available VRAM with no alternatives found."""
    def __init__(self, model_id: str, needed_gb: float, available_gb: float):
        self.model_id = model_id
        self.needed_gb = needed_gb
        self.available_gb = available_gb
        super().__init__(
            f"{model_id} needs {needed_gb:.1f}GB but only {available_gb:.1f}GB available"
        )


class HardwareNotSupportedError(MLFitError):
    """Backend not compatible with current hardware."""
    def __init__(self, backend: str, reason: str = ""):
        self.backend = backend
        super().__init__(f"{backend} is not supported on this hardware: {reason}")
