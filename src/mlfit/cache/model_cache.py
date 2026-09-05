import json
import logging
import pathlib

logger = logging.getLogger(__name__)

SEED_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-72B",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "codellama/CodeLlama-7b-hf",
    "stabilityai/stablelm-2-1_6b",
]


class ModelCache:
    """
    Local disk cache for model config metadata.

    Stores one JSON file per model at ~/.mlfit/models/<org>/<model_id>.json,
    mirroring the HuggingFace model ID path structure. Avoids repeated HF Hub
    HTTP calls for models that have already been analyzed.
    """

    def __init__(self, base_dir: pathlib.Path | None = None):
        self._base_dir = base_dir or pathlib.Path.home() / ".mlfit" / "models"

    @property
    def cache_dir(self) -> pathlib.Path:
        """Root directory where model JSON files are stored."""
        return self._base_dir

    def _path_for(self, model_id: str) -> pathlib.Path:
        parts = model_id.split("/", 1)
        if len(parts) == 2:
            org, name = parts
            return self._base_dir / org / f"{name}.json"
        return self._base_dir / f"{model_id}.json"

    def get(self, model_id: str) -> dict | None:
        """
        Return cached model metadata dict, or None if not cached.

        Args:
            model_id: HuggingFace model ID, e.g. "Qwen/Qwen2.5-7B"

        Returns:
            Parsed JSON dict, or None on a cache miss.
        """
        path = self._path_for(model_id)
        if not path.exists():
            logger.debug("Cache miss: %s", model_id)
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            logger.debug("Cache hit: %s", model_id)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt cache file for %s (%s) — ignoring", model_id, exc)
            return None

    def save(self, model_id: str, data: dict) -> None:
        """
        Persist model metadata dict to disk.

        Args:
            model_id: HuggingFace model ID.
            data: JSON-serialisable dict of model metadata.
        """
        path = self._path_for(model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Cached: %s → %s", model_id, path)

    def list_cached(self) -> list[str]:
        """
        Return a sorted list of all cached model IDs.

        Returns:
            List of model ID strings, e.g. ["Qwen/Qwen2.5-7B", ...].
        """
        if not self._base_dir.exists():
            return []
        results = []
        for json_file in sorted(self._base_dir.rglob("*.json")):
            rel = json_file.relative_to(self._base_dir)
            parts = rel.parts
            if len(parts) == 2:
                org = parts[0]
                name = parts[1].removesuffix(".json")
                results.append(f"{org}/{name}")
            else:
                results.append(rel.stem)
        return results


_default_cache = ModelCache()


def get_default_cache() -> ModelCache:
    """Return the shared default ModelCache instance (~/.mlfit/models/)."""
    return _default_cache
