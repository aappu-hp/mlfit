import os
import pathlib
from abc import ABC, abstractmethod
from typing import AsyncGenerator


def _load_env() -> None:
    """Load .env from project root or ~/.mlfit/.env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    search_paths = [
        pathlib.Path.cwd() / ".env",
        pathlib.Path(__file__).parents[5] / ".env",  # project root
        pathlib.Path.home() / ".mlfit" / ".env",
    ]
    # Also check src/mlfit/.env for dev convenience
    search_paths.append(pathlib.Path(__file__).parents[2] / ".env")

    for path in search_paths:
        if path.exists():
            load_dotenv(path, override=False)
            break


_load_env()


class BaseChatProvider(ABC):
    """
    Pluggable LLM chat backend.

    Concrete implementations (GeminiProvider, AzureProvider) send a conversation
    history to their respective APIs and stream back the assistant reply token by
    token as an async generator.
    """

    @abstractmethod
    async def send_message(
        self,
        messages: list[dict],
        system_prompt: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Stream an assistant reply for the given conversation.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str} dicts
                      in chronological order.
            system_prompt: Optional system instruction injected before the conversation.

        Yields:
            Text chunks of the assistant reply as they arrive from the API.
        """
        ...


def _has_azure_key() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY"))


def _has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def resolve_provider() -> "BaseChatProvider | None":
    """
    Instantiate the LLM provider from environment variables.

    Provider selection order:
      1. MLFIT_LLM_PROVIDER=gemini|azure — explicit override
      2. GEMINI_API_KEY present → GeminiProvider
      3. AZURE_OPENAI_API_KEY present → AzureProvider
      4. None — no key found; caller shows a help message.
    """
    explicit = os.getenv("MLFIT_LLM_PROVIDER", "").lower()

    # Default priority: Gemini > Azure (set MLFIT_LLM_PROVIDER=azure to override)
    if explicit == "gemini" or (not explicit and _has_gemini_key()):
        try:
            from mlfit.tui.chat.gemini import GeminiProvider
            return GeminiProvider()
        except (ImportError, ValueError):
            pass

    if explicit == "azure" or (not explicit and _has_azure_key()):
        try:
            from mlfit.tui.chat.azure import AzureProvider
            return AzureProvider()
        except (ImportError, ValueError):
            pass

    return None
