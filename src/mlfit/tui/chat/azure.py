import os
from typing import AsyncGenerator

from mlfit.tui.chat.base import BaseChatProvider


class AzureProvider(BaseChatProvider):
    """
    LLM chat backend using Azure OpenAI (openai SDK).

    Requires: pip install openai

    Reads from .env:
      AZURE_OPENAI_API_KEY    — API key
      AZURE_OPENAI_ENDPOINT   — e.g. https://gpt-4o-mini-us-e.cognitiveservices.azure.com/
      AZURE_OPENAI_DEPLOYMENT — deployment name (default: gpt-4o-mini)
      AZURE_OPENAI_API_VERSION — API version (default: 2024-12-01-preview)
    """

    def __init__(self) -> None:
        try:
            from openai import AsyncAzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for Azure chat. Run: pip install openai"
            ) from exc

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise ValueError(
                "Azure credentials missing. Set AZURE_OPENAI_API_KEY and "
                "AZURE_OPENAI_ENDPOINT in your .env file."
            )

        self._deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    async def send_message(
        self,
        messages: list[dict],
        system_prompt: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Stream an Azure OpenAI reply for the given conversation.

        Args:
            messages: Conversation in {"role": "user"|"assistant", "content": str} format.
            system_prompt: Injected as the first system message if provided.

        Yields:
            Text chunks streamed from the Azure OpenAI API.
        """
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            openai_messages.append({"role": msg["role"], "content": msg["content"]})

        stream = await self._client.chat.completions.create(
            model=self._deployment,
            messages=openai_messages,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
