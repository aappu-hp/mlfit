import os
from typing import AsyncGenerator

from mlfit.tui.chat.base import BaseChatProvider

class GeminiProvider(BaseChatProvider):
    """
    LLM chat backend using Google Gemini API (google-genai SDK).

    Reads from .env:
      GEMINI_API_KEY  — API key (required)
      GEMINI_MODEL    — model name (required, e.g. gemini-2.5-flash-lite)

    Uses the aio.chats API for proper multi-turn streaming conversation.
    """

    def __init__(self) -> None:
        try:
            from google import genai as _genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Gemini chat. Run: pip install google-genai"
            ) from exc

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY in your .env file."
            )

        model = os.getenv("GEMINI_MODEL")
        if not model:
            raise ValueError(
                "GEMINI_MODEL is not set. Add GEMINI_MODEL=gemini-2.5-flash-lite to your .env file."
            )

        self._client = _genai.Client(api_key=api_key)
        self._model = model

    async def send_message(
        self,
        messages: list[dict],
        system_prompt: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Stream a Gemini reply for the given conversation.

        Uses aio.chats.create with history so Gemini maintains multi-turn context.

        Args:
            messages: Conversation in {"role": "user"|"assistant", "content": str} format.
                      The last message must be a user message.
            system_prompt: Injected as the system instruction for this chat session.

        Yields:
            Text chunks streamed from the Gemini API.
        """
        from google.genai import types

        # Build prior turns as history (all except the last user message)
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )

        last_message = messages[-1]["content"] if messages else ""

        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
        )

        chat = self._client.aio.chats.create(
            model=self._model,
            history=history,
            config=config,
        )

        stream = await chat.send_message_stream(last_message)
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
