import json
import logging
import pathlib
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ChatHistory:
    """
    Persists a single TUI chat session to ~/.mlfit/chat/<date>_<uuid>.json.

    Each session has a unique ID. When the TUI opens it loads the most recent
    session so context carries over across restarts.
    """

    _CHAT_DIR = pathlib.Path.home() / ".mlfit" / "chat"

    def __init__(
        self,
        hardware_context: dict | None = None,
        session_id: str | None = None,
    ) -> None:
        self._session_id = session_id or str(uuid.uuid4())
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._hardware_context = hardware_context or {}
        self._messages: list[dict] = []
        self._path: pathlib.Path | None = None

    @property
    def session_id(self) -> str:
        """Unique identifier for this chat session."""
        return self._session_id

    @property
    def messages(self) -> list[dict]:
        """Ordered list of {"role", "content", "ts"} message dicts."""
        return self._messages

    def add(self, role: str, content: str) -> None:
        """
        Append a message and persist to disk.

        Args:
            role: "user" or "assistant".
            content: Message text.
        """
        self._messages.append(
            {"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()}
        )
        self._save()

    def _session_path(self) -> pathlib.Path:
        if self._path is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._path = self._CHAT_DIR / f"{date_str}_{self._session_id}.json"
        return self._path

    def _save(self) -> None:
        path = self._session_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "session_id": self._session_id,
                    "started_at": self._started_at,
                    "hardware_context": self._hardware_context,
                    "messages": self._messages,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: pathlib.Path) -> "ChatHistory":
        """
        Load a past session from a JSON file.

        Args:
            path: Absolute path to the session JSON file.

        Returns:
            ChatHistory instance with messages populated from the file.
        """
        with open(path) as f:
            data = json.load(f)
        instance = cls(
            hardware_context=data.get("hardware_context"),
            session_id=data.get("session_id"),
        )
        instance._started_at = data.get("started_at", instance._started_at)
        instance._messages = data.get("messages", [])
        instance._path = path
        return instance

    @classmethod
    def load_most_recent(cls) -> "ChatHistory | None":
        """
        Return the most recently modified session, or None if no sessions exist.

        Returns:
            ChatHistory for the most recent session, or None.
        """
        if not cls._CHAT_DIR.exists():
            return None
        json_files = sorted(cls._CHAT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not json_files:
            return None
        try:
            return cls.load(json_files[-1])
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Could not load most recent chat session: %s", exc)
            return None

    @classmethod
    def list_sessions(cls) -> list[dict]:
        """
        Return a list of past sessions sorted newest-first.

        Returns:
            List of dicts with keys: path, session_id, started_at, message_count.
        """
        if not cls._CHAT_DIR.exists():
            return []
        result = []
        for p in sorted(cls._CHAT_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                with open(p) as f:
                    data = json.load(f)
                result.append(
                    {
                        "path": p,
                        "session_id": data.get("session_id", p.stem),
                        "started_at": data.get("started_at", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue
        return result
