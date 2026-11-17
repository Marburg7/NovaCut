"""Base interfaces and data structures for live stream platform adapters (Twitch, Kick, YouTube Live)."""
import ipaddress
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from urllib.parse import urlparse


_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_TAG_RE = re.compile(r"<[^>]{1,120}>")


def sanitize_chat_text(text: str, limit: int = 300) -> str:
    """Strip control characters and HTML tags from untrusted live chat messages."""
    clean = _CTRL_RE.sub(" ", str(text or ""))
    clean = _TAG_RE.sub("", clean)
    return re.sub(r"\s{2,}", " ", clean).strip()[:limit]


def sanitize_username(name: str, limit: int = 60) -> str:
    clean = _CTRL_RE.sub("", str(name or ""))
    clean = re.sub(r"[^\w\-. ]", "", clean).strip()
    return (clean or "viewer")[:limit]


def check_ssrf_safe(url: str, allowed_domains: tuple) -> str:
    """Validate URL scheme, hostname allowlist, and ensure host does not point to private/loopback IPs."""
    raw = str(url or "").strip()
    if not raw:
        raise ValueError("Enter a live stream URL.")
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Stream URL must start with https:// or http://.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("Invalid stream URL.")
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("Private or local network addresses are not permitted for stream URLs.")
    except ValueError as e:
        if "Private or local" in str(e):
            raise
    if not any(host == d or host.endswith("." + d) for d in allowed_domains):
        raise ValueError(f"Unsupported stream host '{host}'. Allowed: {', '.join(allowed_domains)}")
    return raw


@dataclass
class ChatMessage:
    platform: str
    user_id: str
    username: str
    text: str
    timestamp: float          # wall-clock unix timestamp
    stream_sec: float         # synchronized stream clock second
    emotes: List[str] = field(default_factory=list)


@dataclass
class StreamInfo:
    platform: str
    url: str
    channel: str
    title: str
    uploader: str
    stream_urls: List[str]
    is_live: bool = True
    viewers: Optional[int] = None
    started_at: Optional[float] = None
    chat_meta: dict = field(default_factory=dict)


class PlatformAdapter:
    """Abstract base class for a live streaming platform adapter."""
    name: str = "base"
    domains: tuple = ()

    def __init__(self):
        self.lock = threading.Lock()
        self.status: str = "disconnected"     # disconnected | connecting | connected | reconnecting | error
        self.reconnects: int = 0
        self.last_error: str = ""
        self.last_msg_at: float = 0.0
        self.messages_received: int = 0
        self._stop_event: Optional[threading.Event] = None

    def matches(self, url: str) -> bool:
        try:
            raw = url.strip()
            if "://" not in raw:
                raw = "https://" + raw
            host = (urlparse(raw).hostname or "").lower().rstrip(".")
            return any(host == d or host.endswith("." + d) for d in self.domains)
        except Exception:
            return False

    def validate_url(self, url: str) -> str:
        return check_ssrf_safe(url, self.domains)

    def normalize_url(self, url: str) -> str:
        return self.validate_url(url)

    def extract_channel(self, url: str) -> str:
        raise NotImplementedError

    def resolve_stream(self, url: str) -> StreamInfo:
        raise NotImplementedError

    def set_status(self, status: str, error: str = ""):
        with self.lock:
            self.status = status
            if error:
                self.last_error = str(error)[:200]

    def connection_state(self) -> dict:
        with self.lock:
            return {
                "platform": self.name,
                "status": self.status,
                "connected": self.status == "connected",
                "reconnects": self.reconnects,
                "messages_received": self.messages_received,
                "last_msg_at": self.last_msg_at,
                "last_error": self.last_error,
            }

    def record_message(self):
        with self.lock:
            self.messages_received += 1
            self.last_msg_at = time.time()

    def connect_chat(self, url: str, on_message: Callable[[ChatMessage], None],
                     stop_event: threading.Event, stream_clock_fn: Callable[[], float]):
        """Run chat ingestion loop until stop_event is set."""
        raise NotImplementedError

    def disconnect(self):
        if self._stop_event is not None:
            self._stop_event.set()
        self.set_status("disconnected")

