"""Platform adapter registry for Twitch, Kick, and YouTube Live."""
from typing import Optional

from .base import ChatMessage, PlatformAdapter, StreamInfo, check_ssrf_safe, sanitize_chat_text, sanitize_username
from .kick import KickAdapter
from .twitch import TwitchAdapter
from .youtube import YouTubeAdapter

ADAPTER_CLASSES = {
    "twitch": TwitchAdapter,
    "kick": KickAdapter,
    "youtube": YouTubeAdapter,
}


def detect_platform(url: str) -> str:
    """Return 'twitch', 'kick', 'youtube', or '' if unsupported."""
    for name, cls in ADAPTER_CLASSES.items():
        if cls().matches(url):
            return name
    return ""


def get_adapter(url: str) -> PlatformAdapter:
    """Instantiate the right PlatformAdapter for a URL, or raise ValueError."""
    plat = detect_platform(url)
    if not plat:
        raise ValueError("Stream URL must be from Twitch, Kick, or YouTube Live.")
    adapter = ADAPTER_CLASSES[plat]()
    adapter.validate_url(url)
    return adapter


def validate_live_url(url: str) -> dict:
    """Validate a live stream URL and return its platform, normalized URL, and channel identifier."""
    adapter = get_adapter(url)
    norm = adapter.normalize_url(url)
    ch = adapter.extract_channel(norm)
    return {"platform": adapter.name, "url": norm, "channel": ch}


__all__ = [
    "ADAPTER_CLASSES",
    "ChatMessage",
    "KickAdapter",
    "PlatformAdapter",
    "StreamInfo",
    "TwitchAdapter",
    "YouTubeAdapter",
    "check_ssrf_safe",
    "detect_platform",
    "get_adapter",
    "sanitize_chat_text",
    "sanitize_username",
    "validate_live_url",
]

