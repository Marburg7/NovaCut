"""YouTube Live Stream & Innertube Live Chat Adapter (Zero API Keys Required)."""
import json
import re
import threading
import time
import urllib.request
from typing import Callable, List
from urllib.parse import parse_qs, urlparse

from .base import ChatMessage, PlatformAdapter, StreamInfo, sanitize_chat_text, sanitize_username

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


class YouTubeAdapter(PlatformAdapter):
    name = "youtube"
    domains = ("youtube.com", "youtu.be")

    def extract_channel(self, url: str) -> str:
        clean = self.validate_url(url)
        parsed = urlparse(clean)
        host = (parsed.hostname or "").lower()
        path = parsed.path.strip("/")
        if host == "youtu.be":
            vid = path.split("/")[0] if path else ""
            if re.fullmatch(r"[\w\-]{11}", vid):
                return vid
            raise ValueError("Invalid YouTube short link.")
        q = parse_qs(parsed.query)
        if "v" in q and q["v"] and re.fullmatch(r"[\w\-]{11}", q["v"][0]):
            return q["v"][0]
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] == "live" and re.fullmatch(r"[\w\-]{11}", parts[1]):
            return parts[1]
        if parts and (parts[0].startswith("@") or parts[0] in ("channel", "c", "user")):
            return "/".join(parts[:2]) if parts[0] in ("channel", "c", "user") and len(parts) >= 2 else parts[0]
        raise ValueError("Invalid YouTube Live URL. Use a watch link, youtu.be link, or @channel/live link.")

    def normalize_url(self, url: str) -> str:
        clean = self.validate_url(url)
        m = re.match(r"^(https?://(?:www\.)?youtube\.com/(?:@[^/?#]+|channel/[^/?#]+|c/[^/?#]+))/?(?:[?#].*)?$", clean)
        if m:
            return m.group(1) + "/live"
        return clean

    def resolve_stream(self, url: str) -> StreamInfo:
        from .. import live
        norm = self.normalize_url(url)
        ch = self.extract_channel(norm)
        info = live.resolve(norm)
        return StreamInfo(
            platform=self.name,
            url=norm,
            channel=ch,
            title=info.get("title") or f"{ch} live",
            uploader=info.get("uploader") or ch,
            stream_urls=info["urls"],
            is_live=True,
            chat_meta={"url": norm},
        )

    @staticmethod
    def parse_innertube_actions(actions: list, stream_sec: float = 0.0) -> List[ChatMessage]:
        """Extract ChatMessages from YouTube liveChatContinuation.actions."""
        out: List[ChatMessage] = []
        for act in actions or []:
            if not isinstance(act, dict):
                continue
            item = (act.get("addChatItemAction") or {}).get("item") or {}
            renderer = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
            if not isinstance(renderer, dict):
                continue
            author_obj = renderer.get("authorName") or {}
            username = sanitize_username(author_obj.get("simpleText") or "viewer")
            user_id = sanitize_username(renderer.get("authorExternalChannelId") or username)
            runs = (renderer.get("message") or {}).get("runs") or []
            text_parts = []
            emotes = []
            for r in runs:
                if not isinstance(r, dict):
                    continue
                if "text" in r:
                    text_parts.append(str(r["text"]))
                elif "emoji" in r:
                    em = r["emoji"]
                    shortcuts = em.get("shortcuts") or []
                    label = shortcuts[0] if shortcuts else (em.get("emojiId") or "")
                    if label:
                        emotes.append(str(label))
                        text_parts.append(str(label))
            text = sanitize_chat_text(" ".join(text_parts))
            if not text:
                continue
            out.append(ChatMessage(
                platform="youtube",
                user_id=user_id,
                username=username,
                text=text,
                timestamp=time.time(),
                stream_sec=round(max(0.0, float(stream_sec)), 2),
                emotes=emotes,
            ))
        return out

    def _fetch_initial_continuation(self, url: str) -> tuple[str, str]:
        req = urllib.request.Request(self.normalize_url(url), headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"})
        with urllib.request.urlopen(req, timeout=12) as r:
            html = r.read(4_000_000).decode("utf-8", errors="replace")
        key_m = re.search(r'"INNERTUBE_API_KEY"\s*:\s*"([^"]+)"', html)
        cont_m = re.search(r'"liveChatRenderer"\s*:\s*\{.*?"continuation"\s*:\s*"([^"]+)"', html)
        if not cont_m:
            cont_m = re.search(r'"reloadContinuationData"\s*:\s*\{\s*"continuation"\s*:\s*"([^"]+)"', html)
        return (key_m.group(1) if key_m else ""), (cont_m.group(1) if cont_m else "")

    def connect_chat(self, url: str, on_message: Callable[[ChatMessage], None],
                     stop_event: threading.Event, stream_clock_fn: Callable[[], float]):
        self._stop_event = stop_event
        backoff = 3.0
        self.set_status("connecting")
        seen_ids = set()
        while not stop_event.is_set():
            try:
                api_key, continuation = self._fetch_initial_continuation(url)
                if not continuation:
                    raise RuntimeError("YouTube live chat continuation token not found (chat may be disabled).")
                self.set_status("connected")
                backoff = 3.0
                endpoint = "https://www.youtube.com/youtubei/v1/live_chat/get_live_chat"
                if api_key:
                    endpoint += f"?key={api_key}"
                while not stop_event.is_set() and continuation:
                    body = json.dumps({
                        "context": {
                            "client": {
                                "clientName": "WEB",
                                "clientVersion": "2.20240726.00.00",
                                "hl": "en",
                                "gl": "US",
                            }
                        },
                        "continuation": continuation,
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        endpoint,
                        data=body,
                        headers={"User-Agent": _UA, "Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=12) as r:
                        data = json.loads(r.read(4_000_000))
                    cont_contents = (
                        (data.get("continuationContents") or {}).get("liveChatContinuation") or {}
                    )
                    actions = cont_contents.get("actions") or []
                    msgs = self.parse_innertube_actions(actions, stream_clock_fn())
                    for msg in msgs:
                        sig = f"{msg.user_id}:{msg.text}:{int(msg.timestamp // 5)}"
                        if sig in seen_ids:
                            continue
                        seen_ids.add(sig)
                        if len(seen_ids) > 2000:
                            seen_ids.clear()
                        self.record_message()
                        on_message(msg)
                    # Extract next continuation token
                    next_cont = ""
                    for c_item in cont_contents.get("continuations") or []:
                        for k in ("invalidationContinuationData", "timedContinuationData", "reloadContinuationData"):
                            if k in c_item and c_item[k].get("continuation"):
                                next_cont = c_item[k]["continuation"]
                                break
                        if next_cont:
                            break
                    continuation = next_cont
                    stop_event.wait(2.5)
            except Exception as e:  # noqa: BLE001
                if stop_event.is_set():
                    break
                with self.lock:
                    self.reconnects += 1
                self.set_status("reconnecting", str(e))
                stop_event.wait(backoff)
                backoff = min(30.0, backoff * 1.8)
        self.set_status("disconnected")

