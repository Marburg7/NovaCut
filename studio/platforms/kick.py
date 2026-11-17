"""Kick Live Stream & Pusher WebSocket Live Chat Adapter (Zero API Keys Required)."""
import base64
import json
import os
import re
import socket
import ssl
import struct
import threading
import time
import urllib.request
from typing import Callable
from urllib.parse import urlparse

from .base import ChatMessage, PlatformAdapter, StreamInfo, sanitize_chat_text, sanitize_username

_PUSHER_HOST = "ws-us2.pusher.com"
_PUSHER_KEY = "32cbd69e4b950bf97679"
_PUSHER_PATH = f"/app/{_PUSHER_KEY}?protocol=7&client=js&version=8.4.0-rc2&flash=false"
_KICK_EMOTE_RE = re.compile(r"\[emote:(\d+):([^\]]+)\]")
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _ws_send_text(sock, payload: str):
    """Send a masked RFC 6455 text frame over a client TLS socket."""
    data = payload.encode("utf-8")
    mask = os.urandom(4)
    header = bytearray([0x81])  # FIN + text opcode
    n = len(data)
    if n < 126:
        header.append(0x80 | n)
    elif n < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", n))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", n))
    header.extend(mask)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    sock.sendall(bytes(header) + masked)


def _ws_recv_frame(sock) -> tuple[int, bytes]:
    """Read one RFC 6455 WebSocket frame. Returns (opcode, payload_bytes)."""
    def read_exact(k: int) -> bytes:
        buf = bytearray()
        while len(buf) < k:
            chunk = sock.recv(k - len(buf))
            if not chunk:
                raise ConnectionError("WebSocket closed")
            buf.extend(chunk)
        return bytes(buf)

    b1, b2 = read_exact(2)
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", read_exact(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", read_exact(8))[0]
    if length > 2_000_000:
        raise ValueError("WebSocket frame too large")
    mask_key = read_exact(4) if masked else b""
    payload = read_exact(length)
    if masked:
        payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return opcode, payload


class KickAdapter(PlatformAdapter):
    name = "kick"
    domains = ("kick.com",)

    def extract_channel(self, url: str) -> str:
        clean = self.validate_url(url)
        parts = [p for p in urlparse(clean).path.strip("/").split("/") if p]
        if not parts or not re.fullmatch(r"[A-Za-z0-9_\-]{2,40}", parts[0]):
            raise ValueError("Invalid Kick channel URL. Expected https://kick.com/<channel>")
        return parts[0].lower()

    def normalize_url(self, url: str) -> str:
        slug = self.extract_channel(url)
        return f"https://kick.com/{slug}"

    def fetch_channel_meta(self, slug: str) -> dict:
        req = urllib.request.Request(
            f"https://kick.com/api/v2/channels/{slug}",
            headers={"User-Agent": _UA, "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read(2_000_000))

    def resolve_stream(self, url: str) -> StreamInfo:
        from .. import live
        norm = self.normalize_url(url)
        slug = self.extract_channel(norm)
        info = live.resolve(norm)
        chatroom_id = None
        try:
            meta = self.fetch_channel_meta(slug)
            chatroom_id = (meta.get("chatroom") or {}).get("id")
        except Exception:
            pass
        return StreamInfo(
            platform=self.name,
            url=norm,
            channel=slug,
            title=info.get("title") or f"{slug} live",
            uploader=info.get("uploader") or slug,
            stream_urls=info["urls"],
            is_live=True,
            chat_meta={"slug": slug, "chatroom_id": chatroom_id},
        )

    @staticmethod
    def parse_pusher_event(raw_json: str, stream_sec: float = 0.0) -> ChatMessage | None:
        """Parse a Pusher frame JSON string from Kick chatrooms.<id>.v2 into a ChatMessage."""
        try:
            outer = json.loads(raw_json)
        except ValueError:
            return None
        ev = str(outer.get("event") or "")
        if "ChatMessageEvent" not in ev:
            return None
        data = outer.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                return None
        if not isinstance(data, dict):
            return None
        content = str(data.get("content") or "")
        emotes = [m.group(2) for m in _KICK_EMOTE_RE.finditer(content)]
        clean_content = _KICK_EMOTE_RE.sub(r" \2 ", content)
        text = sanitize_chat_text(clean_content)
        if not text:
            return None
        sender = data.get("sender") or {}
        uid = sanitize_username(str(sender.get("id") or sender.get("slug") or sender.get("username") or "viewer"))
        uname = sanitize_username(str(sender.get("username") or sender.get("slug") or "viewer"))
        return ChatMessage(
            platform="kick",
            user_id=uid,
            username=uname,
            text=text,
            timestamp=time.time(),
            stream_sec=round(max(0.0, float(stream_sec)), 2),
            emotes=emotes,
        )

    def connect_chat(self, url: str, on_message: Callable[[ChatMessage], None],
                     stop_event: threading.Event, stream_clock_fn: Callable[[], float],
                     chatroom_id: int | None = None):
        self._stop_event = stop_event
        slug = self.extract_channel(url)
        backoff = 2.0
        self.set_status("connecting")
        while not stop_event.is_set():
            sock = None
            try:
                if not chatroom_id:
                    meta = self.fetch_channel_meta(slug)
                    chatroom_id = (meta.get("chatroom") or {}).get("id")
                    if not chatroom_id:
                        raise RuntimeError(f"Could not find Kick chatroom ID for {slug}")
                raw_sock = socket.create_connection((_PUSHER_HOST, 443), timeout=15)
                ctx = ssl.create_default_context()
                sock = ctx.wrap_socket(raw_sock, server_hostname=_PUSHER_HOST)
                ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
                req = (
                    f"GET {_PUSHER_PATH} HTTP/1.1\r\n"
                    f"Host: {_PUSHER_HOST}\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {ws_key}\r\n"
                    "Sec-WebSocket-Version: 13\r\n"
                    f"User-Agent: {_UA}\r\n\r\n"
                )
                sock.sendall(req.encode("utf-8"))
                resp = b""
                while b"\r\n\r\n" not in resp:
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("Kick Pusher handshake failed")
                    resp += chunk
                if b" 101 " not in resp.split(b"\r\n", 1)[0]:
                    raise ConnectionError("Kick Pusher rejected WebSocket upgrade")
                sub = json.dumps({
                    "event": "pusher:subscribe",
                    "data": {"auth": "", "channel": f"chatrooms.{chatroom_id}.v2"},
                })
                _ws_send_text(sock, sub)
                sock.settimeout(25.0)
                self.set_status("connected")
                backoff = 2.0
                while not stop_event.is_set():
                    try:
                        opcode, payload = _ws_recv_frame(sock)
                    except socket.timeout:
                        _ws_send_text(sock, json.dumps({"event": "pusher:ping", "data": {}}))
                        continue
                    if opcode == 0x8:  # close
                        break
                    if opcode == 0x9:  # ping
                        continue
                    if opcode == 0x1:
                        txt = payload.decode("utf-8", errors="replace")
                        if '"pusher:ping"' in txt:
                            _ws_send_text(sock, json.dumps({"event": "pusher:pong", "data": {}}))
                            continue
                        msg = self.parse_pusher_event(txt, stream_clock_fn())
                        if msg is not None:
                            self.record_message()
                            on_message(msg)
            except Exception as e:  # noqa: BLE001
                if stop_event.is_set():
                    break
                with self.lock:
                    self.reconnects += 1
                self.set_status("reconnecting", str(e))
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
            if not stop_event.is_set():
                stop_event.wait(backoff)
                backoff = min(30.0, backoff * 1.8)
        self.set_status("disconnected")

