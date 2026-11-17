"""Twitch Live Stream & IRCv3 Live Chat Adapter (Zero API Keys Required)."""
import re
import socket
import ssl
import threading
import time
from typing import Callable
from urllib.parse import urlparse

from .base import ChatMessage, PlatformAdapter, StreamInfo, sanitize_chat_text, sanitize_username


class TwitchAdapter(PlatformAdapter):
    name = "twitch"
    domains = ("twitch.tv",)

    def extract_channel(self, url: str) -> str:
        clean = self.validate_url(url)
        parts = [p for p in urlparse(clean).path.strip("/").split("/") if p]
        if not parts or not re.fullmatch(r"[A-Za-z0-9_]{2,40}", parts[0]):
            raise ValueError("Invalid Twitch channel URL. Expected https://twitch.tv/<channel>")
        return parts[0].lower()

    def normalize_url(self, url: str) -> str:
        ch = self.extract_channel(url)
        return f"https://www.twitch.tv/{ch}"

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
            chat_meta={"channel": ch},
        )

    @staticmethod
    def parse_irc_line(line: str, stream_sec: float = 0.0) -> ChatMessage | None:
        """Parse a Twitch IRCv3 PRIVMSG line (with optional @tags) into a ChatMessage."""
        raw = line.strip("\r\n")
        if not raw or " PRIVMSG #" not in raw:
            return None
        tags = {}
        rest = raw
        if rest.startswith("@"):
            parts = rest.split(" ", 1)
            if len(parts) < 2:
                return None
            tag_str, rest = parts[0][1:], parts[1]
            for kv in tag_str.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    tags[k] = v
        # Format: :<nick>!<user>@<host> PRIVMSG #<channel> :<message>
        m = re.match(r"^:([^!]+)!.*?\s+PRIVMSG\s+#[^\s]+\s+:(.*)$", rest)
        if not m:
            return None
        nick, msg_body = m.group(1), m.group(2)
        if msg_body.startswith("\x01ACTION ") and msg_body.endswith("\x01"):
            msg_body = msg_body[8:-1]
        text = sanitize_chat_text(msg_body)
        if not text:
            return None
        user_id = sanitize_username(tags.get("user-id") or nick.lower())
        username = sanitize_username(tags.get("display-name") or nick)
        emotes = []
        emote_tag = tags.get("emotes") or ""
        if emote_tag:
            # Twitch emote format: emote_id:start-end,start-end/emote_id2:start-end
            for grp in emote_tag.split("/"):
                if ":" in grp:
                    eid, spans = grp.split(":", 1)
                    for sp in spans.split(","):
                        if "-" in sp:
                            emotes.append(f"twitch:{eid}")
        return ChatMessage(
            platform="twitch",
            user_id=user_id,
            username=username,
            text=text,
            timestamp=time.time(),
            stream_sec=round(max(0.0, float(stream_sec)), 2),
            emotes=emotes,
        )

    def connect_chat(self, url: str, on_message: Callable[[ChatMessage], None],
                     stop_event: threading.Event, stream_clock_fn: Callable[[], float]):
        self._stop_event = stop_event
        channel = self.extract_channel(url)
        backoff = 2.0
        self.set_status("connecting")
        while not stop_event.is_set():
            sock = None
            try:
                raw_sock = socket.create_connection(("irc.chat.twitch.tv", 6697), timeout=15)
                ctx = ssl.create_default_context()
                sock = ctx.wrap_socket(raw_sock, server_hostname="irc.chat.twitch.tv")
                nick = f"justinfan{int(time.time()) % 89999 + 10000}"
                handshake = (
                    "CAP REQ :twitch.tv/tags twitch.tv/commands\r\n"
                    f"NICK {nick}\r\n"
                    f"JOIN #{channel}\r\n"
                )
                sock.sendall(handshake.encode("utf-8"))
                sock.settimeout(20.0)
                self.set_status("connected")
                backoff = 2.0
                buf = b""
                while not stop_event.is_set():
                    try:
                        chunk = sock.recv(8192)
                    except socket.timeout:
                        sock.sendall(b"PING :keepalive\r\n")
                        continue
                    if not chunk:
                        break
                    *lines, buf = (buf + chunk).split(b"\r\n")
                    for raw_line in lines:
                        decoded = raw_line.decode("utf-8", errors="replace")
                        if decoded.startswith("PING"):
                            sock.sendall(b"PONG :tmi.twitch.tv\r\n")
                            continue
                        msg = self.parse_irc_line(decoded, stream_clock_fn())
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

