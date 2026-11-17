"""Phase 4: Local Live Chat Intelligence Engine.

Continuously analyses incoming live chat messages from Twitch, Kick, and YouTube Live with zero external LLMs:
1. Per-user duplicate & spam filtering (prevents a single spammer from faking audience excitement).
2. Messages per second (MPS) vs. rolling channel baseline (ratio & z-score).
3. Unique chat participants in the active window vs. baseline.
4. Repeated messages/phrases across multiple distinct users.
5. Emote spam & reaction bursts (Unicode emojis + Twitch/Kick/7TV emote tokens).
6. Excitement keywords (OMG, NO WAY, INSANE, LMAO, LET'S GO, CLIP IT, etc.).
7. Rapid acceleration in chat velocity and configurable cooldowns.
"""
import collections
import math
import re
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

from .platforms.base import ChatMessage

EXCITEMENT_KEYWORDS = {
    "omg", "omfg", "noway", "no way", "aint no way", "ain't no way",
    "insane", "lmao", "lmfao", "rofl", "lets go", "let's go", "letsgo",
    "clip it", "clip that", "clip this", "clipped", "wtf", "wth",
    "holy", "crazy", "unbelievable", "clutch", "godlike", "goated",
    "goat", "sheesh", "wild", "unreal", "impossible", "bro what",
    "what just happened", "haha", "hahaha", "dead", "crying", "screaming",
}

EMOTE_TOKENS = {
    "kekw", "omegalul", "lul", "lulw", "pog", "pogchamp", "poggers",
    "pagman", "icant", "caught", "cinema", "ez", "clap", "monkas",
    "monkaw", "pepega", "sadge", "copium", "gigachad", "based",
    "catjam", "kappa", "notlikethis", "biblethump", "kreygasm",
    "w", "ww", "www", "dub", "massive w",
}

EMOJI_RE = re.compile(
    r"[😂🤣💀🔥😭😱🤯👏🐐⚡👀💯❤️🙏🎉💥🫡☠️🥶🥵🤡👑]"
)


def _norm_msg(text: str) -> str:
    s = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", s).strip()


def _extract_reactions(msg: ChatMessage) -> Tuple[List[str], List[str]]:
    """Return (emotes_found, excitement_keywords_found) for a message."""
    emotes = list(msg.emotes or [])
    for m in EMOJI_RE.findall(msg.text):
        emotes.append(m)
    norm = _norm_msg(msg.text)
    words = norm.split()
    for w in words:
        # Collapse stretched letters like "omgggg" -> "omg", "wwww" -> "w"
        collapsed = re.sub(r"(.)\1{2,}", r"\1", w)
        if w in EMOTE_TOKENS or collapsed in EMOTE_TOKENS:
            emotes.append(w.upper())

    excite = []
    for kw in EXCITEMENT_KEYWORDS:
        if " " in kw:
            if kw in norm:
                excite.append(kw.upper())
        else:
            for w in words:
                collapsed = re.sub(r"(.)\1{2,}", r"\1", w)
                if w == kw or collapsed == kw:
                    excite.append(kw.upper())
                    break
    return emotes, excite


class ChatIntelligenceEngine:
    """Real-time local live chat analyzer with anti-spam deduplication and baseline tracking."""

    def __init__(
        self,
        active_window_sec: float = 8.0,
        baseline_window_sec: float = 180.0,
        max_msgs_per_user_window: int = 2,
        dup_window_sec: float = 15.0,
        spike_multiplier: float = 3.0,
        min_unique_chatters: int = 3,
        cooldown_sec: float = 45.0,
    ):
        self.lock = threading.RLock()
        self.active_window_sec = max(3.0, float(active_window_sec))
        self.baseline_window_sec = max(30.0, float(baseline_window_sec))
        self.max_msgs_per_user_window = max(1, int(max_msgs_per_user_window))
        self.dup_window_sec = max(2.0, float(dup_window_sec))
        self.spike_multiplier = max(1.3, float(spike_multiplier))
        self.min_unique_chatters = max(1, int(min_unique_chatters))
        self.cooldown_sec = max(5.0, float(cooldown_sec))

        self.events: collections.deque = collections.deque()       # accepted messages (stream_sec, user_id, username, text, emotes, excite)
        self.raw_total: int = 0
        self.accepted_total: int = 0
        self.spam_filtered: int = 0
        self._user_recent: Dict[str, List[Tuple[float, str]]] = {}  # user_id -> [(stream_sec, norm_text)]
        self.per_sec_counts: collections.Counter = collections.Counter()
        self.last_spike_sec: float = -9999.0
        self.recent_messages: collections.deque = collections.deque(maxlen=40)

    def configure(self, **kwargs):
        with self.lock:
            if "spike_multiplier" in kwargs and kwargs["spike_multiplier"] is not None:
                self.spike_multiplier = max(1.2, float(kwargs["spike_multiplier"]))
            if "min_unique_chatters" in kwargs and kwargs["min_unique_chatters"] is not None:
                self.min_unique_chatters = max(1, int(kwargs["min_unique_chatters"]))
            if "cooldown_sec" in kwargs and kwargs["cooldown_sec"] is not None:
                self.cooldown_sec = max(5.0, float(kwargs["cooldown_sec"]))

    def ingest(self, msg: ChatMessage) -> bool:
        """Process one incoming ChatMessage. Returns True if accepted (genuine), False if filtered as spam."""
        with self.lock:
            self.raw_total += 1
            t = float(msg.stream_sec)
            uid = msg.user_id or msg.username or "anon"
            norm = _norm_msg(msg.text)

            # Prune user history older than max(dup_window_sec, active_window_sec)
            horizon = max(self.dup_window_sec, self.active_window_sec)
            hist = [item for item in self._user_recent.get(uid, []) if t - item[0] <= horizon]

            # 1. Duplicate message check from the same user within dup_window_sec
            if any(t - prev_t <= self.dup_window_sec and prev_norm == norm for prev_t, prev_norm in hist):
                self.spam_filtered += 1
                self._user_recent[uid] = hist
                return False

            # 2. Rate-limit messages per single user within active_window_sec
            in_active = sum(1 for prev_t, _ in hist if t - prev_t <= self.active_window_sec)
            if in_active >= self.max_msgs_per_user_window:
                self.spam_filtered += 1
                self._user_recent[uid] = hist
                return False

            hist.append((t, norm))
            self._user_recent[uid] = hist

            emotes, excite = _extract_reactions(msg)
            self.events.append({
                "s": t,
                "uid": uid,
                "user": msg.username,
                "text": msg.text,
                "norm": norm,
                "emotes": emotes,
                "excite": excite,
                "platform": msg.platform,
            })
            self.accepted_total += 1
            self.per_sec_counts[int(t)] += 1
            self.recent_messages.append({
                "s": round(t, 1),
                "user": msg.username,
                "text": msg.text[:140],
                "platform": msg.platform,
            })
            self._prune(t)
            return True

    def _prune(self, now_sec: float):
        cutoff = now_sec - (self.baseline_window_sec + self.active_window_sec + 30.0)
        while self.events and self.events[0]["s"] < cutoff:
            self.events.popleft()
        if len(self._user_recent) > 5000:
            horizon = max(self.dup_window_sec, self.active_window_sec)
            self._user_recent = {
                u: [x for x in items if now_sec - x[0] <= horizon]
                for u, items in self._user_recent.items()
                if any(now_sec - x[0] <= horizon for x in items)
            }

    def analyze(self, now_sec: float) -> dict:
        """Compute full chat intelligence snapshot at stream time `now_sec`."""
        with self.lock:
            self._prune(now_sec)
            w_sec = self.active_window_sec
            b_start = now_sec - w_sec - self.baseline_window_sec
            a_start = now_sec - w_sec

            active_evs = [e for e in self.events if a_start < e["s"] <= now_sec]
            base_evs = [e for e in self.events if b_start < e["s"] <= a_start]

            # 1. Messages per second (MPS) & baseline z-score
            current_mps = len(active_evs) / w_sec
            if base_evs:
                span_b = max(10.0, a_start - max(b_start, base_evs[0]["s"] - 1.0))
                baseline_mps = len(base_evs) / span_b
                # Compute per-window variance over baseline slices
                n_slices = max(1, int(span_b // w_sec))
                slice_rates = []
                for idx in range(n_slices):
                    sl_a = a_start - (idx + 1) * w_sec
                    sl_b = a_start - idx * w_sec
                    cnt = sum(1 for e in base_evs if sl_a < e["s"] <= sl_b)
                    slice_rates.append(cnt / w_sec)
                mean_r = sum(slice_rates) / len(slice_rates)
                var_r = sum((r - mean_r) ** 2 for r in slice_rates) / max(1, len(slice_rates))
                std_mps = math.sqrt(var_r)
            else:
                baseline_mps = max(0.5, current_mps * 0.35) if current_mps > 0 else 0.5
                std_mps = max(0.3, baseline_mps * 0.4)

            velocity_ratio = current_mps / max(baseline_mps, 0.25)
            z_score = (current_mps - baseline_mps) / max(std_mps, 0.25)

            # 2. Rapid acceleration (second half of active window vs first half)
            half = w_sec / 2.0
            first_half = sum(1 for e in active_evs if e["s"] <= now_sec - half) / half
            second_half = sum(1 for e in active_evs if e["s"] > now_sec - half) / half
            acceleration = (second_half - first_half) / max(baseline_mps, 0.5)

            # 3. Unique chat participants
            unique_users: Set[str] = {e["uid"] for e in active_evs}
            unique_count = len(unique_users)
            base_unique_per_win = 1.0
            if base_evs:
                span_b = max(w_sec, a_start - max(b_start, base_evs[0]["s"]))
                base_users = len({e["uid"] for e in base_evs})
                base_unique_per_win = max(1.0, base_users * (w_sec / span_b))
            unique_ratio = unique_count / max(1.0, base_unique_per_win)

            # 4. Emote bursts, excitement keywords, and repeated multi-user phrases
            emote_counter = collections.Counter()
            excite_counter = collections.Counter()
            phrase_users: Dict[str, Set[str]] = collections.defaultdict(set)
            reaction_msgs = 0

            for e in active_evs:
                if e["emotes"] or e["excite"]:
                    reaction_msgs += 1
                for em in e["emotes"]:
                    emote_counter[em] += 1
                for ex in e["excite"]:
                    excite_counter[ex] += 1
                if e["norm"] and len(e["norm"]) >= 2:
                    phrase_users[e["norm"]].add(e["uid"])

            repeated_phrases = [
                (phrase, len(users))
                for phrase, users in phrase_users.items()
                if len(users) >= 2
            ]
            repeated_phrases.sort(key=lambda x: -x[1])

            emote_rate = sum(emote_counter.values()) / w_sec
            excitement_rate = sum(excite_counter.values()) / w_sec
            reaction_density = reaction_msgs / max(1, len(active_evs))

            # 5. Normalized sub-scores (0-100) for Phase 6 Viral Scoring Engine
            # Velocity score (0-100)
            vel_score = min(100.0, max(0.0, (velocity_ratio - 1.0) / max(0.5, self.spike_multiplier - 0.5) * 70.0 + max(0.0, acceleration * 10.0)))
            if current_mps < 0.4 or unique_count < 2:
                vel_score *= 0.25

            # Unique chatters score (0-100)
            uniq_score = min(100.0, (unique_count / max(1, self.min_unique_chatters)) * 45.0 + max(0.0, (unique_ratio - 1.0) * 20.0))
            if unique_count < self.min_unique_chatters:
                uniq_score = min(uniq_score, 45.0)

            # Emote & reaction burst score (0-100)
            total_reactions = sum(emote_counter.values()) + sum(excite_counter.values()) * 1.5
            multi_user_repeats = sum(cnt for _, cnt in repeated_phrases)
            burst_score = min(100.0, (total_reactions * 12.0 + multi_user_repeats * 15.0) * (0.5 + 0.5 * reaction_density))
            if unique_count < 2:
                burst_score *= 0.2

            # Determine if this constitutes a chat viral spike
            in_cooldown = (now_sec - self.last_spike_sec) < self.cooldown_sec
            is_spike = (
                not in_cooldown
                and unique_count >= self.min_unique_chatters
                and (
                    velocity_ratio >= self.spike_multiplier
                    or (velocity_ratio >= 1.8 and (burst_score >= 65.0 or len(repeated_phrases) >= 1))
                )
            )

            reasons = []
            if velocity_ratio >= 1.8 and current_mps >= 0.5:
                reasons.append(f"Chat velocity {current_mps:.1f} msg/s ({velocity_ratio:.1f}x baseline)")
            if unique_count >= self.min_unique_chatters:
                reasons.append(f"{unique_count} unique chatters active")
            top_excite = [k for k, _ in excite_counter.most_common(3)]
            top_emotes = [k for k, _ in emote_counter.most_common(3)]
            if top_excite:
                reasons.append(f"Excitement burst: {', '.join(top_excite)}")
            if top_emotes:
                reasons.append(f"Emote burst: {', '.join(top_emotes)}")
            if repeated_phrases:
                top_p, top_u = repeated_phrases[0]
                reasons.append(f"{top_u} viewers spammed '{top_p[:24].upper()}'")

            if is_spike:
                self.last_spike_sec = now_sec

            return {
                "timestamp": round(now_sec, 2),
                "mps": round(current_mps, 2),
                "baseline_mps": round(baseline_mps, 2),
                "velocity_ratio": round(velocity_ratio, 2),
                "z_score": round(z_score, 2),
                "acceleration": round(acceleration, 2),
                "unique_chatters": unique_count,
                "unique_ratio": round(unique_ratio, 2),
                "emote_rate": round(emote_rate, 2),
                "excitement_rate": round(excitement_rate, 2),
                "reaction_density": round(reaction_density, 2),
                "top_emotes": top_emotes,
                "top_excitement": top_excite,
                "repeated_phrases": [{"phrase": p, "users": u} for p, u in repeated_phrases[:3]],
                "velocity_score": round(vel_score, 1),
                "unique_score": round(uniq_score, 1),
                "burst_score": round(burst_score, 1),
                "is_spike": is_spike,
                "in_cooldown": in_cooldown,
                "reasons": reasons,
                "total_messages": self.accepted_total,
                "spam_filtered": self.spam_filtered,
                "recent_messages": list(self.recent_messages)[-12:],
            }

