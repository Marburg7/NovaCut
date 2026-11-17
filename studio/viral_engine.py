"""Phases 5 & 6: Audio/Transcript Intelligence & Configurable Viral Moment Scoring Engine.

Combines live chat intelligence, audio dynamics (volume spikes, silence-to-loud transitions), and
speech-to-text transcript semantics (keywords, hooks, conversational pacing) into a unified 0-100
Viral Score with configurable weights:
  - Chat velocity:            30%
  - Unique chat participants: 15%
  - Emote & reaction bursts:  15%
  - Audio excitement:         20%
  - Transcript/context:       10%
  - Cross-signal agreement:   10%
"""
import math
import time
from typing import Dict, List, Optional

import numpy as np

from . import brain

DEFAULT_WEIGHTS = {
    "chat_velocity": 0.30,
    "unique_chatters": 0.15,
    "emote_bursts": 0.15,
    "audio_excitement": 0.20,
    "transcript_signals": 0.10,
    "cross_signal": 0.10,
}

SENSITIVITY_PRESETS = {
    "low": {"min_viral_score": 78, "chat_spike_mult": 4.0, "min_unique_chatters": 5},
    "medium": {"min_viral_score": 65, "chat_spike_mult": 3.0, "min_unique_chatters": 3},
    "high": {"min_viral_score": 52, "chat_spike_mult": 2.2, "min_unique_chatters": 2},
}


def analyze_audio_window(db_series: List[float], now_sec: float, window_sec: float = 8.0, baseline_sec: float = 180.0) -> dict:
    """Phase 5: Analyse audio volume spikes, sudden loud reactions, and silence-to-loud transitions."""
    if not db_series or len(db_series) < 5:
        return {
            "current_db": -60.0,
            "baseline_db": -60.0,
            "peak_db": -60.0,
            "z_score": 0.0,
            "jump_pct": 0.0,
            "silence_to_loud": False,
            "sustained_loud": False,
            "audio_score": 0.0,
            "reasons": [],
        }

    n = len(db_series)
    w = max(3, int(round(window_sec)))
    b = max(15, int(round(baseline_sec)))
    arr = np.array(db_series[max(0, n - b - w):], dtype=np.float32)

    active = arr[-min(w, len(arr)):]
    base = arr[:-len(active)] if len(arr) > len(active) + 4 else arr

    # Convert dB to linear amplitude for percentage jump
    amp_base = float((10.0 ** (base / 20.0)).mean())
    amp_cur = float((10.0 ** (active / 20.0)).mean())
    amp_peak = float((10.0 ** (active / 20.0)).max())
    jump_pct = (amp_cur / max(amp_base, 1e-5) - 1.0) * 100.0
    peak_jump_pct = (amp_peak / max(amp_base, 1e-5) - 1.0) * 100.0

    med_db = float(np.median(base))
    std_db = float(np.std(base)) + 1.5
    cur_db = float(np.mean(active))
    peak_db = float(np.max(active))
    z_score = (peak_db - med_db) / std_db

    # Detect silence-to-loud transition (quiet dip followed by sharp spike)
    silence_to_loud = False
    if len(arr) >= w + 4:
        pre_window = arr[-(w + 4):-w]
        if float(np.min(pre_window)) <= med_db - 5.0 and peak_db >= med_db + 5.5:
            silence_to_loud = True

    sustained_loud = bool(sum(1 for x in active if x >= med_db + 4.0) >= max(2, w // 2))

    # Compute 0-100 audio excitement score
    score = 0.0
    if z_score > 0:
        score += min(65.0, z_score * 22.0)
    if jump_pct > 20:
        score += min(30.0, (jump_pct - 20.0) * 0.28)
    elif peak_jump_pct > 40:
        score += min(20.0, (peak_jump_pct - 40.0) * 0.15)
    if silence_to_loud:
        score += 15.0
    if sustained_loud:
        score += 10.0
    audio_score = min(100.0, max(0.0, score))

    reasons = []
    if jump_pct >= 50 or z_score >= 1.8:
        reasons.append(f"Audio spike +{max(0, round(jump_pct))}% ({peak_db:.1f} dB, z={z_score:+.1f})")
    if silence_to_loud:
        reasons.append("Sudden silence-to-loud reaction burst")
    elif sustained_loud and audio_score >= 55:
        reasons.append("Sustained high vocal energy")

    return {
        "current_db": round(cur_db, 1),
        "baseline_db": round(med_db, 1),
        "peak_db": round(peak_db, 1),
        "z_score": round(z_score, 2),
        "jump_pct": round(jump_pct, 1),
        "silence_to_loud": silence_to_loud,
        "sustained_loud": sustained_loud,
        "audio_score": round(audio_score, 1),
        "reasons": reasons,
    }


def analyze_transcript_window(words: List[dict], now_sec: float, window_sec: float = 20.0) -> dict:
    """Phase 5: Analyse speech recognition words, important keywords, and conversational pacing."""
    start_sec = max(0.0, now_sec - window_sec)
    win_words = [w for w in (words or []) if start_sec <= float(w.get("s", 0)) <= now_sec]
    if not win_words:
        return {
            "word_count": 0,
            "wps": 0.0,
            "category": "other",
            "keywords": [],
            "transcript_score": 0.0,
            "text": "",
            "reasons": [],
        }

    text = " ".join(str(w.get("w", "")) for w in win_words).strip()
    span = max(3.0, min(window_sec, now_sec - float(win_words[0].get("s", start_sec)) + 0.5))
    wps = len(win_words) / span
    cat, clean_words = brain._analyze_text_semantics(text)

    matched_kws = []
    for w in clean_words:
        for kws in brain.CATEGORY_KEYWORDS.values():
            if w in kws and w not in matched_kws:
                matched_kws.append(w)

    hook_triggers = [
        w for w in clean_words
        if w in ("wait", "look", "watch", "insane", "crazy", "omg", "never", "clutch", "clip", "what", "unbelievable")
    ]
    exclamations = text.count("!") + text.count("?")

    score = 20.0
    score += min(40.0, len(matched_kws) * 14.0)
    score += min(25.0, len(hook_triggers) * 9.0)
    score += min(10.0, exclamations * 4.0)
    if 2.0 <= wps <= 4.5:
        score += 10.0
    elif wps > 4.5:
        score += 12.0  # fast excited talking
    transcript_score = min(100.0, max(0.0, score))

    reasons = []
    if matched_kws:
        reasons.append(f"Speech keywords ({cat}): {', '.join(matched_kws[:4])}")
    if wps >= 3.5 and len(win_words) >= 12:
        reasons.append(f"Fast excited speech pacing ({wps:.1f} words/s)")

    return {
        "word_count": len(win_words),
        "wps": round(wps, 2),
        "category": cat,
        "keywords": matched_kws[:6],
        "transcript_score": round(transcript_score, 1),
        "text": text[:300],
        "reasons": reasons,
    }


class ViralScoringEngine:
    """Phase 6: Configurable multi-signal Viral Moment Scoring Engine."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_viral_score: float = 65.0,
        pre_roll_sec: float = 30.0,
        post_roll_sec: float = 20.0,
        cooldown_sec: float = 45.0,
        sensitivity: str = "medium",
    ):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.set_weights(weights)
        self.sensitivity = sensitivity if sensitivity in SENSITIVITY_PRESETS else "medium"
        self.min_viral_score = float(min_viral_score)
        self.pre_roll_sec = max(5.0, float(pre_roll_sec))
        self.post_roll_sec = max(5.0, float(post_roll_sec))
        self.cooldown_sec = max(5.0, float(cooldown_sec))
        self.last_triggered_sec: float = -9999.0
        self.history: List[dict] = []

    def set_weights(self, new_weights: Dict[str, float]):
        for k in DEFAULT_WEIGHTS:
            if k in new_weights and new_weights[k] is not None:
                self.weights[k] = max(0.0, float(new_weights[k]))
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] = round(self.weights[k] / total, 4)

    def configure(self, **kwargs):
        if "sensitivity" in kwargs and kwargs["sensitivity"] in SENSITIVITY_PRESETS:
            self.sensitivity = kwargs["sensitivity"]
            preset = SENSITIVITY_PRESETS[self.sensitivity]
            if "min_viral_score" not in kwargs:
                self.min_viral_score = float(preset["min_viral_score"])
        if "min_viral_score" in kwargs and kwargs["min_viral_score"] is not None:
            self.min_viral_score = max(10.0, min(99.0, float(kwargs["min_viral_score"])))
        if "pre_roll_sec" in kwargs and kwargs["pre_roll_sec"] is not None:
            self.pre_roll_sec = max(5.0, min(120.0, float(kwargs["pre_roll_sec"])))
        if "post_roll_sec" in kwargs and kwargs["post_roll_sec"] is not None:
            self.post_roll_sec = max(5.0, min(120.0, float(kwargs["post_roll_sec"])))
        if "cooldown_sec" in kwargs and kwargs["cooldown_sec"] is not None:
            self.cooldown_sec = max(5.0, min(600.0, float(kwargs["cooldown_sec"])))
        if "weights" in kwargs and isinstance(kwargs["weights"], dict):
            self.set_weights(kwargs["weights"])

    def evaluate(
        self,
        now_sec: float,
        chat_snap: Optional[dict] = None,
        audio_snap: Optional[dict] = None,
        transcript_snap: Optional[dict] = None,
        chat_connected: bool = True,
    ) -> dict:
        """Combine chat, audio, and transcript intelligence into a ViralMoment evaluation."""
        chat_snap = chat_snap or {}
        audio_snap = audio_snap or {}
        transcript_snap = transcript_snap or {}

        vel_s = float(chat_snap.get("velocity_score", 0.0))
        uniq_s = float(chat_snap.get("unique_score", 0.0))
        burst_s = float(chat_snap.get("burst_score", 0.0))
        aud_s = float(audio_snap.get("audio_score", 0.0))
        txt_s = float(transcript_snap.get("transcript_score", 0.0))

        # Cross-signal agreement score (0-100): high when multiple independent modalities spike together
        chat_strong = (vel_s >= 55.0 or burst_s >= 55.0) and uniq_s >= 40.0
        audio_strong = aud_s >= 50.0
        txt_strong = txt_s >= 50.0

        if chat_strong and audio_strong and txt_strong:
            agree_s = 100.0
        elif (chat_strong and audio_strong) or (chat_strong and txt_strong):
            agree_s = 85.0
        elif audio_strong and txt_strong:
            agree_s = 75.0
        elif chat_strong or audio_strong:
            agree_s = 40.0
        else:
            agree_s = max(0.0, (vel_s + aud_s + txt_s) / 4.0)

        w = self.weights
        has_chat_data = chat_connected and (chat_snap.get("total_messages", 0) > 0 or vel_s > 0 or uniq_s > 0 or burst_s > 0)

        if has_chat_data:
            raw_score = (
                w["chat_velocity"] * vel_s
                + w["unique_chatters"] * uniq_s
                + w["emote_bursts"] * burst_s
                + w["audio_excitement"] * aud_s
                + w["transcript_signals"] * txt_s
                + w["cross_signal"] * agree_s
            )
        else:
            # Stream has no chat messages/connection: normalize across audio, transcript, and agreement
            av_total = max(0.01, w["audio_excitement"] + w["transcript_signals"] + w["cross_signal"])
            raw_score = (
                (w["audio_excitement"] / av_total) * aud_s
                + (w["transcript_signals"] / av_total) * txt_s
                + (w["cross_signal"] / av_total) * agree_s
            )

        viral_score = int(round(min(100.0, max(0.0, raw_score))))

        reasons = []
        for r in (chat_snap.get("reasons") or []) + (audio_snap.get("reasons") or []) + (transcript_snap.get("reasons") or []):
            if r not in reasons:
                reasons.append(r)
        if agree_s >= 75.0:
            reasons.append("Cross-signal agreement (Chat + Audio/Speech aligned)")

        in_cooldown = (now_sec - self.last_triggered_sec) < self.cooldown_sec
        should_clip = viral_score >= self.min_viral_score and not in_cooldown

        # Recommended clip duration based on pre-roll + post-roll + excitement intensity
        extra = 5.0 if viral_score >= 85 else 0.0
        rec_duration = round(self.pre_roll_sec + self.post_roll_sec + extra, 1)

        moment = {
            "timestamp": round(now_sec, 2),
            "wall_time": time.time(),
            "viral_score": viral_score,
            "threshold": self.min_viral_score,
            "should_clip": should_clip,
            "in_cooldown": in_cooldown,
            "reasons": reasons or ["Normal stream baseline"],
            "recommended_duration": rec_duration,
            "pre_roll_sec": self.pre_roll_sec,
            "post_roll_sec": self.post_roll_sec,
            "Clip_window": [round(max(0.0, now_sec - self.pre_roll_sec), 2), round(now_sec + self.post_roll_sec, 2)],
            "component_scores": {
                "chat_velocity": round(vel_s, 1),
                "unique_chatters": round(uniq_s, 1),
                "emote_bursts": round(burst_s, 1),
                "audio_excitement": round(aud_s, 1),
                "transcript_signals": round(txt_s, 1),
                "cross_signal": round(agree_s, 1),
            },
            "weights": dict(self.weights),
            "chat_activity": {
                "mps": chat_snap.get("mps", 0.0),
                "baseline_mps": chat_snap.get("baseline_mps", 0.0),
                "velocity_ratio": chat_snap.get("velocity_ratio", 0.0),
                "unique_chatters": chat_snap.get("unique_chatters", 0),
                "emote_rate": chat_snap.get("emote_rate", 0.0),
                "excitement_rate": chat_snap.get("excitement_rate", 0.0),
                "top_emotes": chat_snap.get("top_emotes", []),
                "top_excitement": chat_snap.get("top_excitement", []),
            },
            "audio_activity": {
                "current_db": audio_snap.get("current_db", -60.0),
                "peak_db": audio_snap.get("peak_db", -60.0),
                "z_score": audio_snap.get("z_score", 0.0),
                "jump_pct": audio_snap.get("jump_pct", 0.0),
                "silence_to_loud": audio_snap.get("silence_to_loud", False),
                "wps": transcript_snap.get("wps", 0.0),
                "category": transcript_snap.get("category", "other"),
            },
        }

        if should_clip:
            self.last_triggered_sec = now_sec
            self.history.append(moment)
            if len(self.history) > 100:
                self.history = self.history[-100:]

        return moment

