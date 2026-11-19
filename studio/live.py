"""Live sessions: record a stream into a rolling buffer, transcribe as it goes, let the AI pick clips."""
import collections
import json
import re
import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import alerts, auth, brain, chat_intel, config, creators, ctx, jobs, load, media, platforms, transcribe, usage, viral_engine

SESSIONS: dict = {}
SEG_SEC = 6           # length of each buffer piece
PAD = 8               # seconds kept either side of a clip so it can be re-trimmed later
CHECK_EVERY = 180     # look at the stream every 3 minutes...
QUIET_MAX = 600       # ...but only ask the AI when something happened, or at most every 10 minutes when calm
WINDOW = 240          # seconds of transcript shown to the AI per check
SR = 16000


class NotLive(Exception):
    pass


def media_safe(e) -> str:
    from . import safe
    return safe.friendly(e, 300)


def normalize(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url):
        url = "https://" + url
    m = re.match(r"^(https?://(?:www\.)?youtube\.com/(?:@[^/?#]+|channel/[^/?#]+|c/[^/?#]+))/?(?:[?#].*)?$", url)
    return m.group(1) + "/live" if m else url


class _Quiet:
    def debug(self, m): pass
    def info(self, m): pass
    def warning(self, m): pass
    def error(self, m): pass


def resolve(url: str) -> dict:
    """Ask yt-dlp for the stream's playable address. Raises NotLive if the channel is offline."""
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "logger": _Quiet(),
            "nocheckcertificate": config.INSECURE_TLS, "format": "b[height<=720]/bv*[height<=720]+ba/b"}
    if media.is_youtube(url) and media.youtube_paused():
        raise NotLive(media.YT_BLOCKED)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:  # noqa: BLE001
        if media.note_refusal(url, e):
            raise NotLive(media.YT_BLOCKED) from e
        msg = str(e)
        if re.search(r"not currently live|is offline|not live|no live stream|premieres in", msg, re.I):
            raise NotLive("This channel is not live right now.") from e
        clean_msg = re.sub(r";?\s*please report this issue on https?://\S+.*", "", msg, flags=re.I).strip()
        clean_msg = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*[^:]+:\s*)?", "", clean_msg).strip()
        raise NotLive((clean_msg or "This channel is not live right now.")[-200:]) from e
    if not info or not info.get("is_live"):
        raise NotLive("This channel is not live right now.")
    fmts = info.get("requested_formats") or [info]
    title = info.get("title") or ""
    if "(live)" in title and info.get("description"):  # Twitch puts the real stream title here
        title = str(info["description"]).splitlines()[0][:120]
    return {"urls": [f["url"] for f in fmts], "title": title,
            "uploader": info.get("uploader") or info.get("channel") or ""}


def active():
    return [s for s in SESSIONS.values() if not s.finished and not s.user_stopped]


def active_mine():
    u = ctx.uid()
    return [s for s in active() if s.job["user"] == u]


def active_for(cid):
    return next((s for s in active_mine() if s.job["creator"] == cid), None)


def mine(jid):
    """The signed-in user's recording with this id, or None."""
    s = SESSIONS.get(jid)
    return s if s and s.job["user"] == ctx.uid() else None


class Session:
    def __init__(self, job, settings):
        self.job, self.settings = job, dict(creators.DEFAULTS, **(settings or {}))
        self.dir = Path(job["dir"])
        self.buf = self.dir / "buffer"
        self.buf.mkdir(exist_ok=True)
        (self.dir / "clips").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.stop_flag = threading.Event()
        self.ended = False            # recording has stopped (stream over, stopped, or failed)
        self.finished = False         # everything is wrapped up
        self.end_note = "Stream ended"
        self.user_stopped = False     # Stop pressed: wrap up at once, no last AI check
        self.proc = None
        self.pending = bytearray()    # audio received but not yet transcribed
        self.clock_fix = 0.0          # how far the video-piece clock is behind the audio clock, last measured
        self.pending_base = 0         # sample index where `pending` starts
        self.samples = 0              # total audio samples received
        self.db = []                  # loudness per second
        self.words = []
        self.t_done = 0.0             # transcript is complete up to this second
        self.segments = []            # (path, start, end) on the session clock
        self.seg_clock = 0.0
        self.csv_seen = {}
        self.part = 0
        self.chat = collections.Counter()   # second -> accepted messages
        self.chat_total = 0
        self.chat_on = False
        self.last_check = 0.0
        self.words_at_check = 0
        self.checks = 0
        self.rejected = 0
        self.verdicts = []
        self.last_trigger = -999.0
        self.started_at = time.time()
        self.skipped = 0
        self.last_ai = 0.0
        self.tokens = 0
        self.manual = []
        self.n = 0
        self._last_sample_wall = 0.0
        self._initial_check_done = False
        self.spike = None             # a jump in loudness/chat/viral score waiting for post-roll & transcript
        self.said = None              # someone said "clip": that moment, waiting for its transcript
        self.last_said = -999.0
        self.last_spike = -999.0      # when the last jump was sent to the AI (stream seconds)
        self.base = {}                # running averages the jumps are measured against, shown on the card
        self.spikes = 0

        # Autonomous Live Chat & Viral Scoring Engines (Phases 3-7)
        self.adapter = None
        try:
            self.adapter = platforms.get_adapter(self.job["source"]["url"])
        except Exception:
            self.adapter = None
        self.chat_engine = chat_intel.ChatIntelligenceEngine(
            spike_multiplier=float(self.settings.get("chat_spike_mult", 3.0)),
            min_unique_chatters=int(self.settings.get("min_unique_chatters", 3)),
            cooldown_sec=float(self.settings.get("cooldown_sec", 45.0)),
        )
        raw_w = self.settings.get("viral_weights") or {}
        self.viral_engine = viral_engine.ViralScoringEngine(
            weights={k: float(v) for k, v in raw_w.items()} if isinstance(raw_w, dict) and raw_w else None,
            min_viral_score=float(self.settings.get("min_score", 65)),
            pre_roll_sec=float(self.settings.get("pre_roll_sec", 30)),
            post_roll_sec=float(self.settings.get("post_roll_sec", 20)),
            cooldown_sec=float(self.settings.get("cooldown_sec", 45)),
            sensitivity=str(self.settings.get("sensitivity", "medium")),
        )
        self.current_viral: dict = {}
        self.chat_metrics: dict = {}
        self.audio_metrics: dict = {}
        self.detected_moments: list = []      # timeline of detected viral moments
        self.pending_approvals: list = []     # moments awaiting manual user approval when manual_approval=True

    def update_config(self, new_settings: dict):
        with self.lock:
            self.settings.update(new_settings)
            self.chat_engine.configure(
                spike_multiplier=self.settings.get("chat_spike_mult"),
                min_unique_chatters=self.settings.get("min_unique_chatters"),
                cooldown_sec=self.settings.get("cooldown_sec"),
            )
            self.viral_engine.configure(
                sensitivity=self.settings.get("sensitivity"),
                min_viral_score=self.settings.get("min_score"),
                pre_roll_sec=self.settings.get("pre_roll_sec"),
                post_roll_sec=self.settings.get("post_roll_sec"),
                cooldown_sec=self.settings.get("cooldown_sec"),
                weights=self.settings.get("viral_weights"),
            )
        self._publish()

    # ---------------------------------------------------------------- recording
    def start(self):
        SESSIONS[self.job["id"]] = self
        threading.Thread(target=ctx.bind(self._ingest), daemon=True).start()
        threading.Thread(target=ctx.bind(self._analyze), daemon=True).start()
        if self.adapter is not None:
            threading.Thread(target=ctx.bind(self._chat_worker), daemon=True).start()

    def stop(self, note="Stopped by you"):
        """Instant for the user: the card and the creator show as stopped straight away. Clips already being made
        still finish in the background, but nothing new is transcribed or sent to the AI."""
        self.end_note = note
        self.user_stopped = True
        self.stop_flag.set()
        if self.adapter is not None:
            self.adapter.disconnect()
        with self.lock:
            self.job.update(status="done", stage=note)
        jobs.save(self.job)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def stream_clock(self) -> float:
        base = self.samples / SR
        if self.samples > 0 and not self.ended and self._last_sample_wall > 0:
            return base + min(2.5, max(0.0, time.time() - self._last_sample_wall))
        return base

    def _ingest(self):
        fails = 0
        url = self.job["source"]["url"]
        try:
            while not self.stop_flag.is_set():
                try:
                    info = resolve(url)
                except NotLive as e:
                    fails += 1
                    if self.samples == 0 or fails >= 3:
                        if self.samples == 0:
                            self.job["error"] = f"Could not start recording: {e}"
                        break
                    self.stop_flag.wait(10)
                    continue
                if not self.job.get("uploader"):
                    self.job.update(title=info["title"], uploader=info["uploader"])
                    jobs.save(self.job)
                self.part += 1
                with self.lock:
                    self.seg_clock = max(self.seg_clock, self.samples / SR)
                p = f"p{self.part:02d}"
                cmd = ["ffmpeg", "-v", "error"]
                for u in info["urls"]:
                    if str(u).startswith(("http://", "https://")):
                        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
                    cmd += ["-i", u]
                a_in = "1:a:0" if len(info["urls"]) > 1 else "0:a:0"
                cmd += ["-map", "0:v:0", "-map", a_in, "-c", "copy", "-f", "segment", "-segment_time", str(SEG_SEC),
                        "-reset_timestamps", "1", "-segment_format", "mpegts", "-segment_list", f"{p}.csv",
                        "-segment_list_type", "csv", f"{p}_%05d.ts",
                        "-map", a_in, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"]
                with open(self.dir / "ffmpeg.log", "ab") as errlog:
                    self.proc = subprocess.Popen(cmd, cwd=self.buf, stdout=subprocess.PIPE, stderr=errlog,
                                                 creationflags=media.NO_WINDOW)
                    got = 0
                    while True:
                        data = self.proc.stdout.read(SR * 2)  # one second of 16-bit mono
                        if not data:
                            break
                        data = data[: len(data) // 2 * 2]
                        got += len(data)
                        x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        level = float(20 * np.log10(np.sqrt((x ** 2).mean() + 1e-10))) if len(x) else -100.0
                        with self.lock:
                            self.pending += data
                            self.samples += len(data) // 2
                            self.db.append(level)
                            self._last_sample_wall = time.time()
                    self.proc.wait()
                fails = 0 if got > SR * 2 * 20 else fails + 1
                if not self.stop_flag.is_set() and fails >= 3:
                    if self.samples < SR * 5 and not self.job.get("error"):
                        try:
                            err_lines = [ln.strip() for ln in (self.dir / "ffmpeg.log").read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
                        except OSError:
                            err_lines = []
                        clean_errs = [ln for ln in err_lines if not ln.startswith("Error opening input file http") and "Error opening input files:" not in ln]
                        last_err = clean_errs[-1] if clean_errs else (err_lines[-1] if err_lines else f"Stream recording stopped unexpectedly (exit code {self.proc.returncode}).")
                        self.job["error"] = f"Could not record stream: {media_safe(last_err)}"
                    break
                if fails >= 3:
                    break
                self.stop_flag.wait(3)
        finally:
            self.ended = True

    def _pinned_ranges(self) -> list:
        """Time ranges (start, end) that must NOT be pruned from the rolling buffer because a viral moment,
        manual clip, pending approval, or queued render needs their pre-roll/post-roll segments."""
        ranges = []
        pre = float(self.settings.get("pre_roll_sec", 30))
        post = float(self.settings.get("post_roll_sec", 20))
        if self.spike:
            s0 = self.spike.get("start", max(0.0, self.spike["t"] - pre))
            e0 = self.spike.get("end", self.spike["t"] + post)
            ranges.append((max(0.0, s0 - PAD), e0 + PAD))
        if self.said:
            ranges.append((max(0.0, self.said["t"] - self.SAID_BEFORE - PAD), self.said["t"] + self.SAID_AFTER + PAD))
        for m in self.manual:
            ranges.append((max(0.0, m["start"] - PAD), m["end"] + PAD))
        now_wall = time.time()
        for p in self.pending_approvals:
            if p.get("status") == "pending" and now_wall - p.get("wall_time", now_wall) < 900:
                ranges.append((max(0.0, p["start"] - PAD), p["end"] + PAD))
        for c in self.job.get("clips", []):
            if c.get("status") in ("queued", "rendering"):
                ranges.append((max(0.0, float(c.get("start", 0)) - PAD), float(c.get("end", 0)) + PAD))
        return ranges

    def _scan_segments(self):
        """Pick up buffer pieces ffmpeg has finished writing, while preserving any pinned pre-roll segments."""
        with self.lock:
            seen_names = {s[0].name for s in self.segments}
            for i in range(1, self.part + 1):
                f = self.buf / f"p{i:02d}.csv"
                if f.exists():
                    rows = [r.split(",") for r in f.read_text(encoding="utf-8", errors="ignore").splitlines() if r.count(",") >= 2]
                    for name, a, b in (r[:3] for r in rows[self.csv_seen.get(i, 0):]):
                        try:
                            dur = max(0.0, float(b) - float(a))
                        except ValueError:
                            continue
                        seg_p = self.buf / name
                        self.segments.append((seg_p, self.seg_clock, self.seg_clock + dur))
                        seen_names.add(name)
                        self.seg_clock += dur
                    self.csv_seen[i] = len(rows)
            if self.ended:
                for i in range(1, self.part + 1):
                    for seg_p in sorted(self.buf.glob(f"p{i:02d}_*.ts")):
                        if seg_p.name in seen_names:
                            continue
                        try:
                            if seg_p.stat().st_size <= 16384:
                                continue
                            dur = 0.0
                            try:
                                dur = float(media.probe(seg_p).get("duration") or 0.0)
                            except Exception:  # noqa: BLE001
                                dur = 0.0
                            if dur <= 0.1:
                                dur = max(2.0, min(float(SEG_SEC), self.samples / SR - self.seg_clock))
                            self.segments.append((seg_p, self.seg_clock, self.seg_clock + dur))
                            seen_names.add(seg_p.name)
                            self.seg_clock += dur
                        except OSError:
                            pass
            keep_from = self.seg_clock - self.settings["buffer_min"] * 60
            pinned = self._pinned_ranges()
            if pinned:
                earliest_pin = min(r[0] for r in pinned)
                keep_from = min(keep_from, earliest_pin)
            while self.segments and self.segments[0][2] < keep_from:
                self.segments.pop(0)[0].unlink(missing_ok=True)

    # ---------------------------------------------------------------- chat (Twitch, Kick, YouTube Live)
    def _on_chat_message(self, msg: platforms.ChatMessage):
        accepted = self.chat_engine.ingest(msg)
        with self.lock:
            self.chat_on = True
            if accepted:
                self.chat[int(msg.stream_sec)] += 1
                self.chat_total += 1

    def _chat_worker(self):
        if self.adapter is None:
            return
        url = self.job["source"]["url"]
        self.adapter.connect_chat(
            url,
            self._on_chat_message,
            self.stop_flag,
            self.stream_clock,
        )
        self.chat_on = False

    def _chat_spike(self, s, e):
        """True if chat was at least 3x busier than usual during (and just after) these seconds."""
        elapsed = self.samples / SR
        if not self.chat_total or elapsed < 60:
            return False
        avg = self.chat_total / elapsed
        a, b = int(s), int(e) + 6
        rate = sum(self.chat.get(t, 0) for t in range(a, b + 1)) / (b - a + 1)
        return rate > max(3 * avg, 1.0)

    # ---------------------------------------------------------------- analysis
    def _transcribe_step(self):
        with self.lock:
            n = len(self.pending) // 2
            hurry = self.ended or bool(self.manual)
            if n < (SR if hurry else SR * 7):   # about every 7 s, so "what was just said" really is live
                return False
            audio = np.frombuffer(bytes(self.pending[: n * 2]), dtype=np.int16)
            base = self.pending_base
        cut = n
        if not hurry or n > SR * 24:
            # cut at the quietest moment between 4 and 6.5 s, so no word is split; the rest waits for the next pass
            _, cut = next(transcribe._chunks(audio, 4.0, 6.5))
        words = transcribe.transcribe(audio[:cut])
        off = base / SR
        for w in words:
            w["s"], w["e"] = round(w["s"] + off, 2), round(w["e"] + off, 2)
        with self.lock:
            del self.pending[: cut * 2]
            self.pending_base += cut
            self.words += words
            self.t_done = self.pending_base / SR
        self._triggers(words)
        (self.dir / "words.json").write_text(json.dumps(self.words), encoding="utf-8")
        return True

    AUTO_BEFORE, AUTO_AFTER, AUTO_GAP = 30, 20, 60   # default pre-roll, post-roll, and cooldown seconds
    JUMP = 0.6                                        # 60% louder or busier than the last three minutes

    def _watch_spikes(self):
        """Run the Live Chat Intelligence Engine, Audio Intelligence, and Viral Moment Scoring Engine."""
        with self.lock:
            n = len(self.db)
            now_sec = max(float(n), self.samples / SR)
            db_copy = list(self.db[-200:])
            words_copy = list(self.words[-120:])

        pre_roll = float(self.settings.get("pre_roll_sec", self.AUTO_BEFORE))
        post_roll = float(self.settings.get("post_roll_sec", self.AUTO_AFTER))
        cooldown = float(self.settings.get("cooldown_sec", self.AUTO_GAP))

        chat_snap = self.chat_engine.analyze(now_sec)
        audio_snap = viral_engine.analyze_audio_window(db_copy, now_sec)
        transcript_snap = viral_engine.analyze_transcript_window(words_copy, now_sec)
        chat_conn = bool(self.chat_on or self.chat_total > 0 or (self.adapter and self.adapter.status == "connected"))
        warming = n < 12 and chat_snap["total_messages"] < 3
        prev_trig = self.viral_engine.last_triggered_sec
        moment = self.viral_engine.evaluate(
            now_sec, chat_snap, audio_snap, transcript_snap, chat_connected=chat_conn
        )
        if warming:
            self.viral_engine.last_triggered_sec = prev_trig

        with self.lock:
            self.chat_metrics = chat_snap
            self.audio_metrics = audio_snap
            self.current_viral = moment
            if warming:
                self.base = {"warming": True, "viral_score": moment["viral_score"]}
                return
            self.base = {
                "loud_db": audio_snap["baseline_db"],
                "loud_now": round(audio_snap["jump_pct"]),
                "chat": chat_snap["baseline_mps"],
                "chat_now": chat_snap["mps"],
                "unique_chatters": chat_snap["unique_chatters"],
                "viral_score": moment["viral_score"],
                "pct": round(float(self.settings.get("auto_jump_pct", 60))),
            }

            if moment["should_clip"]:
                ev_t = max(0.0, now_sec - 2.0)
                start_s = round(max(0.0, ev_t - pre_roll), 2)
                end_s = round(ev_t + post_roll, 2)
                why_str = "; ".join(moment["reasons"][:3]) or f"Viral score {moment['viral_score']}"
                det = {
                    "id": f"mom_{int(now_sec)}_{len(self.detected_moments) + 1}",
                    "t": round(ev_t, 1),
                    "start": start_s,
                    "end": end_s,
                    "score": moment["viral_score"],
                    "reasons": moment["reasons"],
                    "why": why_str,
                    "chat": moment["chat_activity"],
                    "audio": moment["audio_activity"],
                    "components": moment["component_scores"],
                    "wall_time": time.time(),
                    "status": "pending" if (self.settings.get("manual_approval") or not self.settings.get("auto_clip", True)) else "auto_queued",
                }
                self.detected_moments.append(det)
                del self.detected_moments[:-40]

                if self.settings.get("manual_approval") or not self.settings.get("auto_clip", True):
                    self.pending_approvals.append(det)
                    del self.pending_approvals[:-20]
                    self.spikes += 1
                else:
                    # Merge with an existing pending spike if overlapping, otherwise set new spike
                    if self.spike and start_s <= self.spike.get("end", self.spike["t"] + post_roll):
                        self.spike["end"] = max(self.spike.get("end", self.spike["t"] + post_roll), end_s)
                        if moment["viral_score"] > self.spike.get("score", 0):
                            self.spike["score"] = moment["viral_score"]
                            self.spike["why"] = why_str
                    elif now_sec - self.last_spike >= cooldown:
                        self.spike = {
                            "t": ev_t,
                            "start": start_s,
                            "end": end_s,
                            "score": moment["viral_score"],
                            "why": why_str,
                            "reasons": moment["reasons"],
                        }
                        self.last_spike = now_sec
                        self.spikes += 1

    def approve_moment(self, moment_id: str) -> dict:
        """Manually approve a detected viral moment and queue it for immediate clip generation."""
        with self.lock:
            target = next((m for m in self.pending_approvals if m["id"] == moment_id), None)
            if not target:
                target = next((m for m in self.detected_moments if m["id"] == moment_id), None)
            if not target:
                raise ValueError("Moment not found.")
            target["status"] = "approved"
            self.manual.append({
                "start": target["start"],
                "end": target["end"],
                "score": target.get("score", 80),
                "why": f"Approved viral moment ({target.get('score', 80)} pts): {target.get('why', '')}",
            })
        self._publish()
        return target

    def _fallback_live_clip(self, ws, now, viral_meta=None, why=None):
        st = self.settings
        min_len = float(st.get("min_len", 15))
        max_len = float(st.get("max_len", 60))
        dur = max(0.0, now - ws)
        if dur < 5.0:
            return None
        target_len = min(max_len, max(min_len, 35.0))
        if dur <= max_len:
            s, e = ws, now
        else:
            with self.lock:
                db_copy = list(self.db)
                chat_copy = dict(self.chat)
            z = brain.loud_zscore(np.array(db_copy, dtype=np.float32)) if len(db_copy) >= 5 else np.zeros(max(1, int(now) + 1))
            best_t, best_v = (ws + now) / 2.0, -1e9
            for t in range(int(ws), int(now) + 1):
                lz = float(z[t]) if 0 <= t < len(z) else 0.0
                ch = float(chat_copy.get(t, 0))
                val = lz + 1.5 * ch
                if val > best_v:
                    best_v = val
                    best_t = float(t)
            s = max(ws, min(now - target_len, best_t - target_len * 0.6))
            e = min(now, s + target_len)
        s, e = round(s, 2), round(e, 2)
        if e - s < 4.0:
            return None
        if self.job["clips"] and any(brain._overlaps(s, e, [k]) for k in self.job["clips"]):
            return None
        with self.lock:
            win_words = [w for w in self.words if s <= w["s"] <= e]
        text = " ".join(w["w"] for w in win_words) or (self.job.get("title") or "Insane live stream reaction moment")
        score = int((viral_meta or {}).get("score") or max(68, int(st.get("min_score", 65))))
        c = brain._smart_describe(text, self.job.get("title", ""), self.job.get("uploader", ""),
                                  s=s, e=e, score=score)
        c["reason"] = (viral_meta or {}).get("why") or why or "Live stream audio & chat highlight"
        self._add_clip(c)
        return c

    def _ai_check(self, window=None, why=None, viral_meta=None):
        """Judge 1 reads only the words. If it finds a moment, judge 2 reads only the loudness and chat activity
        for the same stretch, without seeing the words. A clip is made when both say yes (or when triggered by a
        confirmed multi-signal viral moment)."""
        st = self.settings
        every = int(st.get("ai_every_min", 3)) or 3
        if window:                    # just the stretch around a jump, or around someone saying "clip"
            ws, now = max(0.0, window[0]), window[1]
        else:
            now = self.t_done
            ws = max(0.0, now - max(every, 3) * 60 - 90)  # each check overlaps the last by 90 s, so nothing falls between
        self.last_check = self.t_done
        self._initial_check_done = True
        win = [w for w in self.words if ws <= w["s"] <= now]
        recent = [c for c in self.job["clips"] if c.get("score", 0) > 0 and time.time() - c.get("made_at", 0) < 3600]
        last_clip = max((c.get("made_at", 0) for c in self.job["clips"]), default=self.started_at)
        must = st["must_clip_min"] > 0 and time.time() - last_clip >= st["must_clip_min"] * 60
        verdict = {"t": round(now), "transcript": None, "reaction": None, "trigger": why}

        with self.lock:
            db_copy = list(self.db)
            chat_cnt = sum(self.chat.get(t, 0) for t in range(int(ws), int(now) + 1))
            total_chat = self.chat_total
        audio_snap = (
            viral_engine.analyze_audio_window(db_copy[max(0, int(ws)):max(0, int(now) + 1)], now)
            if len(db_copy) >= 5 else {"audio_score": 0.0, "peak_db": -60.0}
        )

        # If speech is sparse (e.g. non-English stream, pure reaction, or initial/high-activity check),
        # create a clip directly using audio/chat activity around the peak moment.
        if len(win) < 8:
            if (
                (viral_meta and viral_meta.get("score", 0) >= st.get("min_score", 65))
                or must
                or audio_snap.get("audio_score", 0) >= 45
                or audio_snap.get("peak_db", -60) > -28
                or chat_cnt >= 3
                or (total_chat > 0 and now - ws >= st.get("min_len", 15))
                or (len(self.job["clips"]) == 0 and now - ws >= st.get("min_len", 15))
            ):
                c = self._fallback_live_clip(ws, now, viral_meta=viral_meta, why=why)
                if c:
                    self._verdict(verdict, "Clipped: " + c["reason"][:60])
                    return
            self.skipped += 1
            return self._verdict(verdict, "Skipped: nobody talking")

        if len(recent) >= st["live_per_hour"] and not must:
            self.skipped += 1
            return self._verdict(verdict, f"Skipped: {st['live_per_hour']} clips this hour already")
        usage.set_context(self.job["creator"], self.job["title"] or "Live stream")
        title, up = self.job["title"], self.job["uploader"]
        try:
            cands, tok = brain.judge_transcript(win, ws, now, title, up, st["min_len"], st["max_len"], self.job["clips"],
                                                n=1 if every <= 2 else 2, must=must or bool(viral_meta))
        except Exception as e:  # noqa: BLE001
            self.job["ai"]["last_error"] = str(e)[-200:]
            if viral_meta or must or (len(self.job["clips"]) == 0 and now >= 30):
                c = self._fallback_live_clip(ws, now, viral_meta=viral_meta, why=why)
                if c:
                    self._verdict(verdict, "Clipped: " + c["reason"][:60])
                    return
            return self._verdict(verdict, "AI error: " + str(e)[-80:])
        self.checks += 1
        self.tokens += tok
        if viral_meta and cands:
            for c in cands:
                c["score"] = max(c.get("score", 0), int(viral_meta.get("score", 0)))
        keep = [c for c in cands if c["score"] >= (40 if must else st["min_score"])]
        verdict["transcript"] = max((c["score"] for c in cands), default=0)
        if not keep:
            if viral_meta or must or (len(self.job["clips"]) == 0 and now >= 30):
                c = self._fallback_live_clip(ws, now, viral_meta=viral_meta, why=why)
                if c:
                    self._verdict(verdict, "Clipped: " + c["reason"][:60])
                    return
            return self._verdict(verdict, "Words judge: nothing clip-worthy")
        z = brain.loud_zscore(np.array(self.db, dtype=np.float32))
        lo, hi = int(ws), int(now) + 1
        chat = [self.chat.get(t, 0) for t in range(lo, hi)] if (self.chat_on or self.chat_total) else None
        try:
            sig, tok2 = brain.judge_signals(z[lo:hi], chat, lo, now)
            self.checks += 1
            self.tokens += tok2
        except Exception as e:  # noqa: BLE001
            sig = []
            self.job["ai"]["last_error"] = str(e)[-200:]
        added = 0
        for c in keep:
            match = [m for m in sig if m["end"] >= c["start"] - 10 and m["start"] <= c["end"] + 10 and m["score"] >= 50]
            rs = max((m["score"] for m in match), default=int(viral_meta.get("score", 0)) if viral_meta else 0)
            verdict["reaction"] = rs
            both = bool(match) or bool(viral_meta)
            ok = both or (st["decide_mode"] == "either" and c["score"] >= 90) or (must and added == 0)
            if not ok:
                self.rejected += 1
                self._verdict(verdict, "Reaction judge said no, not clipped")
                continue
            min_gap_sec = max(15.0, float(st.get("min_gap_min", 1)) * 60.0)
            if any(abs(c["start"] - k["start"]) < min_gap_sec or brain._overlaps(c["start"], c["end"], [k]) for k in self.job["clips"]):
                self._verdict(verdict, "Too close to the last clip")
                continue
            why_label = viral_meta.get("why") if viral_meta else (
                "Both judges agreed" if both else ("Words judge was very sure" if c["score"] >= 90 and
                                                   st["decide_mode"] == "either" else "Forced: no clip for a while")
            )
            c["judges"] = {"words": c["score"], "reaction": rs or None, "why": why_label}
            c["reason"] = f"{c['reason']} ({why_label})"
            self._add_clip(c)
            self._verdict(verdict, "Clipped: " + str(why_label)[:60])
            added += 1
            must = False
            if added >= (1 if every <= 2 else 2):
                break
        if added == 0 and (viral_meta or must or (len(self.job["clips"]) == 0 and now >= 30)):
            c = self._fallback_live_clip(ws, now, viral_meta=viral_meta, why=why)
            if c:
                self._verdict(verdict, "Clipped: " + c["reason"][:60])

    def _verdict(self, v, result):
        self.verdicts.append(dict(v, result=result))
        del self.verdicts[:-10]

    CLIP_WORDS = re.compile(r"\bclip(\s+(it|that|this))?\b", re.I)
    SAID_BEFORE, SAID_AFTER, SAID_GAP = 30, 20, 45

    def _triggers(self, new_words):
        """Built in: when someone says "clip" ("clip it", "clip that"), the AI looks at that moment as soon as its
        transcript is ready; people say it right after something worth clipping. Then the creator's own trigger
        phrases from Setup, which clip straight away with no AI check."""
        pre = float(self.settings.get("pre_roll_sec", self.SAID_BEFORE))
        post = float(self.settings.get("post_roll_sec", self.SAID_AFTER))
        for w in new_words or []:
            if self.CLIP_WORDS.search(re.sub(r"[^a-z' ]", "", w["w"].lower())) and w["s"] - self.last_said > self.SAID_GAP \
                    and not self.said:
                self.said = {"t": w["s"], "start": max(0.0, w["s"] - pre), "end": w["s"] + post, "why": 'Someone said "clip"'}
                self.last_said = w["s"]
        phrases = [p.strip().lower() for p in self.settings.get("trigger_words", "").split(",") if p.strip()]
        if not phrases or not new_words:
            return
        key = [re.sub(r"[^a-z0-9']", "", w["w"].lower()) for w in new_words]
        for p in phrases:
            parts = [re.sub(r"[^a-z0-9']", "", x) for x in p.split()]
            for i in range(len(key) - len(parts) + 1):
                if key[i:i + len(parts)] == parts:
                    t = new_words[i]["s"]
                    if t - self.last_trigger > 45:
                        self.last_trigger = t
                        self.manual.append({"start": max(0.0, t - pre), "end": t + 5, "why": f'Someone said "{p}"'})

    def _manual_step(self):
        cur_sec = self.samples / SR
        for req in [m for m in list(self.manual) if m["end"] <= max(self.t_done, cur_sec) or self.ended]:
            self.manual.remove(req)
            text = " ".join(w["w"] for w in self.words if req["start"] <= w["s"] <= req["end"])
            usage.set_context(self.job["creator"], self.job["title"] or "Live stream")
            c = brain.describe(text, self.job["title"], self.job["uploader"])
            c.update(
                start=round(req["start"], 2),
                end=round(min(req["end"], max(req["start"] + 5.0, cur_sec)), 2),
                score=int(req.get("score") or c.get("score") or 80),
            )
            if req.get("why"):
                c["reason"] = req["why"]
            self._add_clip(c)

    def _add_clip(self, c):
        self.n += 1
        c.update(id=f"clip{self.n:03d}", style=self.settings["style"], layout=self.settings["layout"],
                 status="queued", version=0, made_at=time.time())
        with self.lock:
            self.job["clips"].append(c)
        jobs.save(self.job)
        jobs.submit_live(lambda: self._render(c))

    def _render(self, clip):
        """Runs on the render worker: copy the needed buffer pieces, then make the clip."""
        try:
            deadline = time.time() + 45
            while time.time() < deadline:
                self._scan_segments()
                if self.ended or (self.segments and self.segments[-1][2] >= clip["end"] + 1):
                    break
                time.sleep(1)
            with self.lock:
                segs = [s for s in self.segments if s[2] > clip["start"] - PAD and s[1] < clip["end"] + PAD and s[0].exists()]
            overlap = (min(clip["end"], segs[-1][2]) - max(clip["start"], segs[0][1])) if segs else 0.0
            if not segs or segs[0][1] > clip["start"] + 15 or overlap < min(4.0, (clip["end"] - clip["start"]) * 0.4):
                clip.update(status="error", error="That part of the stream is no longer in the buffer.")
                jobs.save(self.job)
                return
            if "width" not in self.job:
                meta = media.probe(segs[0][0])
                self.job.update(width=meta["width"], height=meta["height"])
            lst = self.dir / "clips" / f"{clip['id']}.txt"
            lst.write_text("".join(f"file '{s[0].as_posix()}'\n" for s in segs), encoding="utf-8")
            src = self.dir / "clips" / f"{clip['id']}.src.ts"
            try:
                media.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                           "-c", "copy", str(src)])
            except Exception as e:  # noqa: BLE001
                clip.update(status="error", error=media_safe(e))
                jobs.save(self.job)
                return
            finally:
                lst.unlink(missing_ok=True)
            dur = segs[-1][2] - segs[0][1]
            offset, heard = self._align(src, segs[0][1] + self.clock_fix, dur)
            self.clock_fix = offset - segs[0][1]  # the two clocks drift ~4 s an hour; start the next clip from here
            clip.update(src=str(src), src_offset=round(offset, 2), src_end=round(offset + dur, 2))
            clip["start"] = max(clip["start"], clip["src_offset"])
            clip["end"] = min(clip["end"], clip["src_end"])
            if clip["end"] - clip["start"] < 2.0:
                clip["start"] = clip["src_offset"]
                clip["end"] = clip["src_end"]
            shifted_heard = [dict(w, s=round(w["s"] + offset, 2), e=round(w["e"] + offset, 2)) for w in heard]
            with self.lock:
                live_words = list(self.words)
            live_in_clip = [w for w in live_words if clip["start"] <= w["s"] <= clip["end"]]
            heard_in_clip = [w for w in shifted_heard if clip["start"] <= w["s"] <= clip["end"]]
            render_words = shifted_heard if len(heard_in_clip) > len(live_in_clip) else live_words
            if len(live_in_clip) < 4 and len(heard_in_clip) >= 4:
                rich = brain._smart_describe(
                    " ".join(w["w"] for w in heard_in_clip),
                    self.job.get("title", ""),
                    self.job.get("uploader", ""),
                    s=clip["start"],
                    e=clip["end"],
                    score=clip.get("score", 75),
                )
                for fk in ("hook", "title", "description", "hashtags", "highlight_words"):
                    if rich.get(fk):
                        clip[fk] = rich[fk]
            jobs.render_one(self.job, clip, render_words)
        except Exception as e:  # noqa: BLE001
            clip.update(status="error", error=media_safe(e))
            jobs.save(self.job)

    def _align(self, src, guess, dur):
        """Where `src` really starts on the transcript clock. The video pieces and the audio feed keep separate
        clocks that drift apart, so transcribe the clip's own audio and line its words up with the live words.
        Local speech model only: no AI calls."""
        key = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
        heard = []
        try:
            wav = src.with_suffix(".wav")
            media.extract_audio(src, wav)
            heard = transcribe.transcribe(media.load_audio(wav))
            wav.unlink(missing_ok=True)
            with self.lock:
                near = [w for w in self.words if guess - 15 <= w["s"] <= guess + dur + 15]
            diffs = []
            for h in heard:
                k = key(h["w"])
                if len(k) < 3:
                    continue
                m = [w["s"] for w in near if key(w["w"]) == k and abs(w["s"] - (guess + h["s"])) < 8]
                if m:
                    diffs.append(min(m, key=lambda s: abs(s - (guess + h["s"]))) - h["s"])
            if len(diffs) >= 4:
                return float(np.median(diffs)), heard
        except Exception:  # noqa: BLE001 - fall back to the piece clock
            pass
        return guess, heard

    def clip_last(self, seconds):
        end = self.samples / SR
        if end < 5:
            raise ValueError("Nothing has been recorded yet.")
        self.manual.append({"start": max(0.0, end - seconds), "end": end, "score": 85, "why": "Clipped manually from Live Monitor"})
        self._publish()

    def _publish(self):
        elapsed = self.samples / SR
        lines = brain.make_lines([w for w in self.words if w["s"] >= self.t_done - 45])[-5:]
        conn_state = self.adapter.connection_state() if self.adapter else {"platform": "", "status": "disconnected", "connected": False, "reconnects": 0}
        with self.lock:
            sec = int(elapsed)
            self.job.update(
                duration=elapsed,
                live={
                    "elapsed": round(elapsed),
                    "buffer_sec": round(self.segments[-1][2] - self.segments[0][1]) if self.segments else 0,
                    "lines": [l["text"] for l in lines],
                    "level": [round(x) for x in self.db[-60:]],
                    "chat": [self.chat.get(t, 0) for t in range(max(0, sec - 60), sec)] if (self.chat_on or self.chat_total or conn_state["connected"]) else None,
                    "chat_connection": conn_state,
                    "chat_intel": {
                        "mps": self.chat_metrics.get("mps", 0.0),
                        "baseline_mps": self.chat_metrics.get("baseline_mps", 0.0),
                        "velocity_ratio": self.chat_metrics.get("velocity_ratio", 0.0),
                        "unique_chatters": self.chat_metrics.get("unique_chatters", 0),
                        "emote_rate": self.chat_metrics.get("emote_rate", 0.0),
                        "excitement_rate": self.chat_metrics.get("excitement_rate", 0.0),
                        "top_emotes": self.chat_metrics.get("top_emotes", []),
                        "top_excitement": self.chat_metrics.get("top_excitement", []),
                        "repeated_phrases": self.chat_metrics.get("repeated_phrases", []),
                        "total_messages": self.chat_metrics.get("total_messages", self.chat_total),
                        "spam_filtered": self.chat_metrics.get("spam_filtered", 0),
                        "recent_messages": self.chat_metrics.get("recent_messages", []),
                    },
                    "viral": {
                        "score": self.current_viral.get("viral_score", 0),
                        "threshold": self.viral_engine.min_viral_score,
                        "reasons": self.current_viral.get("reasons", []),
                        "components": self.current_viral.get("component_scores", {}),
                        "weights": self.viral_engine.weights,
                        "pre_roll_sec": self.viral_engine.pre_roll_sec,
                        "post_roll_sec": self.viral_engine.post_roll_sec,
                        "cooldown_sec": self.viral_engine.cooldown_sec,
                        "sensitivity": self.viral_engine.sensitivity,
                        "auto_clip": bool(self.settings.get("auto_clip", True)),
                        "manual_approval": bool(self.settings.get("manual_approval", False)),
                    },
                    "detected_moments": list(reversed(self.detected_moments[-25:])),
                    "pending_approvals": [p for p in reversed(self.pending_approvals) if p.get("status") == "pending"],
                    "checks": self.checks,
                    "skipped": self.skipped,
                    "tokens": self.tokens,
                    "words": len(self.words),
                    "rejected": self.rejected,
                    "verdicts": list(reversed(self.verdicts)),
                    "every_min": int(self.settings.get("ai_every_min", 3)) or 3,
                    "auto": {"spikes": self.spikes, "base": self.base, "waiting": bool(self.spike or self.said)},
                    "pending_clips": len(self.manual),
                },
            )
        jobs.save(self.job)

    def _analyze(self):
        cid = self.job["creator"]
        started = False
        while True:
            busy = False
            try:
                if not self.user_stopped and not self.ended:      # a recording cannot run for ever
                    if time.time() - self.started_at > config.MAX_LIVE_HOURS * 3600:
                        self.stop(f"Stopped after {config.MAX_LIVE_HOURS:g} hours, the longest one recording can run")
                self._scan_segments()
                busy = self._transcribe_step()
                if not started and self.samples:
                    started = True
                    self.job.update(status="live", stage="Recording")
                    creators.log(cid, f"Recording started: {self.job['title']}")
                self._manual_step()
                every = int(self.settings.get("ai_every_min", 3)) or 3
                self._watch_spikes()
                sp, said = self.spike, self.said
                cur_sec = max(self.t_done, self.samples / SR)
                if self.user_stopped:
                    pass
                elif said and (cur_sec >= said.get("end", said["t"] + self.SAID_AFTER) or (self.ended and not busy)):
                    self.said = None
                    self._ai_check((said.get("start", said["t"] - self.SAID_BEFORE), said.get("end", said["t"] + self.SAID_AFTER)), said["why"])
                elif sp and (cur_sec >= sp.get("end", sp["t"] + self.AUTO_AFTER) or (self.ended and not busy)):
                    self.spike = None
                    s_win = sp.get("start", sp["t"] - float(self.settings.get("pre_roll_sec", self.AUTO_BEFORE)))
                    e_win = sp.get("end", sp["t"] + float(self.settings.get("post_roll_sec", self.AUTO_AFTER)))
                    self._ai_check((s_win, e_win), "Viral: " + sp["why"], viral_meta=sp)
                elif not self._initial_check_done and len(self.job["clips"]) == 0 and cur_sec >= max(30.0, float(self.settings.get("min_len", 15)) + 15.0) and self.segments:
                    self._ai_check((0.0, cur_sec), "Initial stream check")
                elif self.t_done - self.last_check >= every * 60 or (self.ended and not busy and self.t_done > self.last_check):
                    self._ai_check()
                self._publish()
            except Exception as e:  # noqa: BLE001
                self.job["ai"]["last_error"] = str(e)[-200:]
            if (self.user_stopped and not self.manual) or (self.ended and not busy and len(self.pending) < SR * 2 and not self.manual):
                break
            if not busy:
                time.sleep(2)
        self.stop_flag.set()
        if self.adapter is not None:
            self.adapter.disconnect()
        self._scan_segments()
        if (
            not self.job.get("error")
            and len(self.job["clips"]) == 0
            and not self.manual
            and self.segments
            and (self.segments[-1][2] - self.segments[0][1]) >= max(8.0, float(self.settings.get("min_len", 15)) * 0.6)
        ):
            try:
                self._ai_check(
                    window=(self.segments[0][1], self.segments[-1][2]),
                    why="Stream highlight",
                    viral_meta={"score": max(72, int(self.settings.get("min_score", 65))), "why": "Best moment from recorded stream"},
                )
            except Exception as e:  # noqa: BLE001
                self.job["ai"]["last_error"] = str(e)[-200:]
        failed = bool(self.job.get("error")) and self.samples < SR * 5
        self.job.update(status="error" if failed else "done", stage="Could not record" if failed else self.end_note)
        jobs.save(self.job)
        creators.log(cid, f"Recording finished ({self.end_note.lower()}): {len(self.job['clips'])} clips, "
                          f"{round(self.samples / SR / 60)} min" if not failed else f"Recording failed: {self.job['error']}")

        if failed:
            alerts.notify(f"recording failed for {self.job['source']['name']}: {self.job['error']}")
        else:
            alerts.notify(f"recording finished for {self.job['source']['name']}: {len(self.job['clips'])} clips, "
                          f"{round(self.samples / SR / 60)} min.")

        def cleanup():  # runs after any queued renders, then frees the buffer
            for f in self.buf.glob("*"):
                f.unlink(missing_ok=True)
            self.finished = True
            SESSIONS.pop(self.job["id"], None)

        jobs.submit_live(cleanup)


def recover(job):
    """After a restart: finish the clips of an interrupted recording from the stream pieces still on disk."""
    c = creators.get(job["creator"]) or dict(creators.DEFAULTS)
    s = Session(job, c)
    s.ended, s.end_note = True, "Recording stopped when the studio closed"
    s.part = len(list(s.buf.glob("p*.csv")))
    s.seg_clock = 0.0
    s.csv_seen = {}
    with s.lock:
        for i in range(1, s.part + 1):
            f = s.buf / f"p{i:02d}.csv"
            rows = [r.split(",") for r in f.read_text(encoding="utf-8", errors="ignore").splitlines() if r.count(",") >= 2]
            for name, a, b in (r[:3] for r in rows):
                try:
                    dur = max(0.0, float(b) - float(a))
                except ValueError:
                    continue
                s.segments.append((s.buf / name, s.seg_clock, s.seg_clock + dur))
                s.seg_clock += dur
            s.csv_seen[i] = len(rows)
    try:
        s.words = json.loads((s.dir / "words.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        s.words = []
    s.n = len(job["clips"])
    todo = [k for k in job["clips"] if k.get("status") == "queued"]
    for k in todo:
        jobs.submit_live(lambda k=k: s._render(k))

    def cleanup():
        for f in s.buf.glob("*"):
            f.unlink(missing_ok=True)
        job.pop("recover", None)
        jobs.save(job)
        creators.log(job["creator"], f"Finished {len(todo)} clip(s) from a recording interrupted by a restart")

    jobs.submit_live(cleanup)


def start_session(creator: dict, url=None) -> dict:
    url = normalize(url or creator.get("channel") or "")
    if not re.match(r"^https?://[^/]+\.[^/]+/", url + "/"):
        raise ValueError("Add this creator's channel link first.")
    if active_for(creator["id"]):
        raise ValueError("This creator is already being recorded.")
    if not media.site_allowed(url):
        raise ValueError("Channel links must be from YouTube, Twitch or Kick.")
    if media.is_youtube(url) and media.youtube_paused():
        raise ValueError(media.YT_BLOCKED.replace("download videos", "record YouTube streams")
                         .replace("Upload the video file instead, or use a Twitch or Kick link. YouTube links",
                                  "Twitch and Kick streams still work. YouTube"))
    if len(active_mine()) >= config.MAX_LIVE_PER_USER:
        raise ValueError(f"You're already recording {len(active_mine())} streams, the most this server allows at once. Stop one first.")
    if len(active()) >= config.MAX_LIVE_TOTAL or (active() and load.busy()):
        raise ValueError("NovaCut is recording as many streams as it can right now. Try again in a little while.")
    from . import tools
    if tools.missing():
        raise ValueError(tools.HELP)
    job = jobs.create({"type": "live", "url": url, "name": f"{creator['name']} live"}, {}, enqueue=False,
                      creator=creator["id"], kind="live")
    job.update(status="live", stage="Connecting to the stream")
    jobs.save(job)
    sess = Session(job, creator)
    sess._publish()
    sess.start()
    return job


def start_monitor_url(url: str, options: dict = None) -> dict:
    """Start autonomous live monitoring directly from any Twitch, Kick, or YouTube Live URL.
    Finds an existing creator workspace for that channel or creates one automatically."""
    from . import profiles
    val = platforms.validate_live_url(url)
    norm_url = val["url"]
    ch_name = val["channel"].lstrip("@").split("/")[-1] or val["platform"]
    existing = next(
        (c for c in creators.all_creators()
         if norm_url.lower().rstrip("/") in [u.lower().rstrip("/") for u in [c.get("channel", "")] + list((c.get("channels") or {}).values())]),
        None,
    )
    if existing:
        if options:
            existing = creators.update(existing["id"], options)
        profiles.refresh(existing["id"])
        s = active_for(existing["id"])
        if s:
            if options:
                s.update_config(existing)
            return s.job
        return start_session(existing, norm_url)
    creator_data = {"name": ch_name, "channel": norm_url, **(options or {})}
    created = creators.create(creator_data)
    profiles.refresh(created["id"], force=True)
    return start_session(created, norm_url)


def _watch_user(uid):
    """Start recording any of this user's auto-record creators who have gone live."""
    from . import profiles
    with ctx.as_user(uid):
        try:
            todo = [c for c in creators.all_creators() if c["auto_watch"] and c["channel"]]
            if not todo:
                return
        except Exception:  # noqa: BLE001
            return
        for c in todo:
            _watch_one(c)


def check_now(cid):
    """Auto-record was just turned on: look now, on a thread, instead of waiting for the next round."""
    uid = ctx.uid()

    def run():
        from . import profiles
        with ctx.as_user(uid):
            c = creators.get(cid)
            if not c or not c["auto_watch"] or not c["channel"]:
                return
            for url in c.get("channels", {}).values():
                profiles._live.pop(url.rstrip("/").lower(), None)     # a fresh answer, not one from two minutes ago
            _watch_one(c)
            profiles.refresh(cid, force=True)
    threading.Thread(target=run, daemon=True).start()


def _watch_one(c):
    """Start recording this auto-record creator if they are live on any of their channels."""
    from . import profiles
    try:
        if active_for(c["id"]) or len(active_mine()) >= config.MAX_LIVE_PER_USER or len(active()) >= config.MAX_LIVE_TOTAL or (active() and load.busy()):
            return
        # the chosen channel first, then their other platforms
        urls = [c["channel"]] + [u for u in c.get("channels", {}).values() if u != c["channel"]]
        for url in urls:
            # one shared "are they live?" answer per channel, however many people follow it
            if not profiles.live_shared(profiles.platform_of(url), url).get("live"):
                continue
            if media.is_youtube(url):
                # check YouTube will hand over the stream before making a recording that could only fail
                if media.youtube_paused():
                    continue
                try:
                    resolve(normalize(url))
                except NotLive:
                    continue
            start_session(c, url)
            where = creators.platform_key(url).title()
            creators.log(c["id"], f"Went live on {where}: recording started automatically")
            alerts.notify(f"{c['name']} went live on {where}, recording started.")
            break
    except NotLive:
        pass
    except Exception as e:  # noqa: BLE001
        creators.log(c["id"], f"Auto-record check failed: {str(e)[-150:]}")


def watch_loop():
    """Every two minutes, start recording any creator set to auto-record who has gone live. Covers every user,
    several at a time, so the round stays short however many accounts there are."""
    from concurrent.futures import ThreadPoolExecutor
    time.sleep(20)
    pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="watch")
    while True:
        t0 = time.time()
        try:
            list(pool.map(_watch_user, auth.user_ids()))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(max(15, 120 - (time.time() - t0)))


def _resume_recordings(paused):
    """Recordings that were running when the server stopped: start recording the same channel again, if the
    stream is still on. Tried for a few minutes, since the server may come back a little before the network."""
    time.sleep(10)
    for _ in range(6):
        left = []
        for uid, cid, url in paused:
            with ctx.as_user(uid):
                try:
                    c = creators.get(cid)
                    if not c or active_for(cid):
                        continue
                    resolve(normalize(url))      # raises NotLive if the stream has ended: no empty recording is made
                    start_session(c, url)
                    creators.log(cid, "Recording picked up again after a server update")
                except NotLive:
                    continue                  # the stream ended while the server was away
                except Exception:  # noqa: BLE001 - try again shortly
                    left.append((uid, cid, url))
        if not left:
            return
        paused = left
        time.sleep(30)


def start():
    paused = []
    for j in jobs.all_jobs():
        if j.get("recover"):
            with ctx.as_user(j["user"]):
                recover(j)
        if j.get("was_live") and j.get("creator"):
            paused.append((j["user"], j["creator"], j["source"]["url"]))
            j.pop("was_live", None)
            jobs.save(j)
    if paused:
        threading.Thread(target=_resume_recordings, args=(paused,), daemon=True).start()
    threading.Thread(target=watch_loop, daemon=True).start()
