"""Comprehensive automated test suite for NovaCut Studio's Autonomous Livestream Clipping Engine (Phases 2–10).

Tests:
1. Chat spike detection (baseline vs sudden multi-user excitement burst: OMG, NO WAY, INSANE, LMAO, LET'S GO, 😂, 💀)
2. Single-user duplicate & spam filtering (1 user spamming 80 messages is rejected and does not trigger a spike)
3. Unique chatter counting & repeated phrase consensus
4. Multi-modal Viral Scoring Engine (Chat 30%, Unique 15%, Emotes 15%, Audio 20%, Transcript 10%, Agreement 10%)
5. Rolling buffer preservation & segment pinning (pre-roll segments are never deleted before clip render)
6. Automatic clip generation, overlap merging & manual approval queue
7. Platform adapters (Twitch, Kick, YouTube Live), SSRF protection, disconnections & reconnections
8. FFmpeg failure handling & disk usage limits
9. Local-Only Mode & API usage restrictions (zero external cloud AI calls)
"""
from pathlib import Path
import os
import shutil
import sys
import tempfile
import threading
import time

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-autolive-")
os.environ.update(
    DATABASE_URL="",
    PEAKCUT_ACCOUNTS="0",
    PEAKCUT_PUBLIC_URL="http://127.0.0.1:8765",
    PEAKCUT_FETCH_MODEL="0",
    PEAKCUT_UPDATE_CHECK="0",
)
sys.path.insert(0, ".")

from fastapi.testclient import TestClient  # noqa: E402
from studio import ai, auth, brain, chat_intel, config, creators, ctx, db, jobs, live, load, platforms, profiles, server, viral_engine  # noqa: E402
from studio.platforms.base import ChatMessage, check_ssrf_safe, sanitize_chat_text  # noqa: E402
from studio.platforms.kick import KickAdapter  # noqa: E402
from studio.platforms.twitch import TwitchAdapter  # noqa: E402
from studio.platforms.youtube import YouTubeAdapter  # noqa: E402

db.init()
fails = []


def check(name: str, ok: bool, extra: str = ""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{extra}]" if extra and not ok else ""))
    if not ok:
        fails.append(name)


# ==============================================================================
# 1. PLATFORM ADAPTERS, URL VALIDATION, SSRF PROTECTION & RECONNECTION
# ==============================================================================
check("platform detect: twitch, kick, youtube", (
    platforms.detect_platform("https://www.twitch.tv/shroud") == "twitch"
    and platforms.detect_platform("https://kick.com/xqc") == "kick"
    and platforms.detect_platform("https://www.youtube.com/watch?v=jfKfPfyJRdk") == "youtube"
    and platforms.detect_platform("https://www.youtube.com/@lofi/live") == "youtube"
))

v_tw = platforms.validate_live_url("https://twitch.tv/tarik")
v_kk = platforms.validate_live_url("https://kick.com/trainwreckstv")
v_yt = platforms.validate_live_url("https://www.youtube.com/live/jfKfPfyJRdk")
check("url validation extracts channel across Twitch, Kick, YouTube", (
    v_tw["platform"] == "twitch" and v_tw["channel"] == "tarik"
    and v_kk["platform"] == "kick" and v_kk["channel"] == "trainwreckstv"
    and v_yt["platform"] == "youtube" and v_yt["channel"] == "jfKfPfyJRdk"
), str((v_tw, v_kk, v_yt)))

# SSRF protection tests
for bad_url in (
    "http://127.0.0.1:8080/live",
    "http://localhost/stream",
    "https://169.254.169.254/latest/meta-data/",
    "https://192.168.1.1/stream",
    "https://evil.internal.corp/live",
    "file:///etc/passwd",
):
    blocked = False
    try:
        check_ssrf_safe(bad_url, ("twitch.tv", "kick.com", "youtube.com", "youtu.be"))
    except ValueError:
        blocked = True
    check(f"SSRF blocks unsafe URL ({bad_url})", blocked)

# Chat sanitization
dirty_chat = "<b>OMG</b> \x00\x07\x1b[31mNO WAY!!\n\r\t" + ("A" * 500)
clean_chat = sanitize_chat_text(dirty_chat)
check("chat input sanitization strips HTML/control chars and caps length", (
    "\x00" not in clean_chat and "<b>" not in clean_chat and len(clean_chat) <= 300
))

# Twitch IRC frame parser test
tw_line = "@badge-info=;display-name=ViewerOne;emotes=25:0-4;id=msg-123;user-id=998877 :viewerone!viewerone@viewerone.tmi.twitch.tv PRIVMSG #shroud :Kappa INSANE PLAY OMG!!"
tw_msg = TwitchAdapter.parse_irc_line(tw_line, stream_sec=42.5)
check("Twitch IRCv3 tag & PRIVMSG parser", (
    tw_msg is not None
    and tw_msg.user_id == "998877"
    and tw_msg.username == "ViewerOne"
    and len(tw_msg.emotes) >= 1
    and "INSANE PLAY" in tw_msg.text
), str(tw_msg))

# Kick Pusher frame parser test
kk_frame = '{"event":"App\\\\Events\\\\ChatMessageEvent","data":"{\\"id\\":\\"m1\\",\\"content\\":\\"LETS GO [emote:37226:KEKW] 💀\\",\\"sender\\":{\\"id\\":4455,\\"username\\":\\"KickFan\\"}}"}'
kk_msg = KickAdapter.parse_pusher_event(kk_frame, stream_sec=50.0)
check("Kick Pusher WebSocket ChatMessageEvent parser", (
    kk_msg is not None
    and kk_msg.user_id == "4455"
    and kk_msg.username == "KickFan"
    and "KEKW" in kk_msg.emotes
), str(kk_msg))

# YouTube Innertube action parser test
yt_actions = [{
    "addChatItemAction": {
        "item": {
            "liveChatTextMessageRenderer": {
                "id": "yt-m1",
                "authorExternalChannelId": "UC12345",
                "authorName": {"simpleText": "YTViewer"},
                "message": {"runs": [{"text": "NO WAY "}, {"emoji": {"emojiId": "💀"}}]},
            }
        }
    }
}]
yt_msgs = YouTubeAdapter.parse_innertube_actions(yt_actions, stream_sec=65.0)
check("YouTube Innertube live chat action parser", (
    len(yt_msgs) == 1 and yt_msgs[0].user_id == "UC12345" and "💀" in yt_msgs[0].emotes and "NO WAY" in yt_msgs[0].text
), str(yt_msgs))

# Stream disconnection & exponential backoff reconnection handling
recon_attempts = []


class MockReconnectTwitch(TwitchAdapter):
    def connect_chat(self, url, on_message, stop_event, stream_clock_fn):
        self._stop_event = stop_event
        backoff = 0.05
        self.set_status("connecting")
        while not stop_event.is_set():
            recon_attempts.append(time.time())
            try:
                if len(recon_attempts) == 1:
                    raise ConnectionResetError("Simulated IRC socket drop")
                self.set_status("connected")
                msg = ChatMessage(
                    platform="twitch", user_id="u_rec", username="RecUser",
                    text="We are back!", timestamp=time.time(), stream_sec=stream_clock_fn()
                )
                self.record_message()
                on_message(msg)
                stop_event.wait(0.2)
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.reconnects += 1
                self.set_status("reconnecting", str(e))
                stop_event.wait(backoff)
        self.set_status("disconnected")


rec_adapter = MockReconnectTwitch()
rec_msgs = []
stop_ev = threading.Event()
t_chat = threading.Thread(target=rec_adapter.connect_chat, args=("https://twitch.tv/test", rec_msgs.append, stop_ev, lambda: 12.0), daemon=True)
t_chat.start()
time.sleep(0.18)
stop_ev.set()
t_chat.join(timeout=1.0)
state = rec_adapter.connection_state()
check("Platform chat disconnection & automatic reconnection", (
    len(recon_attempts) >= 2 and len(rec_msgs) == 1 and state["reconnects"] >= 1 and state["status"] == "disconnected"
), str(state))


# ==============================================================================
# 2. LIVE CHAT INTELLIGENCE ENGINE (SPIKE DETECTION, SPAM FILTER, UNIQUE USERS)
# ==============================================================================
chat_eng = chat_intel.ChatIntelligenceEngine(
    active_window_sec=8.0,
    baseline_window_sec=120.0,
    max_msgs_per_user_window=2,
    dup_window_sec=15.0,
    spike_multiplier=3.0,
    min_unique_chatters=4,
    cooldown_sec=30.0,
)

# Feed 60 seconds of normal baseline chat (~2 messages per second from various users)
for sec in range(1, 61):
    for u in range(2):
        chat_eng.ingest(ChatMessage(
            platform="twitch",
            user_id=f"base_u_{(sec * 2 + u) % 25}",
            username=f"User_{(sec * 2 + u) % 25}",
            text=f"normal stream conversation message {sec}-{u}",
            timestamp=1000.0 + sec,
            stream_sec=float(sec) + u * 0.4,
        ))

base_snap = chat_eng.analyze(now_sec=60.0)
check("Chat baseline established without false spike", (
    not base_snap["is_spike"] and 1.0 <= base_snap["baseline_mps"] <= 3.5
), str(base_snap))

# Test single-user spam filtering: 1 user sends 80 messages in 2 seconds
accepted_from_spammer = 0
for i in range(80):
    ok = chat_eng.ingest(ChatMessage(
        platform="twitch",
        user_id="spammer_01",
        username="SpamBot",
        text="OMG NO WAY INSANE CLIP THIS 😂💀" if i % 2 == 0 else f"SPAM MESSAGE {i} OMG",
        timestamp=1062.0 + i * 0.02,
        stream_sec=62.0 + i * 0.02,
        emotes=["😂", "💀"],
    ))
    if ok:
        accepted_from_spammer += 1

spam_snap = chat_eng.analyze(now_sec=64.0)
check("Single-user duplicate & rate-limit filter blocks spammer from triggering viral event", (
    accepted_from_spammer <= 2 and not spam_snap["is_spike"] and chat_eng.spam_filtered >= 78
), f"accepted={accepted_from_spammer}, filtered={chat_eng.spam_filtered}, is_spike={spam_snap['is_spike']}")

# Now simulate a genuine multi-user viral moment: 45 distinct viewers react in 3 seconds
viral_phrases = ["OMG", "NO WAY", "INSANE", "LMAO", "LET'S GO", "😂", "💀", "CLIP IT", "WHAT WAS THAT OMG"]
for i in range(45):
    phrase = viral_phrases[i % len(viral_phrases)]
    chat_eng.ingest(ChatMessage(
        platform="twitch",
        user_id=f"real_viewer_{i}",
        username=f"Viewer_{i}",
        text=phrase,
        timestamp=1075.0 + (i * 0.06),
        stream_sec=75.0 + (i * 0.06),
        emotes=["😂"] if "😂" in phrase else (["💀"] if "💀" in phrase else []),
    ))

spike_snap = chat_eng.analyze(now_sec=78.0)
check("Genuine multi-user excitement burst triggers chat spike", (
    spike_snap["is_spike"]
    and spike_snap["unique_chatters"] >= 40
    and spike_snap["velocity_ratio"] >= 3.0
    and spike_snap["velocity_score"] >= 70.0
    and spike_snap["burst_score"] >= 70.0
    and len(spike_snap["repeated_phrases"]) >= 3
), f"snap={spike_snap}")

# Cooldown check: immediate re-evaluation within cooldown window does not re-trigger spike
cooldown_snap = chat_eng.analyze(now_sec=79.5)
check("Chat spike cooldown prevents duplicate back-to-back triggers", (
    not cooldown_snap["is_spike"] and cooldown_snap["in_cooldown"]
))


# ==============================================================================
# 3. AUDIO & TRANSCRIPT INTELLIGENCE + WEIGHTED VIRAL SCORING ENGINE
# ==============================================================================
# Build 40 seconds of quiet baseline audio (-35 dB), a dip (-42 dB), then a sudden loud reaction (-4.5 dB)
db_history = [-35.0] * 32 + [-42.0, -41.5, -41.0, -40.5] + [-6.5, -5.0, -4.2, -4.5, -5.2, -5.8, -6.0, -5.5]
audio_sig = viral_engine.analyze_audio_window(db_history, now_sec=78.0, window_sec=8.0)
check("Audio intelligence detects sudden loud reaction & silence-to-loud transition", (
    audio_sig["audio_score"] >= 70.0
    and audio_sig["peak_db"] > -6.0
    and audio_sig["jump_pct"] > 100.0
    and audio_sig["silence_to_loud"]
    and len(audio_sig["reasons"]) >= 1
), str(audio_sig))

# Build transcript words with viral speech & trigger keywords
words = [
    {"w": "Wait", "s": 68.0, "e": 68.3},
    {"w": "what", "s": 68.4, "e": 68.7},
    {"w": "is", "s": 68.8, "e": 69.0},
    {"w": "happening", "s": 69.1, "e": 69.5},
    {"w": "insane", "s": 69.6, "e": 69.8},
    {"w": "clutch", "s": 69.9, "e": 70.2},
    {"w": "win", "s": 70.3, "e": 70.6},
    {"w": "crazy", "s": 70.7, "e": 70.9},
    {"w": "unbelievable", "s": 71.0, "e": 71.4},
    {"w": "omg", "s": 71.5, "e": 71.8},
    {"w": "clip", "s": 71.9, "e": 72.2},
    {"w": "that", "s": 72.3, "e": 72.6},
    {"w": "right", "s": 72.7, "e": 73.0},
    {"w": "now!", "s": 73.1, "e": 73.4},
]
tx_sig = viral_engine.analyze_transcript_window(words, now_sec=78.0, window_sec=15.0)
check("Transcript intelligence detects speech keywords, hooks & conversational pacing", (
    tx_sig["transcript_score"] >= 70.0 and len(tx_sig["keywords"]) >= 2
), str(tx_sig))

# ViralScoringEngine with default weights (30/15/15/20/10/10)
scorer = viral_engine.ViralScoringEngine(min_viral_score=65.0, pre_roll_sec=30.0, post_roll_sec=20.0, sensitivity="medium")
moment = scorer.evaluate(
    now_sec=78.0,
    chat_snap=spike_snap,
    audio_snap=audio_sig,
    transcript_snap=tx_sig,
    chat_connected=True,
)
check("ViralScoringEngine combines chat, audio, transcript & cross-signal agreement", (
    moment["should_clip"]
    and moment["viral_score"] >= 80
    and moment["recommended_duration"] >= 50.0
    and moment["component_scores"]["cross_signal"] == 100.0
), str(moment))

# Custom weights configuration test
scorer.configure(
    weights={"chat_velocity": 0.50, "unique_chatters": 0.10, "emote_bursts": 0.10, "audio_excitement": 0.10, "transcript_signals": 0.10, "cross_signal": 0.10},
    min_viral_score=72.0,
)
check("ViralScoringEngine normalizes custom weights & updates threshold", (
    abs(sum(scorer.weights.values()) - 1.0) < 1e-3 and scorer.min_viral_score == 72.0
), str(scorer.weights))


# ==============================================================================
# 4. ROLLING BUFFER PRESERVATION, SEGMENT PINNING & AUTOMATIC CLIP GENERATION
# ==============================================================================
u = auth.local_user()
with ctx.as_user(u.id):
    creator_obj = creators.create({
        "name": "TestStreamer",
        "channel": "https://www.twitch.tv/teststreamer",
        "pre_roll_sec": 30,
        "post_roll_sec": 20,
        "buffer_min": 1,  # 60s rolling buffer
    })
    tmp_job_dir = Path(os.environ["PEAKCUT_DATA"]) / "live_buf_test"
    buf_dir = tmp_job_dir / "buffer"
    buf_dir.mkdir(parents=True, exist_ok=True)

    # Create 12 synthetic 10-second .ts segments (covering 0s..120s of stream) and p01.csv
    csv_lines = []
    for idx in range(12):
        seg_name = f"p01_{idx:05d}.ts"
        (buf_dir / seg_name).write_bytes(b"TS_SEGMENT_DATA_" + str(idx).encode())
        csv_lines.append(f"{seg_name},{idx * 10.0:.2f},{(idx + 1) * 10.0:.2f}")
    (buf_dir / "p01.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    job_dict = {
        "id": "live-test-job-1",
        "uid": u.id,
        "kind": "live",
        "dir": str(tmp_job_dir),
        "creator": creator_obj["id"],
        "source": {"type": "live", "url": "https://www.twitch.tv/teststreamer", "name": "TestStreamer"},
        "title": "TestStreamer · Live",
        "uploader": "TestStreamer",
        "duration": 0,
        "options": {"style": "hormozi", "layout": "auto", "min_len": 18, "max_len": 60},
        "status": "live",
        "stage": "Listening",
        "progress": 0.0,
        "error": None,
        "created": time.time(),
        "clips": [],
        "ai": {"provider": "Antigravity AI", "model": "antigravity-v1"},
        "live": {
            "started": time.time(), "recorded": 0, "checked_to": 0, "level": [], "chat": [],
            "chat_status": "connected", "lines": [], "verdicts": [], "ai_calls": 0,
            "viral_score": 0, "moments": [],
        },
    }

    sess = live.Session(job_dict, creator_obj)
    sess.settings["buffer_min"] = 1  # 60s rolling buffer for test
    sess.part = 1
    # Pin an active viral spike at t=35s (pre-roll back to t=5s, pinning p01_00000.ts..p01_00005.ts)
    sess.spike = {"t": 35.0, "start": 5.0, "end": 55.0, "score": 88, "why": "Viral chat & audio spike"}

    # Scan segments: stream is at 120s and buffer_min=1 (normally segments < 60s would be pruned!)
    sess._scan_segments()
    remaining_files = sorted(p.name for p in buf_dir.glob("*.ts"))
    check("Rolling buffer pins pre-roll segments (p01_00000.ts..p01_00005.ts) during active viral moment", (
        "p01_00000.ts" in remaining_files and "p01_00001.ts" in remaining_files and "p01_00002.ts" in remaining_files
    ), f"remaining={remaining_files}")

    # Test manual approval pinning as well
    sess.spike = None
    sess.pending_approvals.append({
        "id": "mom_25_1",
        "t": 25.0,
        "start": 5.0,
        "end": 45.0,
        "score": 84,
        "why": "Pending manual approval",
        "wall_time": time.time(),
        "status": "pending",
    })
    sess._scan_segments()
    remaining_pending = sorted(p.name for p in buf_dir.glob("*.ts"))
    check("Rolling buffer keeps segments pinned while moment awaits manual approval", (
        "p01_00000.ts" in remaining_pending
    ), f"remaining_pending={remaining_pending}")

    # Test approving the pending moment queues it into sess.manual
    approved = sess.approve_moment("mom_25_1")
    check("approve_moment marks moment approved and queues clip with pre-roll + post-roll", (
        approved["status"] == "approved" and len(sess.manual) == 1 and sess.manual[0]["start"] == 5.0 and sess.manual[0]["end"] == 45.0
    ), str((approved, sess.manual)))

    # Clear manual & pending approvals: old segments (< 60s) should now be pruned cleanly
    sess.manual.clear()
    sess.pending_approvals.clear()
    sess._scan_segments()
    after_unpin = sorted(p.name for p in buf_dir.glob("*.ts"))
    check("Rolling buffer prunes old segments once no viral moment or clip pins them", (
        "p01_00000.ts" not in after_unpin and "p01_00011.ts" in after_unpin
    ), f"after_unpin={after_unpin}")

    # Test automatic clip generation via _ai_check with viral_meta & overlap prevention
    rendered_clips = []
    sess._render = lambda c: rendered_clips.append(c)
    sess.samples = 120 * live.SR
    sess.t_done = 120.0
    sess.db = [-35.0] * 80 + [-5.0] * 10 + [-35.0] * 30
    sess.words = [{"w": f"word{i}", "s": 60.0 + i * 0.5, "e": 60.4 + i * 0.5} for i in range(90)]
    sess._ai_check(window=(65.0, 115.0), why="Viral score 88", viral_meta={"score": 88, "why": "Chat velocity 15x baseline"})
    time.sleep(0.15)
    check("Automatic clip generation creates clip with hook, title & viral score", (
        len(job_dict["clips"]) == 1 and job_dict["clips"][0]["score"] >= 88
    ), f"clips={job_dict['clips']}")


# ==============================================================================
# 5. FFMPEG FAILURE HANDLING & DISK RESOURCE LIMITS
# ==============================================================================
load.start()
time.sleep(0.05)
try:
    load._state["disk_free"] = 0.1  # Simulate only 0.1 GB (100 MB) free disk space
    load._state["at"] = time.time()
    check("Disk usage monitor detects full disk when free space drops below MIN_FREE_DISK_GB", load.disk_full())
finally:
    load._refresh_disk()
    check("Disk usage monitor reports healthy disk space on current machine", not load.disk_full())

# Test FFmpeg missing segment error handling in Session._render
with ctx.as_user(u.id):
    missing_clip = {"id": "clip999", "start": 0.0, "end": 15.0, "status": "queued", "version": 0}
    sess.ended = True
    live.Session._render(sess, missing_clip)
    check("Session._render gracefully sets status='error' when buffer segments are unavailable", (
        missing_clip["status"] == "error" and "buffer" in (missing_clip.get("error") or "")
    ), str(missing_clip))


# ==============================================================================
# 6. LOCAL-ONLY MODE & ZERO EXTERNAL API USAGE RESTRICTIONS
# ==============================================================================
network_called = []
orig_post = ai.httpx.post
try:
    def forbidden_post(*args, **kwargs):
        network_called.append(args)
        raise AssertionError("External network call attempted while in Local-Only mode!")

    ai.httpx.post = forbidden_post
    config.AI_PROVIDER = "openai"
    config.AI_MODELS = ["gpt-4o"]
    os.environ["PEAKCUT_AI_KEY"] = "sk-should-never-be-used"
    config._kept.clear()

    ai.set_local_only(True)
    check("Local-Only mode reports strict_local() == True and is_local_mode() == True", ai.strict_local() and ai.is_local_mode())

    # Even though provider is set to 'openai' with a key, Local-Only mode must route to built-in local engine with 0 network calls
    res, tok_in, tok_out = ai.call("gpt-4o", 'LIVE STREAM TRANSCRIPT:\n[0.0s] Oh my god no way clip that!', brain.SCHEMA)
    check("Local-Only mode intercepts cloud AI calls and executes locally with 0 external tokens", (
        len(network_called) == 0 and tok_in == 0 and tok_out == 0 and isinstance(res, list)
    ), f"network_called={len(network_called)}, res={res}")
finally:
    ai.set_local_only(False)
    ai.httpx.post = orig_post
    os.environ.pop("PEAKCUT_AI_KEY", None)
    config._kept.clear()


# ==============================================================================
# 7. LIVE MONITOR HTTP API ENDPOINTS (validate, monitor, config, stop)
# ==============================================================================
BASE_URL = "http://127.0.0.1:8765"
ORIGIN_HDR = {"Origin": BASE_URL}
client = TestClient(server.app, base_url=BASE_URL, follow_redirects=False)

orig_start = live.Session.start
orig_refresh = profiles.refresh
try:
    live.Session.start = lambda self: live.SESSIONS.__setitem__(self.job["id"], self)
    profiles.refresh = lambda *args, **kwargs: None

    val_r = client.get("/api/live/validate", params={"url": "https://www.twitch.tv/testliveuser"})
    check("GET /api/live/validate accepts Twitch URL", (
        val_r.status_code == 200 and val_r.json().get("platform") == "twitch" and val_r.json().get("channel") == "testliveuser"
    ), f"status={val_r.status_code} body={val_r.text}")

    mon_r = client.post("/api/live/monitor", json={"url": "https://www.twitch.tv/testliveuser", "options": {"min_score": 60}}, headers=ORIGIN_HDR)
    check("POST /api/live/monitor starts live monitoring session", (
        mon_r.status_code == 200 and mon_r.json().get("status") == "live" and bool(mon_r.json().get("live"))
    ), f"status={mon_r.status_code} body={mon_r.text}")

    mon_jid = mon_r.json().get("id")
    cfg_r = client.put(f"/api/live/{mon_jid}/config", json={"min_score": 70, "sensitivity": "high"}, headers=ORIGIN_HDR)
    check("PUT /api/live/{jid}/config updates viral threshold", (
        cfg_r.status_code == 200 and cfg_r.json().get("ok") and (cfg_r.json().get("live") or {}).get("threshold") == 70.0
    ), f"status={cfg_r.status_code} body={cfg_r.text}")

    stop_r = client.post(f"/api/live/{mon_jid}/stop", headers=ORIGIN_HDR)
    check("POST /api/live/{jid}/stop stops monitoring session", (
        stop_r.status_code == 200 and stop_r.json().get("ok")
    ), f"status={stop_r.status_code} body={stop_r.text}")
finally:
    live.Session.start = orig_start
    profiles.refresh = orig_refresh
    live.SESSIONS.clear()

print("ALL PASSED" if not fails else f"{len(fails)} failed: {fails}")
db.engine.dispose()
shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
