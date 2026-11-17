"""How busy the server really is, measured every second: processor, memory and disk.

Nothing here counts people. New heavy work (another video, another recording) starts only while the machine has
room, waits in line for a moment when it is nearly full, and is let through the second it frees up again. That is
what keeps a rush from slowing everyone down or crashing the server."""
import os
import shutil
import sys
import threading
import time

from . import config

BUSY_CPU = float(config.env("PEAKCUT_BUSY_CPU", "82"))          # percent: above this, no extra videos are started
MIN_FREE_RAM_GB = float(config.env("PEAKCUT_MIN_FREE_RAM_GB", "1.5"))
MIN_FREE_DISK_GB = float(config.env("PEAKCUT_MIN_FREE_DISK_GB", "3"))

_state = {"cpu": 0.0, "ram_free": 99.0, "disk_free": 99.0, "at": 0.0}
_force = None            # tests only: pretend the server is busy (True) or free (False)
_started = False
_lock = threading.Lock()


def _cpu_times():
    """(busy, total) processor time so far, in any unit."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        n = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime          # noqa: E731
        total = n(kernel) + n(user)                                        # kernel time includes idle time
        return total - n(idle), total
    if sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            class HostCpuLoadInfo(ctypes.Structure):
                _fields_ = [("cpu_ticks", ctypes.c_uint * 4)]
            info = HostCpuLoadInfo()
            count = ctypes.c_uint(4)
            host = libc.mach_host_self()
            if libc.host_statistics(host, 3, ctypes.byref(info), ctypes.byref(count)) == 0:
                user, system, idle, nice = info.cpu_ticks
                return user + system + nice, user + system + idle + nice
        except Exception:
            pass
    if os.path.exists("/proc/stat"):
        with open("/proc/stat", encoding="ascii") as f:
            v = [int(x) for x in f.readline().split()[1:9]]
        return sum(v) - v[3] - v[4], sum(v)                                # everything except idle and waiting on disk
    return None


def _ram_free_gb() -> float:
    if os.name == "nt":
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")]
        m = M()
        m.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail / 1e9
    if sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            class VMStatistics64(ctypes.Structure):
                _fields_ = [
                    ('free_count', ctypes.c_uint32),
                    ('active_count', ctypes.c_uint32),
                    ('inactive_count', ctypes.c_uint32),
                    ('wire_count', ctypes.c_uint32),
                    ('zero_fill_count', ctypes.c_uint64),
                    ('reactivations', ctypes.c_uint64),
                    ('pageins', ctypes.c_uint64),
                    ('pageouts', ctypes.c_uint64),
                    ('faults', ctypes.c_uint64),
                    ('cow_faults', ctypes.c_uint64),
                    ('lookups', ctypes.c_uint64),
                    ('hits', ctypes.c_uint64),
                    ('purges', ctypes.c_uint64),
                    ('purgeable_count', ctypes.c_uint32),
                    ('speculative_count', ctypes.c_uint32),
                    ('decompressions', ctypes.c_uint64),
                    ('compressions', ctypes.c_uint64),
                    ('swapins', ctypes.c_uint64),
                    ('swapouts', ctypes.c_uint64),
                    ('compressor_page_count', ctypes.c_uint32),
                    ('throttled_count', ctypes.c_uint32),
                    ('external_page_count', ctypes.c_uint32),
                    ('internal_page_count', ctypes.c_uint32),
                    ('total_uncompressed_pages_in_compressor', ctypes.c_uint64),
                ]
            info = VMStatistics64()
            count = ctypes.c_uint(ctypes.sizeof(info) // 4)
            host = libc.mach_host_self()
            if libc.host_statistics64(host, 4, ctypes.byref(info), ctypes.byref(count)) == 0:
                page_size = ctypes.c_size_t()
                libc.host_page_size(host, ctypes.byref(page_size))
                ps = page_size.value or 4096
                return ((info.free_count + info.inactive_count + info.purgeable_count) * ps) / 1e9
        except Exception:
            pass
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    return 99.0


def _refresh_disk():
    try:
        _state["disk_free"] = shutil.disk_usage(config.DATA_DIR).free / 1e9
        _state["at"] = time.time()
    except Exception:
        pass


def _watch():
    _refresh_disk()
    last = None
    try:
        last = _cpu_times()
    except Exception:
        last = None
    while True:
        time.sleep(1)
        try:
            if last is not None:
                now = _cpu_times()
                if now is not None:
                    used, total = now[0] - last[0], now[1] - last[1]
                    last = now
                    if total > 0:
                        _state["cpu"] = 0.55 * _state["cpu"] + 0.45 * (100.0 * used / total)
            elif hasattr(os, "getloadavg"):
                cpus = max(1, os.cpu_count() or 4)
                _state["cpu"] = min(100.0, (os.getloadavg()[0] / cpus) * 100.0)
            _state["ram_free"] = _ram_free_gb()
            if time.time() - _state["at"] > 15:
                _refresh_disk()
        except Exception:  # noqa: BLE001
            pass


def start():
    global _started
    with _lock:
        if not _started:
            _started = True
            _refresh_disk()
            threading.Thread(target=_watch, daemon=True, name="load").start()


def busy() -> bool:
    """True while the machine is close to full: nothing extra should start right now."""
    if _force is not None:
        return _force
    start()
    return _state["cpu"] >= BUSY_CPU or _state["ram_free"] < MIN_FREE_RAM_GB


def disk_full() -> bool:
    start()
    if time.time() - _state["at"] > 15:
        _refresh_disk()
    return _state["disk_free"] < MIN_FREE_DISK_GB


def state() -> dict:
    start()
    return {"cpu": round(_state["cpu"]), "ram_free_gb": round(_state["ram_free"], 1), "disk_free_gb": round(_state["disk_free"], 1),
            "busy": busy(), "busy_cpu": BUSY_CPU}
