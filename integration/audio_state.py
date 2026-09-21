import threading
import time

_lock = threading.Lock()
_tts_active = False
_ignore_until = 0.0


def begin_tts():
    global _tts_active
    with _lock:
        _tts_active = True


def end_tts(grace_seconds=1.2):
    global _tts_active, _ignore_until
    with _lock:
        _tts_active = False
        _ignore_until = max(
            _ignore_until,
            time.monotonic() + grace_seconds,
        )


def should_ignore_input():
    with _lock:
        return (
            _tts_active
            or time.monotonic() < _ignore_until
        )
