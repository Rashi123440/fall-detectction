"""Local audio alert module for fall detection warning.
Plays an alert sound locally in an asynchronous background thread without blocking video inference.
Supports a repeating alarm mode that continues until explicitly stopped.
"""

import threading
import time
import sys
import math
import struct
import wave
import tempfile
from pathlib import Path
from src.config import ALERT_FREQUENCY_HZ, ALERT_DURATION_MS, ALERT_COOLDOWN_SECONDS


class AudioAlertManager:
    """Manages sound alerts with rate limiting, repeating alarm, and thread isolation.

    Two operating modes:
      - Single beep   : trigger_alert()   – fires one beep then respects cooldown.
      - Repeating loop: start_alarm() / stop_alarm() – loops until stop_alarm() is called.
    """

    def __init__(
        self,
        frequency: int = ALERT_FREQUENCY_HZ,
        duration_ms: int = ALERT_DURATION_MS,
        cooldown: float = ALERT_COOLDOWN_SECONDS,
        repeat_interval: float = 1.5,
    ):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.cooldown = cooldown
        self.repeat_interval = repeat_interval  # seconds between repeating beeps
        self.last_played_time = 0.0
        self._lock = threading.Lock()
        self._fallback_wav_path = None

        # Repeating alarm state
        self._alarm_active = False
        self._alarm_thread: threading.Thread | None = None

    # ── audio helpers ──────────────────────────────────────────────

    def _generate_fallback_wav(self) -> Path:
        """Generates a small in-memory sine wave audio file for cross-platform playback."""
        if self._fallback_wav_path and Path(self._fallback_wav_path).exists():
            return Path(self._fallback_wav_path)

        sample_rate = 22050
        duration_s = self.duration_ms / 1000.0
        num_samples = int(sample_rate * duration_s)

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._fallback_wav_path = temp_file.name

        with wave.open(temp_file.name, "w") as wav_file:
            wav_file.setnchannels(1)   # Mono
            wav_file.setsampwidth(2)   # 16-bit
            wav_file.setframerate(sample_rate)

            data = bytearray()
            for i in range(num_samples):
                t = i / sample_rate
                envelope = 1.0
                if i < 400:
                    envelope = i / 400.0
                elif i > num_samples - 400:
                    envelope = (num_samples - i) / 400.0
                sample = int(32767.0 * 0.7 * envelope * math.sin(2.0 * math.pi * self.frequency * t))
                data.extend(struct.pack("<h", sample))

            wav_file.writeframes(data)
        return Path(self._fallback_wav_path)

    def _play_beep_once(self):
        """Play a single beep. Errors are logged but do not crash the detector."""
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(int(self.frequency), int(self.duration_ms))
            else:
                wav_file = self._generate_fallback_wav()
                import os
                os.system(f"aplay -q '{wav_file}' 2>/dev/null || afplay '{wav_file}' 2>/dev/null")
        except Exception as e:
            print(f"[AudioAlertManager] Warning: Audio playback error: {e}")

    # ── single-shot API (existing behaviour) ──────────────────────

    def _play_sound_worker(self):
        """Worker for one-shot beep in a daemon thread."""
        self._play_beep_once()

    def trigger_alert(self, force: bool = False) -> bool:
        """Triggers one local beep if cooldown has expired.

        Returns:
            bool: True if alert was fired, False if suppressed by cooldown.
        """
        now = time.time()
        with self._lock:
            if not force and (now - self.last_played_time < self.cooldown):
                return False
            self.last_played_time = now

        thread = threading.Thread(target=self._play_sound_worker, daemon=True)
        thread.start()
        return True

    # ── repeating alarm API (new) ──────────────────────────────────

    def _alarm_loop(self):
        """Daemon thread: plays a beep, waits repeat_interval, then repeats until stopped."""
        while self._alarm_active:
            self._play_beep_once()
            # Wait in small slices so stop_alarm() is responsive
            deadline = time.monotonic() + self.repeat_interval
            while self._alarm_active and time.monotonic() < deadline:
                time.sleep(0.05)

    def start_alarm(self) -> bool:
        """Start a repeating alarm loop in a background daemon thread.

        Idempotent: if already running, does nothing and returns False.

        Returns:
            bool: True if alarm was started, False if already active.
        """
        with self._lock:
            if self._alarm_active:
                return False
            self._alarm_active = True

        self._alarm_thread = threading.Thread(target=self._alarm_loop, daemon=True)
        self._alarm_thread.start()
        return True

    def stop_alarm(self) -> bool:
        """Stop the repeating alarm loop.

        Idempotent: safe to call when no alarm is running.

        Returns:
            bool: True if alarm was running and is now stopped, False if it was already stopped.
        """
        with self._lock:
            was_active = self._alarm_active
            self._alarm_active = False
        return was_active

    @property
    def is_alarm_active(self) -> bool:
        """True if the repeating alarm loop is currently running."""
        return self._alarm_active


# Global singleton instance for easy import across modules
alert_manager = AudioAlertManager()
