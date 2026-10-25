"""Local audio alert module for fall detection warning.
Plays an alert sound locally in an asynchronous background thread without blocking video inference.
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
    """Manages sound alerts with rate limiting and thread isolation."""

    def __init__(self, frequency: int = ALERT_FREQUENCY_HZ, duration_ms: int = ALERT_DURATION_MS, cooldown: float = ALERT_COOLDOWN_SECONDS):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.cooldown = cooldown
        self.last_played_time = 0.0
        self._lock = threading.Lock()
        self._fallback_wav_path = None

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
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # Synthesize beep tone with gentle envelope attack/decay
            data = bytearray()
            for i in range(num_samples):
                t = i / sample_rate
                # Envelope to prevent clicking
                envelope = 1.0
                if i < 400:
                    envelope = i / 400.0
                elif i > num_samples - 400:
                    envelope = (num_samples - i) / 400.0

                sample = int(32767.0 * 0.7 * envelope * math.sin(2.0 * math.pi * self.frequency * t))
                data.extend(struct.pack("<h", sample))

            wav_file.writeframes(data)
        return Path(self._fallback_wav_path)

    def _play_sound_worker(self):
        """Worker executing in a non-blocking daemon thread."""
        try:
            if sys.platform == "win32":
                import winsound
                # Windows built-in speaker beep
                winsound.Beep(int(self.frequency), int(self.duration_ms))
            else:
                # Fallback on macOS/Linux using system audio command if available
                wav_file = self._generate_fallback_wav()
                import os
                os.system(f"aplay -q '{wav_file}' 2>/dev/null || afplay '{wav_file}' 2>/dev/null")
        except Exception as e:
            # Sound failures must not crash real-time fall detection
            print(f"[AudioAlertManager] Warning: Audio playback notice: {e}")

    def trigger_alert(self, force: bool = False) -> bool:
        """Triggers local sound alert if cooldown has expired.

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


# Global singleton instance for easy import across modules
alert_manager = AudioAlertManager()
