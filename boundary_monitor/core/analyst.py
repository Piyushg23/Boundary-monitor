"""
analyst.py — Non-blocking Groq LLM analyst for tactical scene assessment.
"""

import json
import os
import queue
import threading
import time
from typing import List

from utils.config import CFG

try:
    from groq import Groq as GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[GROQ] 'groq' package not found. Run: pip install groq")

_PLACEHOLDERS = {"", "your_groq_api_key_here", "YOUR_GROQ_API_KEY_HERE"}


class GroqAnalyst:
    SYSTEM_PROMPT = (
        "You are a tactical AI analyst for a perimeter boundary monitoring system.\n"
        "You receive real-time track data (JSON) from a computer vision pipeline.\n"
        "Your job: concisely assess the threat level, identify the most dangerous targets,\n"
        "and recommend immediate operator actions.\n"
        "Rules:\n"
        "- Keep response under 220 words.\n"
        "- Start with THREAT LEVEL: [LOW/MEDIUM/HIGH/CRITICAL]\n"
        "- Then ASSESSMENT: 1-2 sentences.\n"
        "- Then ACTION: 1-2 bullet points of recommended operator steps.\n"
        "- Be direct, no preamble, no sign-off.\n"
    )

    def __init__(self, api_key: str):
        self._api_key = api_key.strip()
        self._client = None
        self._result_queue: queue.Queue = queue.Queue(maxsize=5)
        self._busy = False
        self._last_lines: List[str] = ["[GROQ] Standby -- press G to analyse scene"]
        self._last_trigger_time = 0.0
        self._init_client(self._api_key)

    def _init_client(self, key: str):
        """(Re-)initialise the Groq client with the given key."""
        if not GROQ_AVAILABLE:
            return
        key = key.strip()
        if key in _PLACEHOLDERS:
            print("[GROQ] No valid API key -- enter it in the launcher or set GROQ_API_KEY in .env")
            self._client = None
            return
        try:
            self._client = GroqClient(api_key=key)
            self._api_key = key
            print(f"[GROQ] Client initialised with key: {key[:8]}...{key[-4:]}")
        except Exception as e:
            print(f"[GROQ] Init error: {e}")
            self._client = None

    def _get_client(self):
        """Return existing client, or try to build one from current env if key was set late."""
        if self._client is not None:
            return self._client
        # Key might have been set in os.environ after init (e.g. typed in launcher)
        env_key = os.environ.get("GROQ_API_KEY", "").strip()
        if env_key not in _PLACEHOLDERS and env_key != self._api_key:
            self._init_client(env_key)
        return self._client

    def trigger(self, tracks: list, frame_idx: int, reason: str = "AUTO"):
        client = self._get_client()
        if client is None or self._busy:
            return
        now = time.time()
        if now - self._last_trigger_time < 2.0:
            return
        self._last_trigger_time = now
        self._busy = True
        confirmed = [t for t in tracks if t.confirmed]
        scene_data = {
            "frame":               frame_idx,
            "timestamp":           time.strftime("%H:%M:%S UTC"),
            "trigger_reason":      reason,
            "track_count":         len(confirmed),
            "tracks":              [t.to_dict() for t in confirmed],
            "breach_active":       any(t.threat > 0.5 for t in confirmed),
            "high_threat_targets": [t.to_dict() for t in confirmed if t.threat > 0.7],
        }
        threading.Thread(
            target=self._call_groq, args=(scene_data, client), daemon=True
        ).start()

    def _call_groq(self, scene_data: dict, client):
        try:
            prompt = (
                f"Perimeter scene snapshot:\n```json\n"
                f"{json.dumps(scene_data, indent=2, default=float)}\n```\n"
                f"Provide your tactical assessment."
            )
            resp = client.chat.completions.create(
                model=CFG["groq_model"],
                max_tokens=CFG["groq_max_tokens"],
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            self._result_queue.put(text)
        except Exception as e:
            self._result_queue.put(f"[GROQ ERROR] {e}")
        finally:
            self._busy = False

    def poll(self) -> bool:
        try:
            text = self._result_queue.get_nowait()
            words = text.split()
            lines, cur = [], ""
            for w in words:
                if len(cur) + len(w) + 1 > 55:
                    lines.append(cur)
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            if cur:
                lines.append(cur)
            self._last_lines = lines if lines else ["(empty response)"]
            return True
        except queue.Empty:
            return False

    @property
    def lines(self) -> List[str]:
        return self._last_lines

    @property
    def busy(self) -> bool:
        return self._busy
