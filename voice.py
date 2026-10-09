"""Speech-to-text via NVIDIA Canary-1B (Riva HTTP ASR) on the event GPU endpoints."""
import os

import requests


def transcribe(wav_bytes, filename="question.wav"):
    url = os.environ.get("CANARY_1B_URL", "http://166.19.38.112:8004").rstrip("/")
    r = requests.post(
        f"{url}/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {os.environ['GPU_BEARER_TOKEN']}"},
        files={"file": (filename, wav_bytes, "audio/wav")},
        data={"model": "nvidia/canary-1b", "language": "en"},
        timeout=60,
    )
    r.raise_for_status()
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"text": r.text}
    return (body.get("text") or "").strip()
