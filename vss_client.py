"""Thin client for the VAST VSS blueprint REST API (event build environment)."""
import os
from urllib.parse import urlencode

import requests


class VSSClient:
    def __init__(self, base_url=None, username=None, password=None):
        self.base = (base_url or os.environ.get("INGRESS_URL") or os.environ["VSS_BASE_URL"]).rstrip("/") + "/api/v1"
        self.username = username or os.environ.get("USERNAME") or os.environ["VSS_USERNAME"]
        self.password = password or os.environ.get("PASSWORD") or os.environ["VSS_PASSWORD"]
        self.token = None

    def login(self):
        r = requests.post(
            f"{self.base}/auth/login",
            json={"username": self.username, "password": self.password},
            timeout=30,
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]
        return self.token

    def _req(self, method, path, **kw):
        if not self.token:
            self.login()
        kw.setdefault("timeout", 120)
        headers = {"Authorization": f"Bearer {self.token}"}
        r = requests.request(method, f"{self.base}{path}", headers=headers, **kw)
        if r.status_code == 401:  # token expired
            self.login()
            headers["Authorization"] = f"Bearer {self.token}"
            r = requests.request(method, f"{self.base}{path}", headers=headers, **kw)
        r.raise_for_status()
        return r.json()

    # --- tools ---------------------------------------------------------------
    def search(self, query, top_k=10, min_similarity=0.1, metadata_filters=None):
        """Hybrid (caption + visual) search over Cosmos-described 5s segments."""
        return self._req("POST", "/tools/search", json={
            "query": query,
            "top_k": top_k,
            "min_similarity": min_similarity,
            "metadata_filters": metadata_filters or {},
            "llm_top_n": 1,
        })

    def segments(self, original_video):
        """All segments (timeline) for one uploaded video."""
        return self._req("GET", "/tools/segments", params={"original_video": original_video})

    def segment(self, source):
        return self._req("GET", "/tools/segment", params={"source": source})

    def detections(self, source):
        """YOLO11 per-frame bbox sidecar for one segment."""
        return self._req("GET", "/tools/detections", params={"source": source})

    def synthesize(self, original_video, question, max_segments=50):
        """Cosmos-Reason synthesis over a whole video."""
        return self._req("POST", "/tools/synthesize", json={
            "original_video": original_video,
            "question": question,
            "max_segments": max_segments,
        })

    def explore(self, **params):
        return self._req("GET", "/tools/explore", params=params)

    def ask(self, question, original_video=None, top_k=10):
        body = {"question": question, "top_k": top_k}
        if original_video:
            body["original_video"] = original_video
        return self._req("POST", "/agent/ask", json=body)

    def clip_bytes(self, source):
        """Fetch the clip server-side (the VSS host is only reachable inside the event network)."""
        if not self.token:
            self.login()
        r = requests.get(f"{self.base}/videos/stream", params={"source": source, "token": self.token}, timeout=60)
        r.raise_for_status()
        return r.content

    def stream_url(self, source):
        """Backend-proxied stream URL usable directly in a <video> tag."""
        if not self.token:
            self.login()
        return f"{self.base}/videos/stream?" + urlencode({"source": source, "token": self.token})
