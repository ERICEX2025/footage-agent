"""Footage agent: W&B-hosted LLM plans over VSS tools (Cosmos search, YOLO counts, Cosmos synthesis)."""
import json
import os

import openai
import weave

from vss_client import VSSClient

SYSTEM_PROMPT = """You are JARVIS, a voice assistant connected to a network of cameras: highway
traffic cams (Nashville I-24), dashcam drives (Toronto), a neighborhood street cam, San Francisco
street cams, a warehouse, and indoor building cams. The user speaks to you; your reply is read
aloud, so write it to be heard.

Every video is split into ~5 second segments. Each segment has a Cosmos-Reason description
(reasoning_content), YOLO object counts (object_counts), camera and location, and start/end seconds.

How to work:
1. Use search_footage to find candidate moments. If results are weak, try 2-3 other phrasings.
2. Use video_timeline to see what happened before/after a moment in the same video.
3. Use object_detections when the question is about counts or presence ("how many cars").
4. Use summarize_video for a whole-video question.

Reply format:
- First 1-3 sentences: a direct spoken answer, a bit of JARVIS personality, no markdown, no
  file names, no URLs (say "the Toronto drive at 2 minutes 15" instead).
- Then a line "---" followed by a short bullet list of evidence, each cited as
  [video filename @ start-end s].
If the footage does not show it, say so - never invent events."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_footage",
        "description": "Natural-language hybrid search over all video segments. Returns top matching 5s segments.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, e.g. 'person without a hard hat near forklift'"},
            "top_k": {"type": "integer", "default": 8},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "video_timeline",
        "description": "Chronological list of all segment descriptions for one video (use original_video from search results).",
        "parameters": {"type": "object", "properties": {
            "original_video": {"type": "string"},
        }, "required": ["original_video"]},
    }},
    {"type": "function", "function": {
        "name": "object_detections",
        "description": "YOLO object detections (classes, counts, confidences) for one segment (use 'source' from search results).",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string"},
        }, "required": ["source"]},
    }},
    {"type": "function", "function": {
        "name": "summarize_video",
        "description": "Ask Cosmos-Reason a question about an entire video (e.g. 'what happened chronologically?').",
        "parameters": {"type": "object", "properties": {
            "original_video": {"type": "string"},
            "question": {"type": "string"},
        }, "required": ["original_video", "question"]},
    }},
]


def _compact_segment(s):
    return {
        "source": s.get("source"),
        "original_video": s.get("original_video"),
        "filename": s.get("filename"),
        "start": s.get("segment_start_sec"),
        "end": s.get("segment_end_sec"),
        "score": round(s["similarity_score"], 3) if s.get("similarity_score") is not None else None,
        "description": (s.get("reasoning_content") or "")[:600],
        "objects": s.get("object_counts") or s.get("object_classes"),
        "camera": s.get("camera_id"),
        "location": s.get("location"),
    }


def _summarize_detections(d):
    """Detection sidecars can be large; reduce to per-class stats."""
    frames = d.get("frames") if isinstance(d, dict) else None
    if not isinstance(frames, list):
        return json.dumps(d)[:3000]
    stats = {}
    for f in frames:
        per_frame = {}
        for det in f.get("detections") or f.get("boxes") or []:
            label = det.get("label") or det.get("class_name") or det.get("class")
            per_frame[label] = per_frame.get(label, 0) + 1
            stats.setdefault(label, {"max_in_frame": 0, "max_conf": 0.0})
            conf = det.get("confidence") or det.get("conf") or 0
            stats[label]["max_conf"] = round(max(stats[label]["max_conf"], conf), 3)
        for label, n in per_frame.items():
            stats[label]["max_in_frame"] = max(stats[label]["max_in_frame"], n)
    return json.dumps({"frames": len(frames), "classes": stats})


def _wandb_project():
    """VM config sets WANDB_TEAM + WANDB_PROJECT separately; W&B wants 'team/project'."""
    proj = os.environ.get("WANDB_PROJECT")
    team = os.environ.get("WANDB_TEAM")
    if proj and team and "/" not in proj:
        return f"{team}/{proj}"
    return proj


class FootageAgent:
    def __init__(self, vss=None, model=None):
        self.vss = vss or VSSClient()
        self.model = model or os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
        self.llm = openai.OpenAI(
            base_url="https://api.inference.wandb.ai/v1",
            api_key=os.environ["WANDB_API_KEY"],
        )
        self.evidence = {}  # source -> compact segment, for the UI to render clips

    # --- tool implementations ----------------------------------------------
    @weave.op()
    def search_footage(self, query, top_k=8):
        res = self.vss.search(query, top_k=top_k)
        segs = [_compact_segment(s) for s in res.get("results", [])]
        for s in segs:
            self.evidence[s["source"]] = s
        return json.dumps(segs)

    @weave.op()
    def video_timeline(self, original_video):
        res = self.vss.segments(original_video)
        segs = sorted((_compact_segment(s) for s in res.get("segments", [])), key=lambda s: s["start"] or 0)
        for s in segs:
            s["description"] = s["description"][:250]
            s.pop("original_video", None)
        return json.dumps(segs[:60])

    @weave.op()
    def object_detections(self, source):
        return _summarize_detections(self.vss.detections(source))

    @weave.op()
    def summarize_video(self, original_video, question):
        res = self.vss.synthesize(original_video, question)
        return json.dumps(res)[:4000]

    # --- loop ----------------------------------------------------------------
    @weave.op()
    def run(self, question, history=None, max_steps=8, on_step=None):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []),
                    {"role": "user", "content": question}]
        for _ in range(max_steps):
            resp = self.llm.chat.completions.create(
                model=self.model, messages=messages, tools=TOOLS, temperature=0.2,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                if on_step:
                    on_step(call.function.name, args)
                try:
                    out = getattr(self, call.function.name)(**args)
                except Exception as e:  # surface tool errors to the model instead of crashing
                    out = json.dumps({"error": str(e)})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
        messages.append({"role": "user", "content": "Stop calling tools and give your best answer now."})
        return self.llm.chat.completions.create(model=self.model, messages=messages).choices[0].message.content
