# SafeFloor

An AI safety officer for warehouses. It watches the cameras a site already has, finds near-misses between people and forklifts, blocked walkways and other hazards, and gives the safety manager a daily report with clips. Ask by voice ("any close calls in the aisles today?") and get a spoken answer plus the evidence.

**Why:** forklift accidents injure ~35,000 US workers a year and a single serious injury costs a company $40k-150k+. Warehouses already record everything, but nobody watches the footage until after someone gets hurt.

**Demo footage:** the event's pre-indexed warehouse corpus (`sdg_warehouse_cam-2`). Set `SITE_CAMERA` to point it at another camera.

An LLM agent (W&B Inference, traced with Weave) plans multi-step investigations over the VAST VSS stack:

| Step / tool | Backed by |
|------|-----------|
| Voice input | NVIDIA Canary-1B speech-to-text |
| `search_footage` | VAST VastDB hybrid vector search over NVIDIA Cosmos-Reason segment descriptions + Cosmos-Embed visual vectors |
| `video_timeline` | Per-video segment rows in VastDB (before/after context) |
| `object_detections` | YOLO11 detections from the VAST DataEngine ingest pipeline |
| `summarize_video` | NVIDIA Cosmos-Reason whole-video synthesis |

## Run

```bash
cp .env.example .env   # fill in VSS + W&B values
uv venv && uv pip install -r requirements.txt
python smoke_test.py   # verify W&B inference + VSS access
streamlit run app.py
```

Spoken replies use the browser's speech synthesis.

Sponsor tools: VAST Data (AI OS, VastDB, DataEngine), NVIDIA (Cosmos-Reason, Cosmos-Embed, VSS blueprint), Weights & Biases (Inference, Weave), CoreWeave (GPUs), Cursor.
