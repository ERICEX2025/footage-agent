# Food Court Watch

Loss-prevention agent for small food businesses. Owners ask questions about their security footage ("did anyone walk out without paying?") and get answers grounded in timestamped, playable clips.

An LLM agent (W&B Inference, traced with Weave) plans multi-step investigations over the VAST VSS stack:

| Tool | Backed by |
|------|-----------|
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

Sponsor tools: VAST Data (AI OS, VastDB, DataEngine), NVIDIA (Cosmos-Reason, Cosmos-Embed, VSS blueprint), Weights & Biases (Inference, Weave), CoreWeave (GPUs), Cursor.
