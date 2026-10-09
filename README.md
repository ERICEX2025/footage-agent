# JARVIS for your cameras

Talk to hours of camera footage. Ask out loud ("Jarvis, show me a person close to a moving car") and get a spoken answer plus the matching clips.

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
