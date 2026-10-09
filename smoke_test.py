"""Check each dependency independently: python smoke_test.py"""
import json, os
from dotenv import load_dotenv
load_dotenv()

try:
    import openai
    llm = openai.OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=os.environ["WANDB_API_KEY"],
                        project=os.environ.get("WANDB_PROJECT"))
    print("W&B models:", [m.id for m in llm.models.list()])
except Exception as e:
    print("W&B inference FAILED:", e)

try:
    from vss_client import VSSClient
    vss = VSSClient(); vss.login(); print("VSS login OK")
    res = vss.search("a person walking", top_k=3)
    for r in res.get("results", []):
        print(json.dumps({k: r.get(k) for k in ("filename", "source", "original_video", "segment_start_sec",
                                                "similarity_score", "object_counts", "reasoning_content")})[:600])
except Exception as e:
    print("VSS FAILED:", e)
