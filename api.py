# main.py
from fastapi import FastAPI, Request
from lib import Sampler
import os
model_dir = os.getenv("MODEL", "shakespeare")
device = os.getenv("DEVICE", "cpu")
sampler = Sampler(out_dir=f'out-{model_dir}', device="cpu")

app = FastAPI()
@app.post("/generate")
async def generate_text_api(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "\n")
    num_samples = int(data.get("num_samples", 1))
    max_new_tokens = int(data.get("max_new_tokens", 100))
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("top_k", 200))

    result = sampler.generate_text(
        prompt=prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    return {"generated_texts": result}
