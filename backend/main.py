from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
import os
import uvicorn

from dotenv import load_dotenv

from training.inference import load_model_from_source, generate_target

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
CHECKPOINT_REPO_ID = os.getenv("CHECKPOINT_REPO_ID")
CHECKPOINT_FILENAME = os.getenv("CHECKPOINT_FILENAME", "transformer_noam_v2_epoch_30.pt")
CHECKPOINT_REVISION = os.getenv("CHECKPOINT_REVISION")

app = FastAPI()
model = None
config = None


class GenerationRequest(BaseModel):
    user_exs: str | None = None
    decoding_method: Literal["greedy", "beam"] = "greedy"
    beam_width: int = Field(default=3, ge=1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_generation_model():
    global model, config
    print(f"Loading model from checkpoint: {CHECKPOINT_PATH} {CHECKPOINT_REPO_ID} {CHECKPOINT_FILENAME} {CHECKPOINT_REVISION}   ")
    model, config = load_model_from_source(
        checkpoint_path=CHECKPOINT_PATH,
        hf_repo_id=CHECKPOINT_REPO_ID,
        hf_filename=CHECKPOINT_FILENAME,
        hf_revision=CHECKPOINT_REVISION,
    )

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate(request: GenerationRequest):
    generated_sentences = generate_target(
        model,
        config,
        n_examples=1,
        user_exs=request.user_exs,
        decoding_method=request.decoding_method,
        beam_width=request.beam_width,
    )
    return {
        "message": "Generation complete.",
        "decoding_method": request.decoding_method,
        "beam_width": request.beam_width if request.decoding_method == "beam" else None,
        "user_exs": request.user_exs,
        "generated_sentences": generated_sentences,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)