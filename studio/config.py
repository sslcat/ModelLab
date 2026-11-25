import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}
_size_models = json.loads((Path(__file__).resolve().parents[1] / "research" / "size_models.json").read_text(encoding="utf8"))
MODELS.update({m["model"]: m["revision"] for m in _size_models[:4]})
DISK = 40
VOLUME = 10
BUDGET = 20.00
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(default="Receipt extraction · pilot", min_length=1, max_length=80)
    dataset: str = Field(min_length=1, max_length=64)
    model: Literal["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"] = (
        next(iter(MODELS))
    )
    method: Literal["QLoRA", "LoRA"] = "QLoRA"
    gpu: str = Field(default="NVIDIA GeForce RTX 4090", min_length=1, max_length=100)
    minutes: int = Field(default=90, ge=15, le=180)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    epochs: int = Field(default=1, ge=1, le=12)
    batch_size: int = Field(default=1, ge=1, le=8)
    grad_accum: int = Field(default=8, ge=1, le=64)
    rank: int = Field(default=8, ge=2, le=64)
    alpha: int = Field(default=16, ge=1, le=128)
    dropout: float = Field(default=0.05, ge=0, le=0.5)
    seq_len: int = Field(default=256, ge=64, le=2048)
    seed: int = Field(default=42, ge=0, le=2147483647)
