"""Frozen, identical constrained measurement for every size and seed."""

import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

SEEDS = (42, 73, 101)
ROOT = Path("/workspace/modellab")


def protocol():
    return {
        "version": "size-v1",
        "seeds": list(SEEDS),
        "rank": 4,
        "alpha": 8,
        "arms": ["base"] + [f"seed_{s}" for s in SEEDS],
        "decoder": "greedy-schema",
        "grammar": "lm-format-enforcer==0.11.3",
        "max_new_tokens": 96,
        "prompt": "unchanged receipt-v2 extraction instruction",
        "timing": "CUDA-synchronized generation; batch one",
        "primary": "1.5B rank 4 seed 42 versus 7B base on 256 locked notes",
    }


def collect(bundle, seed, generate, save, report):
    arms = (["base"] if seed == SEEDS[0] else []) + [f"seed_{seed}"]
    for arm in arms:
        generate(bundle["train"][0], arm)
    for partition, split in (("development", "validation"), ("locked", "test")):
        groups = report["partitions"].setdefault(partition, {})
        if any(a in groups for a in arms):
            raise ValueError("Refusing to overwrite existing seed predictions.")
        for arm in arms:
            groups[arm] = {"examples": [], "total": 0, "correct": 0}
        for i, row in enumerate(bundle[split]):
            for arm in arms if i % 2 == 0 else list(reversed(arms)):
                example = generate(row, arm)
                group = groups[arm]
                group["examples"].append(example)
                group["total"] += 1
                group["correct"] += int(example["correct"])
            if (i + 1) % 8 == 0:
                save(report)
        save(report)


def run(model, tok, bundle, render, _write):
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )
    from receipt_scores import SCHEMA, score

    path = ROOT / "evaluation.json"
    seed = int(os.environ["MODELLAB_TRAIN_SEED"])
    report = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "protocol": protocol(),
            "split": "Independent receipt benchmark",
            "limitations": "Original generated notes, shared templates and three training seeds.",
            "partitions": {},
        }
    )

    def save(value):
        tmp = path.with_suffix(".partial")
        tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
        tmp.replace(path)

    model.eval()
    alphabet = build_token_enforcer_tokenizer_data(tok)

    def generate(row, arm):
        prompt = render(row)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        constraint = build_transformers_prefix_allowed_tokens_fn(
            alphabet, JsonSchemaParser(SCHEMA)
        )
        with model.disable_adapter() if arm == "base" else nullcontext():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter_ns()
            output = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                prefix_allowed_tokens_fn=constraint,
                max_new_tokens=96,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
            torch.cuda.synchronize()
            latency = (time.perf_counter_ns() - start) / 1e6
        ids = output[0, inputs.input_ids.shape[1] :].tolist()
        prediction = tok.decode(ids, skip_special_tokens=True).strip()
        return {
            "id": row["id"],
            "prompt": row["prompt"],
            "expected": row["completion"],
            "prediction": prediction,
            "template": row["template"],
            "difficulty": row.get("difficulty", "warm-up"),
            "unseen_template": row["unseen_template"],
            "unseen_pair": row["unseen_pair"],
            "latency_ms": latency,
            "generation_peak_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "generated_tokens": len(ids),
            "input_tokens": int(inputs.input_ids.shape[1]),
            "hit_token_limit": len(ids) == 96 and ids[-1] != tok.eos_token_id,
            "correct": score(prediction, row["completion"])["exact"],
        }

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        collect(bundle, seed, generate, save, report)
