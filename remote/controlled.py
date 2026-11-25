"""Frozen five-arm receipt protocol. GPU dependencies load only inside run()."""

import time
from contextlib import nullcontext

try:
    from .receipt_scores import SCHEMA, score, summarize
except ImportError:  # Standalone files staged on the pod.
    from receipt_scores import SCHEMA, score, summarize

VERSION = "receipt-v2.1"
ARMS = ("base", "prompted", "base_constrained", "tuned", "tuned_constrained")
IMPROVED = (
    "Respond with a single JSON object. Start with { and end with }. Omit Markdown and commentary. "
    "Use exactly item, quantity and site. Copy the received product and receiving location. "
    "Use the accepted count as an integer. Ignore order estimates, origins and unreceived goods."
)
MAX_NEW_TOKENS = 96


def protocol():
    return dict(
        version=VERSION,
        arms=list(ARMS),
        improved_instruction=IMPROVED,
        schema=SCHEMA,
        decoder="greedy",
        max_new_tokens=MAX_NEW_TOKENS,
        grammar="lm-format-enforcer==0.11.3",
        timing="CUDA-synchronized generate call in milliseconds, batch 1",
        splits=["development", "locked"],
        primary="rank 8, locked complete record exact match: tuned_constrained vs base_constrained",
        semantic_diagnostic="Remove at most one enclosing Markdown fence, then compare typed JSON records.",
    )


def run(model, tok, bundle, render, write):
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )

    model.eval()
    alphabet = build_token_enforcer_tokenizer_data(tok)

    def generate(row, arm):
        text = row["prompt"]
        if arm == "prompted":
            text = IMPROVED + "\nNote: " + text.split("\nNote: ", 1)[1]
        inputs = tok(
            render({"prompt": text}), return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        options = {}
        if arm.endswith("constrained"):
            options["prefix_allowed_tokens_fn"] = (
                build_transformers_prefix_allowed_tokens_fn(
                    alphabet, JsonSchemaParser(SCHEMA)
                )
            )
        # With frozen base weights, disabling PEFT restores the quantized base even after training.
        with model.disable_adapter() if arm in ARMS[:3] else nullcontext():
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            out = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                **options,
            )
            torch.cuda.synchronize()
            latency = (time.perf_counter_ns() - start) / 1e6
        ids = out[0, inputs.input_ids.shape[1] :].tolist()
        pred = tok.decode(ids, skip_special_tokens=True).strip()
        return dict(
            id=row["id"],
            prompt=row["prompt"],
            input_prompt=text,
            expected=row["completion"],
            prediction=pred,
            correct=score(pred, row["completion"])["exact"],
            template=row["template"],
            unseen_template=row["unseen_template"],
            unseen_pair=row["unseen_pair"],
            latency_ms=latency,
            generated_tokens=len(ids),
            hit_token_limit=len(ids) == MAX_NEW_TOKENS and ids[-1] != tok.eos_token_id,
        )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return collect(bundle, generate, write)


def collect(bundle, generate, write):
    report = dict(
        protocol=protocol(),
        split="locked receipt test",
        limitations="Original synthetic notes, four held-out template families and one training seed.",
        partitions={},
    )
    for arm in ARMS:
        generate(bundle["train"][0], arm)  # Unscored warm-up in each mode.
    for partition, split in (("development", "validation"), ("locked", "test")):
        outputs = {arm: {"examples": [], "correct": 0, "total": 0} for arm in ARMS}
        report["partitions"][partition] = outputs
        for i, row in enumerate(bundle[split]):
            for arm in ARMS[i % len(ARMS) :] + ARMS[: i % len(ARMS)]:
                record = generate(row, arm)
                output = outputs[arm]
                output["examples"].append(record)
                output["total"] += 1
                output["correct"] += int(record["correct"])
            if (i + 1) % 8 == 0:
                write("evaluation.json", report)
        for output in outputs.values():
            output["metrics"] = summarize(output["examples"])
        write("evaluation.json", report)
    return report
