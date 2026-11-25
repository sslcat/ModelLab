"""GPU worker. No training imports or model downloads on the control plane."""

import hashlib
from importlib.metadata import distributions
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace/modellab")


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    path = ROOT / name
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def equivalent(pred, target):
    try:
        # JSON canonicalization retains numeric/Boolean type distinctions.
        def unique_keys(pairs):
            obj = dict(pairs)
            if len(obj) != len(pairs):
                raise ValueError("Ambiguous JSON object")
            return obj

        def normalize(text):
            value = json.loads(text, object_pairs_hook=unique_keys)
            return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)

        return normalize(pred) == normalize(target)
    except ValueError:
        return pred.strip() == target.strip()


def main():
    if platform.system() != "Linux" or not os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("The worker requires a RunPod Linux GPU.")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required.")
    cfg, bundle = read("config.json"), read("data.json")
    encoded = json.dumps(
        bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != cfg["bundle_hash"]:
        raise ValueError("Dataset checksum mismatch.")
    set_seed(cfg["seed"])
    common = dict(revision=cfg["revision"], trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(cfg["model"], **common)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def prompt(row):
        return tok.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

    lengths = {}

    def records(split):
        rows = [
            {"prompt": prompt(row), "completion": row["completion"] + tok.eos_token}
            for row in bundle[split]
        ]
        sizes = sorted(
            len(
                tok(row["prompt"] + row["completion"], add_special_tokens=False)[
                    "input_ids"
                ]
            )
            for row in rows
        )
        lengths[split] = dict(
            min=sizes[0],
            max=sizes[-1],
            mean=sum(sizes) / len(sizes),
            p95=sizes[math.ceil(0.95 * len(sizes)) - 1],
        )
        if sizes[-1] > cfg["seq_len"]:
            raise ValueError(
                f"A {split} example needs {sizes[-1]} tokens. Increase sequence length."
            )
        return Dataset.from_list(rows)

    train_data, val_data = records("train"), records("validation")
    write("lengths.json", lengths)
    quant = cfg["method"] == "QLoRA"
    kwargs = (
        {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        }
        if quant
        else {}
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        use_safetensors=True,
        **common,
        **kwargs,
    )
    if quant:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(
        base,
        LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["alpha"],
            lora_dropout=cfg["dropout"],
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    write(
        "environment.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "packages": sorted(
                f"{d.metadata['Name']}=={d.version}"
                for d in distributions()
                if d.metadata["Name"]
            ),
            "trainable": trainable,
            "parameters": total,
        },
    )
    started = time.monotonic()

    class Metrics(TrainerCallback):
        last_count = 0
        last_time = started

        def on_train_begin(self, args, state, control, **kwargs):
            self.last_time = time.monotonic()

        def on_log(self, args, state, control, logs=None, **kwargs):
            row = {
                k: float(v)
                for k, v in (logs or {}).items()
                if isinstance(v, (int, float)) and math.isfinite(v)
            }
            row.update(
                step=state.global_step, elapsed_s=round(time.monotonic() - started, 3)
            )
            row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            count = row.get("num_tokens")
            if count is not None and count > self.last_count and "loss" in row:
                now = time.monotonic()
                row["tokens_per_second"] = (count - self.last_count) / max(
                    0.001, now - self.last_time
                )
                self.last_count, self.last_time = count, now
            sample = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if sample.returncode == 0:
                util, mem = sample.stdout.strip().split(",")
                row.update(gpu_percent=int(util), vram_mib=int(mem))
            with (ROOT / "metrics.jsonl").open("a") as out:
                out.write(json.dumps(row, allow_nan=False) + "\n")

    args = SFTConfig(
        output_dir=str(ROOT / "checkpoints"),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        max_length=cfg["seq_len"],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        logging_steps=1,
        report_to="none",
        dataset_num_proc=1,
        optim="adamw_torch",
        dataloader_pin_memory=True,
    )
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tok,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[Metrics()],
    )
    if cfg.get("protocol"):
        from controlled import protocol, run
        if cfg["protocol"] != protocol():
            raise ValueError("Frozen evaluation protocol mismatch.")
        trained = trainer.train()
        trainer.log_metrics("train", trained.metrics)
        trainer.save_model(str(ROOT / "adapter"))
        run(model, tok, bundle, prompt, write)
        return

    # Same frozen precision, prompts, greedy decoder, and examples in both arms.
    test = bundle.get("test", bundle["validation"])[:64]

    def evaluate():
        model.eval()
        output = []
        # K-bit preparation keeps some frozen layers in FP32. Match the
        # trainer's autocast context for both greedy inference arms.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in test:
                inputs = tok(
                    prompt(row), return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                pred = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                text = tok.decode(
                    pred[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
                ).strip()
                output.append(
                    dict(
                        prompt=row["prompt"],
                        expected=row["completion"],
                        prediction=text,
                        correct=equivalent(text, row["completion"]),
                    )
                )
        return {
            "correct": sum(x["correct"] for x in output),
            "total": len(test),
            "examples": output,
        }

    with model.disable_adapter():
        baseline = evaluate()
    report = {
        "split": "held-out paraphrase"
        if "test" in bundle
        else "validation (also used for loss)",
        "generation_limit": 96,
        "base": baseline,
        "limitations": "Synthetic language and a small paired sample; exact-match misses semantically equivalent free text.",
    }
    write("evaluation.json", report)
    trained = trainer.train()
    trainer.log_metrics("train", trained.metrics)
    Metrics().on_log(None, trainer.state, None, trained.metrics)
    trainer.save_model(str(ROOT / "adapter"))
    report["tuned"] = evaluate()
    report["difference"] = (report["tuned"]["correct"] - baseline["correct"]) / len(
        test
    )
    write("evaluation.json", report)


if __name__ == "__main__":
    main()
