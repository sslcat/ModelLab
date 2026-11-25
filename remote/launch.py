"""Supervise one immutable job. flock prevents a reconnect starting a second trainer."""

import hashlib
from importlib.metadata import version
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/modellab")


def finish(status, error=""):
    # Only export public experiment records and PEFT adapters. No caches or base weights.
    files = [
        "config.json",
        "metrics.jsonl",
        "lengths.json",
        "environment.json",
        "evaluation.json",
    ]
    candidates = [
        ROOT / "adapter",
        *sorted(
            (ROOT / "checkpoints").glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
            reverse=True,
        ),
    ]
    for directory in candidates:
        pair = [
            directory / name
            for name in ("adapter_config.json", "adapter_model.safetensors")
        ]
        if all(p.is_file() and p.stat().st_size for p in pair):
            files.extend(p.relative_to(ROOT).as_posix() for p in pair)
            break
    info = {}
    for name in files:
        path = ROOT / name
        if path.is_file():
            with path.open("rb") as source:
                info[name] = {
                    "sha256": hashlib.file_digest(source, "sha256").hexdigest(),
                    "size": path.stat().st_size,
                }
    tmp = ROOT / "manifest.tmp"
    tmp.write_text(json.dumps({"status": status, "detail": error, "files": info}))
    tmp.replace(ROOT / "manifest.json")


def main():
    if not os.environ.get("RUNPOD_POD_ID") or sys.platform != "linux":
        raise RuntimeError("RunPod only")
    import fcntl

    os.chdir(ROOT)
    lock = open("worker.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if Path("manifest.json").exists():
        return
    (ROOT / "metrics.jsonl").touch()
    cfg = json.loads((ROOT / "config.json").read_text())
    env = dict(
        os.environ,
        PIP_BREAK_SYSTEM_PACKAGES="1",
        HF_HOME="/tmp/modellab-hf",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    for var in list(env):
        if "KEY" in var or "TOKEN" in var and var != "TOKENIZERS_PARALLELISM":
            env.pop(var)
    try:
        torch_version = version("torch")
        if not torch_version.startswith("2.8."):
            finish(
                "failed",
                "The pinned image did not supply PyTorch 2.8. Image inspection is required.",
            )
            return
        Path("image-constraints.txt").write_text("torch==" + torch_version + "\n")
    except Exception:
        finish("failed", "Could not confirm the image-provided PyTorch package.")
        return
    phases = [
        (
            "dependencies",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "-c",
                "image-constraints.txt",
                "-r",
                "requirements.txt",
                *(["lm-format-enforcer==0.11.3"] if cfg.get("protocol") else []),
            ],
        ),
        ("training", [sys.executable, "-u", "train.py"]),
    ]
    try:
        for phase, command in phases:
            with open("worker.log", "ab") as log:
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    env=env,
                    start_new_session=True,
                )
                while proc.poll() is None:
                    stop = (ROOT / "cancel").exists() or time.time() >= cfg[
                        "deadline"
                    ] - 240
                    if stop:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(12)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        finish(
                            "cancelled" if (ROOT / "cancel").exists() else "timed_out"
                        )
                        return
                    time.sleep(1)
                if proc.returncode:
                    finish(
                        "failed",
                        f"{phase} exited with code {proc.returncode}. Review the recorded configuration and environment.",
                    )
                    # Keep detailed diagnostics on the pod. The manifest carries the phase and exit code.
                    lines = (
                        (ROOT / "worker.log")
                        .read_bytes()[-4096:]
                        .decode(errors="replace")
                        .splitlines()
                    )
                    print(lines[-1:] or ["No diagnostic line."], flush=True)
                    return
        finish("completed")
    except Exception as exc:
        finish("failed", type(exc).__name__)


if __name__ == "__main__":
    main()
