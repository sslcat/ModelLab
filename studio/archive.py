import ctypes
import hashlib
import json
import os
import re
from pathlib import Path

from .cloud import CloudError, artifact_root, one_drive
from .datasets import canon

PLAIN = {
    "config.json",
    "metrics.jsonl",
    "lengths.json",
    "environment.json",
    "evaluation.json",
}
ADAPTER = re.compile(
    r"(?:adapter|checkpoints/checkpoint-[0-9]+)/(adapter_config\.json|adapter_model\.safetensors)"
)


def sumfile(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sync_state(folder):
    if not one_drive() or not folder.is_relative_to(one_drive()):
        return "Local archive"
    if os.name == "nt":
        try:
            from ctypes import wintypes

            # Python can see ordinary files when Windows disguises cloud placeholders.
            # Expose native metadata for this thread, then restore its previous mode.
            mode = ctypes.windll.ntdll.RtlSetThreadPlaceholderCompatibilityMode
            mode.argtypes, mode.restype = [ctypes.c_byte], ctypes.c_byte
            find = ctypes.windll.kernel32.FindFirstFileW
            data_type = ctypes.POINTER(wintypes.WIN32_FIND_DATAW)
            find.argtypes, find.restype = [wintypes.LPCWSTR, data_type], wintypes.HANDLE
            close = ctypes.windll.kernel32.FindClose
            close.argtypes = [wintypes.HANDLE]
            check = ctypes.windll.CldApi.CfGetPlaceholderStateFromFindData
            check.argtypes, check.restype = [data_type], wintypes.DWORD
            previous = mode(2)
            if previous < 0:
                return "OneDrive sync unconfirmed"
            try:
                files = [path for path in folder.rglob("*") if path.is_file()]
                for path in files:
                    data = wintypes.WIN32_FIND_DATAW()
                    handle = find(str(path), ctypes.byref(data))
                    if handle == wintypes.HANDLE(-1).value:
                        return "OneDrive sync unconfirmed"
                    try:
                        state = check(ctypes.byref(data))
                        if state == 0xFFFFFFFF or not state & 8:
                            return "OneDrive sync unconfirmed"
                    finally:
                        close(handle)
                if files:
                    return "OneDrive in sync"
            finally:
                mode(previous)
        except (OSError, AttributeError):
            pass
    return "OneDrive sync unconfirmed"


def safe_quote(quote):
    return {k: v for k, v in (quote or {}).items() if k not in {"token", "account"}}


def recovery_record(doc):
    folder = artifact_root() / doc["id"]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "recovery.json"
    tmp = dest.with_suffix(".tmp")
    record = {
        k: doc.get(k)
        for k in (
            "id",
            "pod",
            "resource_name",
            "approved_at",
            "deadline",
            "config",
            "dataset_name",
            "bundle_hash",
        )
    }
    record["quote"] = safe_quote(doc.get("quote"))
    tmp.write_text(canon(record), encoding="utf8")
    tmp.replace(dest)


def allowed_file(doc, name):
    if doc.get("size_study"):
        from research.size_results import expected_files
        return name in expected_files()
    return name in PLAIN or bool(ADAPTER.fullmatch(name))


def retrieve(doc, manifest, transport):
    files = manifest["files"]
    if (
        manifest["status"] not in {"completed", "failed", "cancelled", "timed_out"}
        or "config.json" not in files
    ):
        raise CloudError("Worker manifest is incomplete.")
    if (
        manifest["status"] == "completed"
        and not {
            "metrics.jsonl",
            "evaluation.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "environment.json",
            "lengths.json",
        }
        <= files.keys()
    ):
        raise CloudError("Required completion artifacts are missing.")
    folder = artifact_root() / doc["id"]
    folder.mkdir(parents=True, exist_ok=True)
    for name, info in files.items():
        if not allowed_file(doc, name):
            raise CloudError("Unexpected artifact path.")
        if (
            not re.fullmatch("[a-f0-9]{64}", info["sha256"])
            or not 0 <= info["size"] <= 384 * 1024**2
        ):
            raise CloudError("Invalid artifact manifest.")
        dest = folder / name
        if dest.exists() and sumfile(dest) == info["sha256"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".partial")
        try:
            with tmp.open("wb") as out:
                transport.command(
                    "cat /workspace/modellab/" + name, timeout=100, output=out
                )
            if tmp.stat().st_size != info["size"] or sumfile(tmp) != info["sha256"]:
                raise CloudError(
                    "Artifact checksum mismatch. The pod disk is retained."
                )
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
    config = json.loads((folder / "config.json").read_text(encoding="utf8"))
    if (
        config.get("experiment") != doc["id"]
        or config.get("bundle_hash") != doc["bundle_hash"]
    ):
        raise CloudError("Artifact identity mismatch.")
    evaluation = (
        json.loads((folder / "evaluation.json").read_text(encoding="utf8"))
        if (folder / "evaluation.json").exists()
        else None
    )
    if manifest["status"] == "completed":
        if doc.get("size_study"):
            from research.size_results import verify_archive
            verify_archive(doc, folder, evaluation, files)
        elif config.get("protocol"):
            from .research import verify_evaluation
            verify_evaluation(config, evaluation)
        elif not evaluation or "base" not in evaluation or "tuned" not in evaluation:
            raise CloudError("Paired evaluation missing.")
    report = {
        "experiment": doc["id"],
        "dataset": doc["dataset_name"],
        "config": config,
        "quote": safe_quote(doc["quote"]),
        "result": manifest["status"],
        "detail": manifest.get("detail", ""),
        "evaluation": evaluation,
        "files": files,
        "model_token_lengths": (json.loads((folder / "lengths.json").read_text(encoding="utf8"))
                                if "lengths.json" in files else None),
        "provider_cost_at_export": doc.get("actual"),
    }
    for name, obj in (("report.json", report), ("manifest.json", manifest)):
        tmp = folder / (name + ".tmp")
        tmp.write_text(canon(obj), encoding="utf8")
        tmp.replace(folder / name)
    return {
        "archive": str(folder),
        "archived": True,
        "adapter_available": any(ADAPTER.fullmatch(n) for n in files),
        "sync": sync_state(folder),
        "evaluation": evaluation,
        "outcome": manifest["status"],
        "detail": manifest.get("detail", ""),
    }


def verified(doc):
    folder = Path(doc["archive"])
    if not folder.is_dir():
        return False
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
        if doc.get("size_study"):
            from research.size_results import expected_files
            report = json.loads((folder / "report.json").read_text(encoding="utf8"))
            if (report.get("experiment") != doc["id"] or report.get("files") != manifest.get("files")
                or report.get("result") != manifest.get("status")
                or (manifest.get("status") == "completed" and set(manifest["files"]) != expected_files())):
                return False
        return (folder / "report.json").is_file() and all(
            allowed_file(doc, name)
            and (folder / name).is_file()
            and sumfile(folder / name) == entry["sha256"]
            for name, entry in manifest["files"].items()
        )
    except (OSError, KeyError, ValueError):
        return False
