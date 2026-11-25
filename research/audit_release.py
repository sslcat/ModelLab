"""Offline audit of Git-indexed release files, including decompressed public evidence."""

import gzip
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BANNED_DIRS = {
    ".venv",
    "node_modules",
    ".next",
    "out",
    "state",
    "artifacts",
    ".ssh",
    ".aws",
    "__pycache__",
    ".cache",
}
BANNED_EXT = {
    ".pem",
    ".key",
    ".pfx",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".safetensors",
    ".ckpt",
    ".pth",
    ".pt",
    ".bin",
    ".toml.private",
}
INDICATORS = {
    "private key": rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----",
    "provider token": rb"(?<![A-Za-z0-9])(?:rpa_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{25,}|AKIA[A-Z0-9]{16})",
    "personal path": rb"(?i)(?:[a-z]:[\\/]+Users[\\/]+(?!Public\b|Default\b)[a-z\d._-]+|/home/[a-z\d._-]+/(?:\.ssh|OneDrive)|file:/{3})",
    "credential URL": rb"https?://[^/\s:@]+:[^/@\s]+@",
    "literal credential": rb"(?i)(?:runpod_api_key|access_token|client_secret|refresh_token)\s*[=:]\s*[\x22\x27][A-Za-z0-9_.+/=-]{16,}[\x22\x27]",
}


def audit():
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    findings = []
    size = 0
    count = 0
    for name in filter(None, names):
        path = PurePosixPath(name)
        if (
            set(path.parts) & BANNED_DIRS
            or path.suffix.lower() in BANNED_EXT
            or (path.name.startswith(".env") and path.name != ".env.example")
        ):
            findings.append((name, "excluded file"))
        data = (ROOT / name).read_bytes()
        size += len(data)
        count += 1
        if name.endswith(".json.gz"):
            data = gzip.decompress(data)
            record = json.loads(data)
            if record["id"] + ".json.gz" != path.name or any(
                k in record
                for k in (
                    "pod",
                    "archive",
                    "account",
                    "quote",
                    "resource_name",
                    "token",
                    "deadline",
                )
            ):
                findings.append((name, "operational field in public record"))
        for why, pattern in INDICATORS.items():
            if re.search(pattern, data):
                findings.append((name, why))
    if findings:
        raise SystemExit("Release audit findings (values withheld): " + str(findings))
    print(
        f"Scanned {count} tracked files ({size:,} bytes), including decompressed research records."
    )


if __name__ == "__main__":
    audit()
