import math
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import httpx

from . import store
from .config import BUDGET, DISK, IMAGE, VOLUME


class CloudError(RuntimeError):
    pass


class CreateRejected(CloudError):
    """Provider explicitly refused creation before allocating a resource."""


def key():
    if os.getenv("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    path = Path.home() / ".runpod" / "config.toml"
    if path.is_file():
        data = tomllib.loads(path.read_text(encoding="utf8"))
        return (
            data.get("api_key")
            or data.get("apikey")
            or data.get("default", {}).get("api_key")
        )
    return None


def storage_rate():
    return (DISK + VOLUME) * 0.10 / 720


def estimate(rate, minutes):
    hourly = rate + storage_rate()
    # Up to 24 hours of stopped-disk recovery plus a 20% price/tax contingency.
    storage_reserve = VOLUME * 0.20 / 720 * 24
    maximum = math.ceil((hourly * minutes / 60 + storage_reserve) * 1.20 * 100) / 100
    return {
        "gpu_hourly": rate,
        "storage_hourly": storage_rate(),
        "stopped_hourly": VOLUME * 0.20 / 720,
        "recovery_hours": 24,
        "minutes": minutes,
        "max_usd": maximum,
    }


def one_drive():
    return next(
        (
            Path(os.environ[k])
            for k in ("OneDriveConsumer", "OneDrive", "OneDriveCommercial")
            if os.getenv(k) and Path(os.environ[k]).is_dir()
        ),
        None,
    )


def artifact_root():
    return (one_drive() or store.home()) / "ModelLab" / "experiments"


class Provider:
    def __init__(self, client=None):
        self.client = client

    def request(self, method, path, **kwargs):
        if self.client is None and os.getenv("MODELLAB_OFFLINE") == "1":
            raise CloudError("Cloud access is disabled for offline verification.")
        api_key = key()
        if not api_key:
            raise CloudError(
                "Set RUNPOD_API_KEY in the local service environment to inspect the account."
            )
        try:
            response = (self.client or httpx).request(
                method,
                "https://api.runpod.io/v2/" + path,
                headers={"Authorization": "Bearer " + api_key},
                timeout=20,
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise CloudError(
                "RunPod did not confirm the request. Check connectivity and reconcile inventory."
            ) from exc
        if response.status_code == 404 and method == "GET" and path.startswith("pods/"):
            return None
        if (
            method == "POST"
            and path == "pods"
            and response.status_code in (400, 401, 402, 403, 404, 422)
        ):
            try:
                payload = response.json()
                diagnostic = payload.get("message") or payload.get("error") or "Request refused"
                if isinstance(diagnostic, dict):
                    diagnostic = diagnostic.get("message") or diagnostic.get("code") or "Request refused"
            except (ValueError, AttributeError):
                diagnostic = "Request refused"
            # Provider diagnostics are visible in the local experiment record.
            diagnostic = " ".join(str(diagnostic).replace(api_key, "[redacted]").split())
            diagnostic = re.sub(r"[A-Za-z0-9_+/=-]{32,}", "[redacted]", diagnostic)[:220]
            raise CreateRejected(
                f"RunPod rejected the allocation (HTTP {response.status_code}): {diagnostic}"
            )
        if response.is_error:
            raise CloudError(
                f"RunPod {method} {path} returned HTTP {response.status_code}."
            )
        return response.json() if response.content else {}

    def pages(self, path, field, params=None):
        items = []
        args = dict(params or {})
        while True:
            page = self.request("GET", path, params=args)
            items.extend(page[field])
            pagination = page["pagination"]
            if not pagination["hasNextPage"]:
                return items
            cursor = pagination["nextCursor"]
            if not cursor or cursor == args.get("cursor"):
                raise CloudError("Incomplete provider pagination.")
            args["cursor"] = cursor

    def gpus(self):
        gpus = self.request(
            "GET",
            "catalog/gpus",
            params={
                "include": "AVAILABILITY",
                "product": "POD",
                "cloud": "SECURE",
                "minCudaVersion": "12.8",
                "minRamPerGpu": 32,
            },
        )["gpus"]
        return sorted(
            [
                {
                    "id": g["id"],
                    "name": g["name"],
                    "vram": g["memory"],
                    "hourly": g["price"]["secure"],
                    "stock": g.get("availability", "UNKNOWN"),
                }
                for g in gpus
                if g["secure"]
                and g["memory"] >= 24
                and (g["price"].get("secure") or 0) > 0
            ],
            key=lambda g: g["hourly"],
        )

    def inventory(self):
        pods = [
            {k: p.get(k) for k in ("id", "name", "status", "cost", "disk", "mounts")}
            for p in self.pages("pods", "pods", {"includeClusterPods": "true"})
        ]
        volumes = [
            {k: v[k] for k in ("id", "name", "size", "type")}
            for v in self.pages("network-volumes", "networkVolumes")
        ]
        for v in volumes:
            v["monthly_estimate"] = (
                round(min(v["size"], 1000) * 0.07 + max(0, v["size"] - 1000) * 0.05, 2)
                if v["type"] == "STANDARD"
                else None
            )
        endpoints = [
            {k: e.get(k) for k in ("id", "name", "workers")}
            for e in self.pages("serverless", "endpoints")
        ]
        return {
            "pods": pods,
            "volumes": volumes,
            "endpoints": endpoints,
            "balance": self.request("GET", "billing/balance"),
        }

    def quote(self, config):
        gpu = next((g for g in self.gpus() if g["id"] == config["gpu"]), None)
        if (
            not gpu
            or gpu["stock"] in ("NONE", "UNKNOWN")
            or gpu["vram"] < (40 if config["method"] == "LoRA" else 24)
        ):
            raise CloudError(
                "Selected GPU lacks confirmed stock or sufficient memory for this method."
            )
        offer = estimate(gpu["hourly"], config["minutes"])
        if offer["max_usd"] > BUDGET:
            raise CloudError("This request exceeds the $20 mission budget.")
        return dict(offer, gpu=gpu["id"], stock=gpu["stock"])

    def create(self, run, pub):
        script = (store.ROOT / "remote" / "boot.sh").read_text(encoding="utf8")
        body = {
            "name": run["resource_name"],
            "cloud": "SECURE",
            "gpu": {
                "id": run["config"]["gpu"],
                "count": 1,
                "minCudaVersion": "12.8",
                "minRamPerGpu": 32,
            },
            "image": IMAGE,
            "disk": DISK,
            "mounts": {"persistent": {"size": VOLUME, "path": "/workspace"}},
            "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": pub, "MODELLAB_DEADLINE": str(int(run["deadline"]))},
            "entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
            "cmd": ["/bin/bash", "-c", script],
        }
        return self.request("POST", "pods", json=body)["id"]

    def pod(self, pid):
        return self.request("GET", "pods/" + pid)

    def stop(self, pid):
        self.request("POST", "pods/" + pid + "/action", json={"action": "stop"})

    def delete(self, pid):
        self.request("DELETE", "pods/" + pid)

    def bill(self, pid, start):
        from datetime import datetime, timezone

        # RunPod places a pod's usage in a UTC-day bucket, starting at midnight.
        date = datetime.fromtimestamp(start, timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        data = self.request(
            "GET", "billing/pods",
            params={"podId": pid, "startTime": date, "endTime": datetime.now(timezone.utc).isoformat()},
        )
        return data["metadata"]["totals"] if data["records"] else None


def transport_environment():
    return {k: v for k, v in os.environ.items() if k.upper() != "RUNPOD_API_KEY"}


def identity():
    path = store.home() / "transport_ed25519"
    if not shutil.which("ssh") or not shutil.which("ssh-keygen"):
        raise CloudError("Install the OpenSSH client before starting a pod.")
    if not path.exists():
        result = subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(path)],
            capture_output=True,
            env=transport_environment(),
        )
        if result.returncode:
            raise CloudError("Could not create the local transport key.")
    public = Path(str(path) + ".pub").read_text().split()
    if len(public) < 2 or public[0] != "ssh-ed25519":
        raise CloudError("Invalid transport public key.")
    return path, " ".join(public[:2])


class Transport:
    def __init__(self, pod):
        endpoint = pod["ssh"]["direct"]
        host, port = endpoint["host"], int(endpoint["port"])
        if (
            not re.fullmatch(r"[a-zA-Z0-9.:-]+", host)
            or host.startswith("-")
            or not 0 < port < 65536
        ):
            raise CloudError("Invalid SSH endpoint returned by provider.")
        path, _ = identity()
        known = (store.home() / ("hosts-" + pod["id"])).as_posix()
        self.args = [
            "ssh",
            "-T",
            "-i",
            str(path),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f'UserKnownHostsFile="{known}"',
            "-o",
            "ConnectTimeout=6",
            "-p",
            str(port),
            "root@" + host,
        ]

    def command(self, cmd, data=None, timeout=20, output=None):
        try:
            r = subprocess.run(
                self.args + [cmd],
                env=transport_environment(),
                input=data,
                stdout=output or subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CloudError("SSH transfer interrupted.") from exc
        if r.returncode:
            raise CloudError("The pod SSH command failed or the pod is starting.")
        return r.stdout
