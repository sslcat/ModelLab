import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from remote import launch
from studio import archive, store
from studio.cloud import CreateRejected
from studio import lifecycle
from test_studio import StoreCase


class RecoveryTests(StoreCase):
    def test_explicit_capacity_refusal_releases_reservation(self):
        self.approved()
        self.p.create.side_effect = CreateRejected("Capacity request refused.")
        lifecycle.tick(self.rid, self.p)
        self.assertEqual(store.run(self.rid)["status"], "failed")
        self.assertEqual(store.run(self.rid)["reserved"], 0)
        from studio.api import display_run

        self.assertNotIn("estimated_usage", display_run(store.run(self.rid)))

    def test_staging_can_be_recreated_after_verified_transfer(self):
        from studio.datasets import dispatch_rows

        args = ("Public import", "MIT", dispatch_rows()[:6], 17, 0.2)
        meta = store.save_dataset(*args)
        cfg = dict(self.cfg, dataset=meta["id"])
        doc = store.draft(cfg)
        offer = lifecycle.quote(doc["id"], self.p)
        store.clear_staging(meta["id"])
        with self.assertRaisesRegex(ValueError, "Re-import"):
            lifecycle.authorize(doc["id"], offer["token"], self.p)
        self.assertEqual(store.save_dataset(*args)["id"], meta["id"])
        self.assertEqual(
            store.dataset(meta["id"])[0]["bundle_hash"], doc["bundle_hash"]
        )

    def test_checkpoint_manifest_and_credentials_exclusion(self):
        remote = self.root / "remote"
        cp = remote / "checkpoints" / "checkpoint-12"
        cp.mkdir(parents=True)
        (cp / "adapter_config.json").write_text("{}")
        (cp / "adapter_model.safetensors").write_bytes(b"fixture")
        (cp / "optimizer.pt").write_bytes(b"optimizer excluded from archive")
        (remote / "model.safetensors").write_bytes(b"base weights excluded")
        (remote / "config.json").write_text("{}")
        (remote / "metrics.jsonl").write_bytes(b"")
        with patch.object(launch, "ROOT", remote):
            launch.finish("cancelled")
        mf = json.loads((remote / "manifest.json").read_text())
        self.assertEqual(
            set(mf["files"]),
            {
                "config.json",
                "metrics.jsonl",
                "checkpoints/checkpoint-12/adapter_config.json",
                "checkpoints/checkpoint-12/adapter_model.safetensors",
            },
        )
        doc = self.approved()
        archive.recovery_record(doc)
        text = (self.root / "archive" / self.rid / "recovery.json").read_text()
        self.assertNotIn(doc["quote"]["token"], text)
        self.assertNotIn("balance", text)
        self.assertNotIn("fixture-only", text)

    def test_archive_disappearing_blocks_termination(self):
        self.running()
        store.update(
            self.rid,
            status="deleting",
            archived=True,
            archive=str(self.root / "missing"),
        )
        lifecycle.tick(self.rid, self.p)
        self.assertEqual(store.run(self.rid)["status"], "stopping")
        self.p.delete.assert_not_called()

    def test_prepaid_balance_and_concurrent_allocation(self):
        offer = lifecycle.quote(self.rid, self.p)
        self.p.inventory.return_value["balance"]["balanceAmount"] = 0
        with self.assertRaisesRegex(ValueError, "balance"):
            lifecycle.authorize(self.rid, offer["token"], self.p)
        self.p.inventory.return_value["balance"]["balanceAmount"] = 20
        self.approved()
        other = store.draft(self.cfg)
        quote = lifecycle.quote(other["id"], self.p)
        with self.assertRaisesRegex(ValueError, "existing active"):
            lifecycle.authorize(other["id"], quote["token"], self.p)


class GuardTests(unittest.TestCase):
    def test_detached_guard_invokes_stop_for_own_pod(self):
        bash = (
            Path("C:/Program Files/Git/bin/bash.exe")
            if os.name == "nt"
            else Path(shutil.which("bash") or "/bin/bash")
        )
        if not bash.is_file():
            self.skipTest(
                "A POSIX shell is required for the boot guard integration check."
            )
        boot = (store.ROOT / "remote/boot.sh").read_text(encoding="utf8")
        guard = boot.split("\n(\n", 1)[1].split("\n) &", 1)[0]
        script = (
            'set -eu\nguard_deadline=0\nRUNPOD_POD_ID=fixture-pod\nrunpodctl(){ echo "$*" > observed; }\nsleep(){ exit 0; }\n'
            + guard
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "guard.sh"
            path.write_bytes(script.encode())
            subprocess.run([str(bash), "guard.sh"], cwd=temp, timeout=45, check=True)
            self.assertEqual(
                (Path(temp) / "observed").read_text().strip(), "pod stop fixture-pod"
            )
