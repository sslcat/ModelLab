import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from remote.train import equivalent
from studio import archive, lifecycle as life, store
from studio.api import app
from studio.cloud import CloudError, Provider, estimate
from studio.config import Config
from studio.datasets import canon, dispatch_rows, parse, validate


class DatasetTests(unittest.TestCase):
    def test_csv_and_jsonl_same_split_order_invariant(self):
        rows = parse('\ufeffprompt,completion\r\n"a,b",c\r\nd,e\r\nx,y\r\n', "csv")
        a, s = validate(rows + rows[:1])
        b, t = validate(parse("\n".join(canon(r) for r in reversed(rows)), "jsonl"))
        self.assertEqual((a["fingerprint"], s), (b["fingerprint"], t))
        self.assertEqual(a["duplicates"], 1)

    def test_conflict_secret_and_schema(self):
        rows = dispatch_rows()[:4]
        for bad in [
            {"prompt": "x", "completion": "   "},
            {"prompt": "x", "completion": False},
            {"prompt": "x", "completion": "rpa_" + "x" * 30},
            dict(rows[0], completion="conflict"),
            {"a": "b"},
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate(rows + [bad])

    def test_pilot_partition_and_reproducibility(self):
        rows = dispatch_rows()
        meta, s = validate(rows, 42, 0.15)
        self.assertEqual(
            (meta["train"], meta["validation"], meta["rows"]), (204, 36, 240)
        )
        self.assertFalse(
            {r["prompt"] for r in s["train"]} & {r["prompt"] for r in s["validation"]}
        )
        other, _ = validate(rows, 43, 0.15)
        self.assertNotEqual(meta["split_id"], other["split_id"])
        self.assertFalse(
            {r["completion"] for r in rows}
            & {r["completion"] for r in dispatch_rows(True)}
        )
        self.assertEqual(len(dispatch_rows(True)), 48)

    def test_pair_scoring(self):
        self.assertFalse(equivalent('{"qty":0,"qty":1}', '{"qty":1}'))
        self.assertTrue(equivalent('{"b":1,"a":"x"}', '{"a":"x","b":1}'))
        self.assertFalse(equivalent('{"qty":true}', '{"qty":1}'))
        self.assertFalse(equivalent('Answer: {"qty":1}', '{"qty":1}'))


class StoreCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = patch.dict(
            os.environ,
            {"MODELLAB_STATE": str(self.root), "RUNPOD_API_KEY": "fixture-only"},
        )
        env.start()
        self.addCleanup(env.stop)
        for target, value in [
            ("studio.lifecycle.identity", (Path("local"), "ssh-ed25519 fixture")),
            ("studio.archive.artifact_root", self.root / "archive"),
            ("studio.archive.one_drive", None),
        ]:
            m = patch(target, return_value=value)
            m.start()
            self.addCleanup(m.stop)
        store.init()
        self.cfg = Config(dataset=store.datasets()[0]["id"]).model_dump()
        self.doc = store.draft(self.cfg)
        self.rid = self.doc["id"]
        self.p = MagicMock()
        self.p.quote.return_value = dict(
            estimate(0.89, 90), gpu=self.cfg["gpu"], stock="LOW"
        )
        self.p.inventory.return_value = {
            "pods": [],
            "volumes": [],
            "endpoints": [],
            "balance": {"billingMode": "PREPAID", "balanceAmount": 20},
        }

    def approved(self):
        offer = life.quote(self.rid, self.p)
        life.authorize(self.rid, offer["token"], self.p)
        return store.run(self.rid)

    def running(self):
        self.approved()
        self.p.create.return_value = "pod1"
        life.tick(self.rid, self.p)
        self.p.pod.return_value = {
            "name": self.doc["resource_name"],
            "id": "pod1",
            "ssh": {"direct": {"host": "localhost"}},
            "cost": 0.89,
            "actions": ["stop"],
            "status": "RUNNING",
        }
        self.ssh = MagicMock()
        self.ssh.command.return_value = (
            self.doc["bundle_hash"] + "  data.json"
        ).encode()
        life.tick(self.rid, self.p, lambda p: self.ssh)
        self.assertTrue(store.run(self.rid)["launched"])


class LifecycleTests(StoreCase):
    def test_draft_and_quote_do_not_create_and_approval_single_use(self):
        self.approved()
        self.p.create.assert_not_called()
        with self.assertRaises(ValueError):
            life.authorize(self.rid, store.run(self.rid)["quote"]["token"], self.p)
        self.assertEqual(store.run(self.rid)["reserved"], 1.70)

    def test_expired_or_repriced_quote(self):
        offer = life.quote(self.rid, self.p)
        self.p.quote.return_value = dict(self.p.quote.return_value, gpu_hourly=0.99)
        with self.assertRaisesRegex(ValueError, "increased"):
            life.authorize(self.rid, offer["token"], self.p)
        store.update(self.rid, quote=dict(offer, expires=time.time() - 1))
        with self.assertRaisesRegex(ValueError, "expired"):
            life.authorize(self.rid, offer["token"], self.p)
        self.p.create.assert_not_called()

    def test_cumulative_budget(self):
        old = store.draft(self.cfg)
        store.update(old["id"], reserved=19)
        offer = life.quote(self.rid, self.p)
        with self.assertRaisesRegex(ValueError, "budget"):
            life.authorize(self.rid, offer["token"], self.p)

    def test_lost_create_response_reattaches_without_second_post(self):
        self.approved()
        self.p.create.side_effect = CloudError("interrupted")
        with self.assertRaises(CloudError):
            life.tick(self.rid, self.p)
        self.p.pages.return_value = []
        life.tick(self.rid, self.p)
        self.assertEqual(store.run(self.rid)["status"], "uncertain")
        self.p.pages.return_value = [{"name": self.doc["resource_name"], "id": "pod1"}]
        life.tick(self.rid, self.p)
        self.assertEqual(store.run(self.rid)["pod"], "pod1")
        self.p.create.assert_called_once()

    def test_cancel_before_allocation(self):
        self.approved()
        store.update(self.rid, cancel=True)
        life.tick(self.rid, self.p)
        self.assertEqual(store.run(self.rid)["reserved"], 0)
        self.p.create.assert_not_called()

    def test_deadline_stops_compute_keeps_disk(self):
        self.running()
        store.update(self.rid, deadline=time.time() - 1)
        life.tick(self.rid, self.p)
        self.p.stop.assert_called_once_with("pod1")
        self.p.pod.return_value["status"] = "EXITED"
        life.tick(self.rid, self.p)
        self.assertEqual(store.run(self.rid)["status"], "recovery")
        self.p.delete.assert_not_called()

    def test_identity_guard(self):
        self.running()
        self.p.pod.return_value["name"] = "other-project"
        with self.assertRaises(CloudError):
            life.tick(self.rid, self.p)
        self.p.delete.assert_not_called()
        self.p.stop.assert_not_called()

    def test_archival_checksum_gate_then_verified_teardown(self):
        self.running()
        cfg = life.worker_config(store.run(self.rid))
        files = {
            "config.json": canon(cfg).encode(),
            "metrics.jsonl": b'{"step":1,"loss":1}\n',
            "environment.json": b"{}",
            "lengths.json": b"{}",
            "adapter/adapter_config.json": b"{}",
            "adapter/adapter_model.safetensors": b"test fixture",
            "evaluation.json": b'{"base":{},"tuned":{}}',
        }
        manifest = {
            "status": "completed",
            "files": {
                k: {"size": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                for k, v in files.items()
            },
        }
        corrupt = [True]

        def command(cmd, *args, output=None, **kwargs):
            if output:
                name = cmd.split("cat /workspace/modellab/")[1]
                output.write(b"bad" if corrupt[0] else files[name])
            elif "manifest.json" in cmd:
                return canon(manifest).encode()
            return b""

        self.ssh.command.side_effect = command
        with self.assertRaisesRegex(CloudError, "checksum"):
            life.tick(self.rid, self.p, lambda p: self.ssh)
        self.p.delete.assert_not_called()
        corrupt[0] = False
        life.tick(self.rid, self.p, lambda p: self.ssh)
        self.assertTrue(archive.verified(store.run(self.rid)))
        life.tick(self.rid, self.p)
        self.p.delete.assert_called_once_with("pod1")
        self.p.pod.return_value = None
        life.tick(self.rid, self.p)
        self.assertEqual(store.run(self.rid)["status"], "completed")

    def test_cancel_running_propagates_and_missing_adapter_blocks_deletion(self):
        self.running()
        store.update(self.rid, cancel=True)
        self.ssh.command.side_effect = [
            b"",
            b"",
            b'{"status":"completed","files":{"config.json":{}}}',
        ]
        with self.assertRaises(CloudError):
            life.tick(self.rid, self.p, lambda p: self.ssh)
        self.ssh.command.assert_any_call("touch /workspace/modellab/cancel")
        self.p.delete.assert_not_called()

    def test_path_traversal_rejected(self):
        doc = self.doc
        files = {"../outside": {}, "config.json": {}}
        with self.assertRaisesRegex(CloudError, "path"):
            archive.retrieve(doc, {"status": "failed", "files": files}, MagicMock())
        self.assertFalse((self.root / "outside").exists())

    def test_provider_request_shape_and_pagination(self):
        requests = []

        def handler(req):
            requests.append(req)
            if req.method == "POST":
                return httpx.Response(201, json={"id": "pod-id"})
            if "cursor" in req.url.params:
                return httpx.Response(
                    200,
                    json={
                        "pods": [2],
                        "pagination": {"hasNextPage": False, "nextCursor": None},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "pods": [1],
                    "pagination": {"hasNextPage": True, "nextCursor": "next"},
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            provider = Provider(client)
            self.assertEqual(provider.pages("pods", "pods"), [1, 2])
            self.assertEqual(
                provider.create(dict(self.doc, deadline=1234), "ssh-ed25519 public"),
                "pod-id",
            )
            body = json.loads(requests[-1].content)
            self.assertEqual(set(body["env"]), {"PUBLIC_KEY", "MODELLAB_DEADLINE"})
            self.assertEqual(body["ports"], ["22/tcp"])
            self.assertNotIn("fixture-only", requests[-1].content.decode())

    def test_allocation_rejection_diagnostic_redacts_account_key(self):
        def refuse(req):
            return httpx.Response(400, json={"message": "No capacity; fixture-only"})

        with httpx.Client(transport=httpx.MockTransport(refuse)) as client:
            with self.assertRaises(CloudError) as rejected:
                Provider(client).request("POST", "pods", json={})
        self.assertIn("No capacity", str(rejected.exception))
        self.assertNotIn("fixture-only", str(rejected.exception))

    def test_billing_is_scoped_and_empty_records_remain_pending(self):
        def pending(req):
            self.assertEqual(req.url.path, "/v2/billing/pods")
            self.assertEqual(req.url.params["podId"], "study-pod")
            self.assertEqual(req.url.params["startTime"], "1970-01-01T00:00:00+00:00")
            self.assertLess(req.url.params["startTime"], req.url.params["endTime"])
            return httpx.Response(200, json={"records": [], "metadata": {"totals": {}}})

        with httpx.Client(transport=httpx.MockTransport(pending)) as client:
            self.assertIsNone(Provider(client).bill("study-pod", 1000))

    def test_provider_read_failure_is_not_absence(self):
        with httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(503))
        ) as client:
            with self.assertRaises(CloudError):
                Provider(client).pod("id")


class ApiTests(StoreCase):
    def test_import_preview_draft_and_persistence(self):
        with TestClient(app) as client:
            payload = {
                "name": "Example",
                "format": "jsonl",
                "content": "\n".join(canon(x) for x in dispatch_rows()[:4]),
                "license": "MIT",
                "public_data": True,
            }
            headers = {"X-ModelLab": "studio"}
            self.assertEqual(
                client.post("/api/datasets", json=payload).status_code, 403
            )
            ds = client.post("/api/datasets", json=payload, headers=headers)
            self.assertEqual(ds.status_code, 200, ds.text)
            cfg = dict(self.cfg, dataset=ds.json()["id"])
            run = client.post("/api/runs", json=cfg, headers=headers).json()
            self.assertEqual(run["metrics"], [])
            self.assertIsNone(run["actual"])
            self.assertEqual(
                client.get("/api/runs/" + run["id"] + "/export/adapter").status_code,
                404,
            )
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/studio").json()["runs"][0]["config"], cfg)
            with patch("studio.cloud.key", return_value=None):
                result = client.post(
                    "/api/runs/" + run["id"] + "/quote", headers=headers, json={}
                )
                self.assertEqual(result.status_code, 503)

    def test_api_rejects_foreign_origin_and_invalid_hyperparameters(self):
        with TestClient(app) as client:
            data = dict(self.cfg, epochs=0)
            hdr = {"X-ModelLab": "studio", "Origin": "https://example.org"}
            self.assertEqual(
                client.post("/api/runs", headers=hdr, json=data).status_code, 403
            )
            self.assertEqual(
                client.post(
                    "/api/runs", headers={"X-ModelLab": "studio"}, json=data
                ).status_code,
                422,
            )
