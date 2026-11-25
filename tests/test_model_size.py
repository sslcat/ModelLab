import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from research import receipts, size_corpus, size_design, size_results
from research.size_worker import controlled
from studio import archive, lifecycle, store
from studio.api import app
from studio.cloud import Provider
from studio.datasets import canon


class SizeStudyTests(unittest.TestCase):
    def test_corpus_lock_and_cross_partition_identity(self):
        a = size_corpus.locked_bundle()
        old = receipts.locked_bundle()
        self.assertEqual(a["train"], old["train"])
        self.assertEqual(
            [len(a[k]) for k in ("train", "validation", "test")], [384, 64, 256]
        )
        parts = [
            old["train"] + old["validation"] + old["test"],
            a["validation"],
            a["test"],
        ]
        for key in ("prompt", "completion", "id"):
            sets = [{r[key] for r in rows} for rows in parts]
            self.assertFalse(
                sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]
            )
        historical = {r["template"] for r in parts[0]}
        self.assertFalse(historical & set(size_corpus.TEST))
        self.assertFalse(set(size_corpus.DEV) & set(size_corpus.TEST))

        def pairs(rows):
            return {
                (v["item"], v["site"])
                for r in rows
                for v in [json.loads(r["completion"])]
            }

        seen = pairs(a["train"])
        for k, n in (("validation", 32), ("test", 128)):
            novel = [r for r in a[k] if r["unseen_pair"]]
            self.assertEqual(len(novel), n)
            self.assertFalse(pairs(novel) & seen)
        self.assertFalse((pairs(a["test"]) - seen) & (pairs(a["validation"]) - seen))
        self.assertEqual(sum(r["difficulty"] == "long_context" for r in a["test"]), 64)
        self.assertGreater(
            min(
                len(r["prompt"].split())
                for r in a["test"]
                if r["difficulty"] == "long_context"
            ),
            100,
        )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for patcher in (
            patch.dict(os.environ, {"MODELLAB_STATE": self.tmp.name}),
            patch.object(
                Provider,
                "request",
                side_effect=AssertionError("No cloud calls in tests"),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_free_drafts_staging_and_price_guard(self):
        keys = size_design.prepare()
        self.assertEqual(keys, size_design.prepare())
        docs = [store.run(k) for k in keys]
        self.assertTrue(
            all(
                d["status"] == "draft" and not d.get("quote") and d["reserved"] == 0
                for d in docs
            )
        )
        self.assertEqual(
            [size_design.check(d)["max_usd"] for d in docs], [1.16, 1.43, 1.7, 2.24]
        )
        self.assertEqual(
            sum(size_design.check(d)["expected_trainable"] for d in docs), 2998272
        )
        with self.assertRaises(ValueError):
            store.draft(docs[0]["config"])
        with self.assertRaises(ValueError):
            size_design.check(dict(docs[0], config=dict(docs[0]["config"], rank=8)))
        with TestClient(app) as c:
            detail = c.get("/api/research/model-size").json()
            self.assertEqual(detail["points"], [])
            self.assertEqual(len(detail["sessions"]), 4)
            self.assertEqual(
                c.get(
                    f"/api/research/model-size/{keys[0]}/cases?partition=locked&arm=base"
                ).status_code,
                404,
            )
            cfg = c.get(f"/api/runs/{keys[0]}/configuration").json()
            self.assertEqual(cfg["protocol"]["seeds"], [42, 73, 101])
            self.assertIsNone(cfg["deadline"])
        ssh = MagicMock()
        ssh.command.side_effect = lambda command, *a, **kw: (
            (docs[0]["bundle_hash"] + " data.json").encode()
            if command.startswith("sha256sum")
            else b""
        )
        lifecycle.stage(docs[0], ssh)
        staged = {
            c.args[0].split("cat > ")[1].split(".tmp")[0].split("/")[-1]: c.args[1]
            for c in ssh.command.call_args_list
            if "cat > " in c.args[0]
        }
        for name, sha in cfg["worker_sha256"].items():
            if name != "boot.sh":
                self.assertEqual(hashlib.sha256(staged[name]).hexdigest(), sha)
        # A stale/pricier quote fails before identity creation or provisioning.
        p = MagicMock()
        p.inventory.return_value = {
            "pods": [],
            "volumes": [],
            "endpoints": [],
            "balance": {},
        }
        p.quote.return_value = {"max_usd": 1.60, "gpu_hourly": 0.95}
        q = lifecycle.quote(keys[1], p)
        with self.assertRaisesRegex(ValueError, "allowance"):
            lifecycle.authorize(keys[1], q["token"], p)
        p.create.assert_not_called()

    def test_full_protocol_archive_and_cross_model_math(self):
        keys = size_design.prepare()
        data = size_corpus.locked_bundle()
        d = store.run(keys[0])
        cfg = lifecycle.worker_config(d)
        report = {
            "protocol": controlled.protocol(),
            "partitions": {},
            "split": "test fixture",
            "limitations": "fixture",
        }
        calls = []

        def fake(row, arm):
            prediction = row["completion"] if arm != "base" else "{}"
            calls.append((row["id"], arm))
            return {
                "id": row["id"],
                "template": row["template"],
                "difficulty": row.get("difficulty", "warm-up"),
                "unseen_pair": row["unseen_pair"],
                "unseen_template": row["unseen_template"],
                "prompt": row["prompt"],
                "prediction": prediction,
                "expected": row["completion"],
                "correct": arm != "base",
                "latency_ms": 4.0,
                "input_tokens": 22,
                "generated_tokens": 5,
                "generation_peak_mib": 125,
            }

        for s in controlled.SEEDS:
            controlled.collect(data, s, fake, lambda r: None, report)
        self.assertEqual(len(calls), 1284)
        size_results.verify_predictions(report, data)
        bad = copy.deepcopy(report)
        bad["partitions"]["locked"]["base"]["examples"][0]["id"] = "bad"
        with self.assertRaisesRegex(ValueError, "identity"):
            size_results.verify_predictions(bad, data)
        root = Path(self.tmp.name) / "archive"
        count = size_design.check(d)["expected_trainable"]
        payloads = {"evaluation.json": canon(report).encode()}
        for s in controlled.SEEDS:
            prefix = size_results.prefix(s)
            meta = {
                "t": {"shape": [count], "dtype": "F32", "data_offsets": [0, 4 * count]}
            }
            header = canon(meta).encode()
            history = {
                "train_runtime": 5,
                "train_tokens_per_second": 100,
                "peak_allocated_mib": 100,
                "eval_loss": 0.1,
            }
            for name, b in (
                ("config.json", canon(dict(cfg, seed=s)).encode()),
                ("environment.json", canon({"trainable": count}).encode()),
                ("metrics.jsonl", canon(history).encode()),
                ("lengths.json", b"{}"),
                (
                    "adapter/adapter_model.safetensors",
                    len(header).to_bytes(8, "little") + header + bytes(4 * count),
                ),
                (
                    "adapter/adapter_config.json",
                    canon(
                        {
                            "r": 4,
                            "lora_alpha": 8,
                            "target_modules": ["q_proj", "v_proj"],
                        }
                    ).encode(),
                ),
            ):
                payloads[prefix + name] = b
        m = {
            "status": "completed",
            "files": {
                k: {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
                for k, b in payloads.items()
            },
        }
        ssh = MagicMock()
        ssh.command.side_effect = lambda cmd, timeout, output: output.write(
            payloads[cmd.split("/workspace/modellab/")[1]]
        )
        with patch("studio.archive.artifact_root", return_value=root):
            archived = archive.retrieve(d, m, ssh)
        completed = dict(
            d,
            **archived,
            status="completed",
            terminated=True,
            actual={"totalAmount": 0.7},
        )
        self.assertTrue(archive.verified(completed))
        v = size_results.view([completed])
        p = v["points"][0]
        self.assertEqual(
            (p["accuracy"], p["seed_sd"], p["trainable"], p["cost"]), (1, 0, count, 0.7)
        )
        self.assertEqual(p["seed_rows"][0]["own_base"]["gains"], 256)
        self.assertEqual(
            size_results.view([completed], condition="base")["points"][0][
                "adapter_bytes"
            ],
            0,
        )
        # A separate fixture session supplies the larger-model reference. One seed loses a note.
        small = copy.deepcopy(completed)
        row = small["evaluation"]["partitions"]["locked"]["seed_73"]["examples"][0]
        row.update(correct=False, prediction="{}")
        larger = dict(
            completed,
            id="reference",
            config=dict(completed["config"], model=size_design.MODELS[3]["model"]),
        )
        cross = size_results.view([small, larger])
        self.assertEqual(len(cross["points"]), 2)
        self.assertEqual(len(cross["pairs"]), 6)
        self.assertGreater(cross["points"][0]["seed_sd"], 0)
        self.assertEqual(
            next(
                x
                for x in cross["pairs"]
                if x["model"] == small["config"]["model"] and x["arm"] == "seed_73"
            )["gains"],
            255,
        )
        from research.size_package import export

        with (
            patch("research.size_package.one_drive", return_value=Path(self.tmp.name)),
            self.assertRaisesRegex(ValueError, "verified"),
        ):
            export(measured=True)
        matrix = [
            dict(
                completed,
                id="fixture-" + str(i),
                config=dict(completed["config"], model=m["model"]),
            )
            for i, m in enumerate(size_design.MODELS[:4])
        ]
        # Provider-free report integration: four session records reuse the validated fixture files.
        with (
            patch("research.size_package.one_drive", return_value=Path(self.tmp.name)),
            patch("research.size_package.store.runs", return_value=matrix),
            patch("research.size_package.archive.verified", return_value=True),
        ):
            report_folder = export(measured=True)
            snapshot = json.loads(
                (report_folder / "comparisons.json").read_text(encoding="utf8")
            )
            self.assertEqual(len(snapshot["locked-tuned"]["points"]), 4)
            self.assertEqual(len(snapshot["locked-tuned"]["pairs"]), 12)
            self.assertTrue((report_folder / "difficulty.csv").is_file())
        self.assertFalse(archive.allowed_file(d, "seeds/73/../../private.key"))
        self.assertFalse(archive.allowed_file({}, "seeds/73/environment.json"))
        self.assertEqual(set(size_results.expected_files()), set(payloads))
        mf = root / d["id"] / "manifest.json"
        original = mf.read_bytes()
        mf.write_text(json.dumps(dict(m, files={})))
        self.assertFalse(archive.verified(completed))
        mf.write_bytes(original)
        (root / d["id"] / "seeds/73/adapter/adapter_model.safetensors").write_bytes(
            b"broken"
        )
        self.assertFalse(archive.verified(completed))

    def test_interrupted_session_preserves_completed_and_partial_adapters(self):
        import importlib.util
        import sys

        from remote import launch as legacy_launcher

        with patch.dict(
            sys.modules, {"controlled": controlled, "launcher_core": legacy_launcher}
        ):
            spec = importlib.util.spec_from_file_location(
                "size_launcher_fixture", size_design.sources()["launch.py"]
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        root = Path(self.tmp.name)
        (root / "adapter").mkdir()
        partial = root / "seeds/73/checkpoints/checkpoint-16"
        partial.mkdir(parents=True)
        for folder in (root / "adapter", partial):
            (folder / "adapter_config.json").write_text("{}")
            (folder / "adapter_model.safetensors").write_bytes(b"unit test only")
        (root / "config.json").write_text("{}")
        with patch.object(module, "ROOT", root):
            module.finish("timed_out", "Unit-test deadline")
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "timed_out")
        self.assertIn("adapter/adapter_model.safetensors", manifest["files"])
        self.assertIn("seeds/73/adapter/adapter_model.safetensors", manifest["files"])
        self.assertFalse(
            any(name.startswith("seeds/101") for name in manifest["files"])
        )

    def test_serialized_matrix_audit_checks_every_seed_and_extent(self):
        from research.size_diagnostics import adapter_inventory

        root = Path(self.tmp.name)
        for seed in controlled.SEEDS:
            adapter = (
                root
                / size_results.prefix(seed)
                / "adapter"
                / "adapter_model.safetensors"
            )
            adapter.parent.mkdir(parents=True)
            header = {}
            offset = 0
            for target, width in (("q_proj", 8), ("v_proj", 4)):
                for side, shape in (("A", [4, 8]), ("B", [width, 4])):
                    length = 4 * shape[0] * shape[1]
                    name = f"base_model.model.model.layers.0.self_attn.{target}.lora_{side}.weight"
                    header[name] = {
                        "shape": shape,
                        "dtype": "F32",
                        "data_offsets": [offset, offset + length],
                    }
                    offset += length
            raw = json.dumps(header).encode()
            adapter.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(offset))
        doc = {"archive": str(root), "id": "fixture"}
        model = {"geometry": [8, 1, 2, 1], "model": "test fixture"}
        rows = adapter_inventory(doc, model)
        self.assertEqual([r["tensor_parameters"] for r in rows], [112, 112, 112])
        with adapter.open("ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(ValueError, "trailing bytes"):
            adapter_inventory(doc, model)

    def test_portable_release_rejects_changed_bytes_and_outside_files(self):
        from research.verify_size_release import verified_file

        root = Path(self.tmp.name)
        inside = root / "release"
        inside.mkdir()
        file = inside / "predictions.json"
        file.write_bytes(b"original")
        expected = hashlib.sha256(file.read_bytes()).hexdigest()
        self.assertEqual(verified_file(inside, file.name, expected), file.resolve())
        file.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            verified_file(inside, file.name, expected)
        (root / "unrelated.json").write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "Invalid archive path"):
            verified_file(inside, "../unrelated.json", expected)

    def test_frontier_ties_missing_values_and_conflicting_objectives(self):
        pts = [
            {"id": s, "accuracy": a, "cost": c}
            for s, a, c in [
                ("a", 0.9, 1),
                ("b", 0.9, 1),
                ("c", 0.8, 1),
                ("d", 1, 2),
                ("e", 1, None),
            ]
        ]
        self.assertEqual(size_results.frontier(pts, "cost"), ["a", "b", "d"])
        for p, latency in zip(pts, [2, 2, 1, 3, 0], strict=True):
            p["latency"] = latency
        self.assertEqual(
            size_results.frontier(pts, ("cost", "latency")), ["a", "b", "c", "d"]
        )
