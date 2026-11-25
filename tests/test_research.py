import json
import os
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch
from fastapi.testclient import TestClient
from remote import controlled
from remote.receipt_scores import score, summarize
from research import receipts, prepare
from studio import store
from studio.api import app
from studio.cloud import CloudError, Provider
from studio.research import analysis, verify_evaluation


class ScoringTests(unittest.TestCase):
    def test_typed_schema_and_unfiltered_denominators(self):
        expected = '{"item":"bolts","quantity":1,"site":"Dock"}'
        examples = [
            {"expected": expected, "prediction": v}
            for v in [
                expected,
                "\x60\x60\x60json\n" + expected + "\n\x60\x60\x60",
                expected.replace(":1", ":true"),
                expected.replace('"Dock"', '"Yard"'),
                expected.replace('"item":', '"item":"nuts","item":'),
                expected[:-1] + ',"extra":1}',
                "NaN",
            ]
        ]
        out = summarize(examples)
        self.assertEqual(
            (out["n"], out["valid"], out["schema"], out["exact"], out["content"]),
            (7, 5, 2, 1, 2),
        )
        self.assertEqual(out["fields"], {"item": 4, "quantity": 3, "site": 3})
        self.assertIsNone(out["latency_p50_ms"])
        self.assertFalse(score("{}", expected)["schema"])

    def test_fence_diagnostic_keeps_field_credit_separate(self):
        expected = '{"item":"bolts","quantity":3,"site":"Bay"}'
        wrapped = (
            chr(96) * 3
            + "json\n"
            + expected.replace('"bolts"', '"nuts"')
            + "\n"
            + chr(96) * 3
        )
        doc = {
            "evaluation": {
                "base": {
                    "examples": [
                        {
                            "prompt": "receipt",
                            "expected": expected,
                            "prediction": wrapped,
                            "correct": False,
                        }
                    ],
                    "correct": 0,
                    "total": 1,
                }
            }
        }
        view = analysis(doc)
        self.assertEqual(
            view["summary"]["fields"], {"item": 0, "quantity": 0, "site": 0}
        )
        self.assertEqual(
            view["summary"]["content_fields"], {"item": 0, "quantity": 1, "site": 1}
        )
        self.assertEqual(
            view["examples"][0]["content_scores"]["errors"], ["Wrong item"]
        )

    def test_generic_imported_prediction_remains_inspectable(self):
        d = {
            "evaluation": {
                "base": {
                    "examples": [
                        {
                            "prompt": "a",
                            "prediction": "x",
                            "expected": "x",
                            "correct": True,
                        }
                    ],
                    "total": 1,
                    "correct": 1,
                }
            }
        }
        result = analysis(d)
        self.assertIsNone(result["summary"])
        self.assertEqual(result["correct"], 1)


class CorpusTests(unittest.TestCase):
    def test_split_lock_and_crossed_generalization_design(self):
        data = receipts.locked_bundle()
        self.assertEqual(data, receipts.build())
        self.assertEqual(
            [len(data[k]) for k in ("train", "validation", "test")], [384, 64, 64]
        )

        def pairs(rows):
            return {
                (v["item"], v["site"])
                for r in rows
                for v in [json.loads(r["completion"])]
            }

        trained = pairs(data["train"])
        novel = {}
        for k in ("validation", "test"):
            novel[k] = pairs([r for r in data[k] if r["unseen_pair"]])
            self.assertTrue(novel[k].isdisjoint(trained))
            self.assertTrue(
                pairs([r for r in data[k] if not r["unseen_pair"]]) <= trained
            )
            self.assertEqual(
                set(
                    Counter(
                        (r["unseen_pair"], r["unseen_template"]) for r in data[k]
                    ).values()
                ),
                {16},
            )
            self.assertEqual(
                len({(r["unseen_pair"], r["unseen_template"]) for r in data[k]}), 4
            )
            self.assertTrue(
                all(
                    r["unseen_template"] == (r["template"] not in receipts.TRAIN)
                    for r in data[k]
                )
            )
        self.assertTrue(novel["validation"].isdisjoint(novel["test"]))
        self.assertTrue(set(receipts.TEST).isdisjoint(receipts.DEV))
        rows = sum(data.values(), [])
        for field in ("id", "prompt", "completion"):
            self.assertEqual(len({r[field] for r in rows}), len(rows))
        preview = receipts.metadata(data)["preview"]
        self.assertTrue(
            all(p["prompt"] in {r["prompt"] for r in data["train"]} for p in preview)
        )

    def test_all_arms_order_and_archive_gate(self):
        data = receipts.locked_bundle()
        calls = []

        def generate(r, a):
            calls.append((r["id"], a))
            return {
                **r,
                "expected": r["completion"],
                "prediction": r["completion"],
                "correct": True,
                "latency_ms": 2.0,
            }

        cfg = {"protocol": controlled.protocol()}
        report = controlled.collect(data, generate, lambda name, report: None)
        self.assertEqual(len(calls), 645)
        self.assertEqual(len(set(calls[5:])), 640)
        self.assertEqual(
            [a for _, a in calls[10:15]],
            list(controlled.ARMS[1:] + controlled.ARMS[:1]),
        )
        verify_evaluation(cfg, report)
        view = analysis({"evaluation": report}, "locked", "base_constrained")
        self.assertEqual(
            (view["summary"]["unseen_n"], view["summary"]["latency_p95_ms"]), (32, 2)
        )
        self.assertEqual(len(view["groups"]), 4)
        report["partitions"]["locked"]["tuned"]["examples"].pop()
        with self.assertRaises(CloudError):
            verify_evaluation(cfg, report)


class DraftTests(unittest.TestCase):
    def test_idempotent_free_rank_drafts_and_export(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"MODELLAB_STATE": tmp}),
            patch.object(
                Provider, "request", side_effect=AssertionError("unexpected network")
            ),
        ):
            ids = prepare.prepare()
            self.assertEqual(ids, prepare.prepare())
            from studio import lifecycle

            with patch.object(lifecycle, "worker_sources", return_value={}):
                with self.assertRaisesRegex(ValueError, "worker changed"):
                    lifecycle.check_research_lock(store.run(ids[0]))
            self.assertEqual(len(store.runs()), 4)
            with TestClient(app) as client:
                configs = []
                for key in ids:
                    response = client.get(f"/api/runs/{key}/configuration")
                    self.assertEqual(response.status_code, 200)
                    c = response.json()
                    configs.append(c)
                    self.assertEqual(c["alpha"] / c["rank"], 2)
                    self.assertEqual(c["protocol"]["schema"], controlled.SCHEMA)
                    self.assertIsNone(c["deadline"])
                    self.assertIsNone(store.run(key)["quote"])
                    self.assertEqual(store.run(key)["status"], "draft")
                ignore = {"rank", "alpha", "name", "experiment"}
                self.assertEqual(
                    len(
                        {
                            json.dumps(
                                {k: v for k, v in c.items() if k not in ignore},
                                sort_keys=True,
                            )
                            for c in configs
                        }
                    ),
                    1,
                )
                self.assertEqual([c["rank"] for c in configs], [4, 8, 16, 32])
                self.assertEqual(
                    client.get(f"/api/runs/{ids[0]}/analysis").json()["examples"], []
                )
