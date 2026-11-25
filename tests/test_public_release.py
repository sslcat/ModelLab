import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from research import public_results, reproduce, report
from research.size_design import sources
from studio import lifecycle, store
from studio.demo import EVIDENCE, app, records


class PublicReleaseTests(unittest.TestCase):
    def test_release_tables_report_and_every_prediction(self):
        for name, content in public_results.outputs().items():
            self.assertEqual((EVIDENCE / name).read_bytes(), content)
        self.assertEqual(
            (store.ROOT / "docs/research.md").read_text("utf8"), report.report()
        )

    def test_demo_rejects_writes_without_credentials_database_or_network(self):
        with (
            patch("studio.cloud.key", side_effect=AssertionError),
            patch("studio.cloud.Provider.request", side_effect=AssertionError),
            patch("studio.store.db", side_effect=AssertionError),
            TestClient(app) as c,
        ):
            state = c.get("/api/studio").json()
            self.assertEqual(
                (state["mode"], state["active"], len(state["runs"])), ("demo", 0, 10)
            )
            self.assertAlmostEqual(state["reserved"], 3.621711880317889)
            for method in ("post", "delete", "patch", "put"):
                self.assertEqual(
                    getattr(c, method)("/api/runs/pilot/approve").status_code, 405
                )
            for part in ("development", "locked"):
                for condition in ("tuned", "base"):
                    view = c.get(
                        "/api/research/model-size",
                        params=dict(partition=part, condition=condition),
                    ).json()
                    self.assertEqual(len(view["points"]), 4)
                self.assertEqual(
                    len(
                        c.get(
                            "/api/research/study", params=dict(partition=part)
                        ).json()["ranks"]
                    ),
                    4,
                )
            row = c.get(
                "/api/research/model-size/size-1.5b/cases",
                params=dict(partition="locked", arm="seed_42", failures=True),
            ).json()
            self.assertEqual(row["total"], 126)
            self.assertTrue(all(not r["correct"] for r in row["examples"]))
            self.assertEqual(
                c.get(
                    "/api/research/model-size", params={"benchmark": "historical64"}
                ).json()["points"][0]["seed_rows"][0]["correct"],
                63,
            )
            self.assertEqual(c.get("/api/runs/missing/configuration").status_code, 404)

    def test_reproduction_drafts_match_recorded_sources_with_no_allocation(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"MODELLAB_STATE": tmp}),
            patch("studio.cloud.Provider.request", side_effect=AssertionError),
        ):
            pilot = reproduce.pilot()
            ranks = reproduce.prepare.prepare()
            sizes = reproduce.size_design.prepare()
            self.assertEqual(reproduce.pilot(), pilot)
            for rid in [pilot, *ranks, *sizes]:
                d = store.run(rid)
                self.assertEqual(d["status"], "draft")
                fixture = next(
                    r for r in records().values() if r["config"] == d["config"]
                )
                c = lifecycle.worker_config(d)
                keys = ["bundle_hash", "worker_sha256", "revision"]
                if fixture["study"] != "pilot":
                    keys.extend(["cloud_image", "protocol"])
                for k in keys:
                    self.assertEqual(c[k], fixture["execution"][k])
                lifecycle.check_research_lock(d)
                paths = (
                    sources()
                    if d.get("size_study")
                    else reproduce.pilot_sources()
                    if d.get("pilot_snapshot")
                    else {n: store.ROOT / "remote" / n for n in c["worker_sha256"]}
                )
                self.assertEqual(
                    {
                        n: hashlib.sha256(p.read_bytes()).hexdigest()
                        for n, p in paths.items()
                    },
                    c["worker_sha256"],
                )
