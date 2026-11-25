import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from research.study_results import matched, paired_counts, resource_metrics, study_view


class PairedStudyTests(unittest.TestCase):
    def test_history_export_keeps_sparse_validation_measurements(self):
        import csv
        import io
        from research.export_results import csv_bytes

        data = csv_bytes([{"step": 1, "loss": 0.2}, {"step": 2, "eval_loss": 0.1}])
        rows = list(csv.DictReader(io.StringIO(data.decode("utf8"))))
        self.assertEqual(rows[0], {"step": "1", "loss": "0.2", "eval_loss": ""})
        self.assertEqual(rows[1], {"step": "2", "loss": "", "eval_loss": "0.1"})

    def test_files_require_current_completed_study(self):
        from studio.api import published_study

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("research.export_results.result_directory", return_value=Path(tmp)),
        ):
            docs = [
                {"id": str(r), "status": "completed", "research_design": True}
                for r in (4, 8, 16, 32)
            ]
            self.assertIsNone(published_study(docs))
            evidence = Path(tmp) / "evidence.json"
            evidence.write_text(
                json.dumps({"runs": [{"experiment": x["id"]} for x in docs]})
            )
            self.assertEqual(published_study(docs), Path(tmp))
            self.assertIsNone(published_study(docs[:3]))
            self.assertIsNone(
                published_study([*docs[:3], dict(docs[3], status="running")])
            )
            self.assertIsNone(published_study([*docs[:3], dict(docs[3], id="new")]))

    def test_equal_counts_with_discordance(self):
        p = paired_counts(3, 3, 64)
        self.assertEqual((p["delta"], p["exact_p"], p["ties"]), (0, 1, 58))
        self.assertAlmostEqual(-p["lower"], p["upper"])
        q = paired_counts(5, 0, 64)
        self.assertEqual(q["exact_p"], 0.0625)
        self.assertLess(q["lower"], q["delta"])
        self.assertLess(q["delta"], q["upper"])
        zero = paired_counts(0, 0, 64)
        self.assertLess(zero["lower"], 0)
        self.assertGreater(zero["upper"], 0)

    def test_pair_by_identity_and_distinguish_format(self):
        ref = '{"item":"bolts","quantity":4,"site":"Dock"}'
        a = [
            {"id": str(i), "prompt": str(i), "expected": ref, "prediction": ref}
            for i in range(2)
        ]
        fence = chr(96) * 3
        b = [dict(a[1], prediction=fence + "json\n" + ref + "\n" + fence), a[0]]
        self.assertEqual(matched(a, b)["losses"], 1)
        self.assertEqual(matched(a, b, "content")["ties"], 2)
        with self.assertRaises(ValueError):
            matched(a, [b[0], b[0]])
        with self.assertRaises(ValueError):
            matched(a, [dict(b[0], expected="{}"), b[1]])

    def test_draft_has_no_telemetry_or_posted_cost(self):
        draft = {
            "id": "r",
            "status": "draft",
            "config": {"rank": 4, "alpha": 8},
            "research_design": "v",
        }
        r = study_view([draft])["ranks"][0]
        self.assertIsNone(r["summary"])
        self.assertIsNone(r["resources"]["posted_usd"])
        self.assertIsNone(resource_metrics(draft)["trainable"])
        with self.assertRaises(ValueError):
            study_view([draft], partition="unknown")
