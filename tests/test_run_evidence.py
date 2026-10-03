"""The score files in runs/ are complete and match the numbers stated in README.md (standard library only)."""
import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "di-0.2.1-h100"
AREAS = {"knowledge": 45.58, "language": 73.81, "retrieval": 67.54, "tools": 84.58, "arts": 51.51}


class RunEvidence(unittest.TestCase):
    def setUp(self):
        self.index = json.loads((RUN / "index.json").read_text())
        self.scores = json.loads((RUN / "scores.json").read_text())
        self.summary = json.loads((RUN / "benchmark-summary.json").read_text())

    def test_edition_and_completeness(self):
        self.assertEqual(self.index["edition"], "0.2.1")
        self.assertEqual(self.scores["edition"], "0.2.1")
        self.assertTrue(self.scores["complete"])
        self.assertEqual(self.scores["counts"], {"ok": self.scores["completed"]})
        self.assertEqual(self.summary["counts"], {"ok": 155390})
        for a in self.index["areas"]:
            self.assertEqual(a["coverage"], 1.0, a["id"])
        for b in self.summary["benchmarks"]:
            self.assertEqual(b["answered"], b["requests"], b["dataset"])
            self.assertEqual(b["errors"], 0, b["dataset"])

    def test_headline_numbers(self):
        self.assertEqual(self.index["index"], 65.0)
        self.assertEqual(self.index["raw_index"], 73.21)
        self.assertEqual(self.scores["decision_index"], 65.0)
        got = {a["id"]: round(100 * a["skill"], 2) for a in self.index["areas"]}
        self.assertEqual(got, AREAS)

    def test_readme_matches(self):
        readme = (ROOT / "README.md").read_text()
        row = re.search(r"^\| \*\*Torchcast Decision 27B\*\* \|(.*)$", readme, re.M).group(1)
        nums = [float(x) for x in re.findall(r"\d+\.\d+", row)]
        self.assertEqual(nums, [65.0, *AREAS.values()])

    def test_engine_label(self):
        self.assertEqual(self.scores["engine"], "torchcast-decision-27b")
        self.assertEqual(self.summary["engine"], "torchcast-decision-27b")


if __name__ == "__main__":
    unittest.main()
