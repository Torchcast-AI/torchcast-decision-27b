"""eval/merge_results.py keeps one record per run_id, preferring a non-error record (standard library only)."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class MergeResults(unittest.TestCase):
    def test_merge(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "a.jsonl").write_text("\n".join(json.dumps(r) for r in [
                {"run_id": "1", "status": "ok", "src": "a"},
                {"run_id": "2", "status": "error", "src": "a"},
            ]) + "\n{truncated")
            (d / "b.jsonl").write_text("\n".join(json.dumps(r) for r in [
                {"run_id": "1", "status": "ok", "src": "b"},
                {"run_id": "2", "status": "ok", "src": "b"},
                {"run_id": "3", "status": "error", "src": "b"},
            ]) + "\n")
            out = subprocess.run([sys.executable, str(ROOT / "eval" / "merge_results.py"), str(d / "m.jsonl"),
                                  str(d / "a.jsonl"), str(d / "b.jsonl")], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(out.stdout), {"records": 3, "errors": 1})
            got = {r["run_id"]: (r["status"], r["src"]) for r in map(json.loads, (d / "m.jsonl").read_text().splitlines())}
            self.assertEqual(got, {"1": ("ok", "a"), "2": ("ok", "b"), "3": ("error", "b")})


if __name__ == "__main__":
    unittest.main()
