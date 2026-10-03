# Copyright (c) 2026 Torchcast AI. MIT licence (see LICENSE).
"""Merge the results.jsonl files of runs that split one suite (e.g. a forward and a --reverse di_batch.py run) into one
file for `python -m decision_index score`: one record per run_id, the first non-error record in argument order.

    python eval/merge_results.py OUT.jsonl RESULTS_A.jsonl RESULTS_B.jsonl [...]
"""
import json, sys


def main():
    out, inputs = sys.argv[1], sys.argv[2:]
    best = {}
    for path in inputs:
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            prev = best.get(r["run_id"])
            if prev is None or (prev["status"] == "error" and r["status"] != "error"):
                best[r["run_id"]] = r
    with open(out, "w") as f:
        for r in best.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(best), "errors": sum(r["status"] == "error" for r in best.values())}))


if __name__ == "__main__":
    main()
