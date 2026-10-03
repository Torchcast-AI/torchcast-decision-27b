# Copyright (c) 2026 Torchcast AI. MIT licence (see LICENSE).
"""Decision Index bulk run of a torchcast_decision model folder with decide_batch, writing results.jsonl in the format of
the official kit (https://github.com/apolinario/decision-index), which then scores it with `python -m decision_index score`.

    python eval/di_batch.py MODEL_DIR ROWS.jsonl.gz OUT_DIR [--chunk 64] [--shard i/n] [--reverse]

MODEL_DIR is a local model folder that ships `torchcast_decision/` (this model's Hugging Face repository).  ROWS is the
suite's request rows, e.g. `cat suite-0.2/selected-rows.jsonl.gz suite-0.2/added-rows.jsonl.gz > all-rows.jsonl.gz`.
The kit must be importable (`pip install -e` its checkout) or its checkout named in DECISION_INDEX_DIR.

Rows are batched `--chunk` requests at a time; a failing batch falls back to one decide() per request. total_wall_ms is
the amortised batch time per request (fine for accuracy; latency must be measured separately with the server).
Resumable: run_ids already in OUT_DIR/results.jsonl with a non-error status are skipped.  Two GPUs can split one run:
one forward, one with --reverse (longest rows first) into separate directories; merge their results by run_id.
"""
import argparse, gzip, json, os, sys, time, traceback
from datetime import datetime, timezone

if os.environ.get("DECISION_INDEX_DIR"):
    sys.path.insert(0, os.environ["DECISION_INDEX_DIR"])
from decision_index.engines.base import validate

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("rows"); ap.add_argument("out")
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--limit", type=int)
ap.add_argument("--reverse", action="store_true", help="longest rows first (a second GPU working from the other end)")
a = ap.parse_args()
sys.path.insert(0, a.model)
from torchcast_decision import TorchcastDecision
from torchcast_decision.model import choice_confidence

si, sn = map(int, a.shard.split("/"))
os.makedirs(a.out, exist_ok=True)
res_path = os.path.join(a.out, "results.jsonl")
done = set()
if os.path.exists(res_path):
    for line in open(res_path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["status"] != "error":
            done.add(r["run_id"])
rows = []
with gzip.open(a.rows, "rt") as f:
    for k, line in enumerate(f):
        if k % sn != si or not line.strip():
            continue
        r = json.loads(line)
        if r["_evaluation"]["run_id"] not in done:
            rows.append(r)
if a.limit:
    rows = rows[:a.limit]
print(f"shard {a.shard}: {len(rows)} rows to run, {len(done)} done", flush=True)
# length-sort so batches pad little
rows.sort(key=lambda r: len(json.dumps(r["state"])) + len(json.dumps(r["questions"])), reverse=a.reverse)

m = TorchcastDecision(a.model, graphs=False, max_batch_tokens=int(os.environ.get("MAX_BATCH_TOKENS", 65536)))
stamp = lambda: datetime.now(timezone.utc).isoformat()
# an interrupted run can leave a partial last line: cut it off so the file ends at the last complete record (its row
# was skipped as unparsable above, so it is re-run), keeping the file readable by the kit's read_jsonl
if os.path.exists(res_path) and os.path.getsize(res_path):
    with open(res_path, "rb+") as f:
        data = f.read()
        if not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)
out = open(res_path, "a")
t0, n = time.time(), 0


def _wide_many(reqs):
    J = sys.modules["torchcast_decision.jevfmt"]
    plans, first_rows = [], []
    for state, spec in reqs:
        keys = list(spec["criteria"])
        n_groups = -(-len(keys) // m.group)
        size, extra = divmod(len(keys), n_groups)
        groups, start = [], 0
        for g in range(n_groups):
            end = start + size + (1 if g < extra else 0)
            groups.append(keys[start:end]); start = end
        plans.append((state, spec, keys, groups, len(first_rows)))
        first_rows += [J.from_systemone(state, dict(spec, criteria={k: spec["criteria"][k] for k in g})) for g in groups]
    first, _ = m._probs(first_rows)
    fin_rows, fin_meta, out = [], [], [None] * len(reqs)
    for i, (state, spec, keys, groups, off) in enumerate(plans):
        ps = first[off:off + len(groups)]
        first_p = {k: p for g, pg in zip(groups, ps) for k, p in zip(g, pg)}
        finalists = [k for g, pg in zip(groups, ps) for k, _ in sorted(zip(g, pg), key=lambda x: -x[1])[:m.keep]]
        if len(finalists) > J.MAX_OPTIONS:
            out[i] = m._wide(state, spec)[0]
            continue
        fin_meta.append((i, keys, finalists, first_p))
        fin_rows.append(J.from_systemone(state, dict(spec, criteria={k: spec["criteria"][k] for k in finalists})))
    final, _ = m._probs(fin_rows) if fin_rows else ([], 0)
    for (i, keys, finalists, first_p), fp in zip(fin_meta, final):
        final_p = dict(zip(finalists, fp))
        rest = [k for k in keys if k not in set(finalists)]
        mass = sum(first_p[k] for k in rest) or 1.0
        probs = {k: final_p[k] * (1 - m.residual) for k in finalists}
        probs.update({k: m.residual * first_p[k] / mass for k in rest})
        z = sum(probs.values())
        out[i] = {k: v / z for k, v in probs.items()}
    return out


def decide_batch_wide(requests):
    """decide_batch, with every >26-option choice question of the batch decided by _wide_many instead of one _wide call
    per request (the package's own decide_batch runs those sequentially, unbatched)."""
    J = sys.modules["torchcast_decision.jevfmt"]
    wide, slim = [], []
    for ri, (state, questions) in enumerate(requests):
        keep = {}
        for k, q in questions.items():
            crit = q.get("criteria") or {}
            if q.get("type", "choice") == "choice" and isinstance(crit, dict) and len(crit) > J.MAX_OPTIONS:
                wide.append((ri, k, state, q))
            else:
                keep[k] = q
        slim.append((state, keep))
    answers = m.decide_batch([(s, q) for s, q in slim]) if any(q for _, q in slim) else [{} for _ in slim]
    answers = [a if a is not None else {} for a in answers]
    if wide:
        G = m.graphs; m.graphs = {}
        try:
            probs = _wide_many([(state, q) for _, _, state, q in wide])
        finally:
            m.graphs = G
        for (ri, k, _, _), p in zip(wide, probs):
            best = max(p, key=p.get)
            answers[ri][k] = {"type": "choice", "choice": best, "confidence": choice_confidence(list(p.values())),
                              "probabilities": p}
    return answers


def emit(row, status, ms, response=None, err=None):
    e = row["_evaluation"]
    rec = {**e, "started_utc": stamp(), "engine": "torchcast-decision-27b", "status": status, "completed_utc": stamp(),
           "total_wall_ms": ms, "model_request_wall_ms": ms}
    if response is not None:
        rec["response"] = response
    if err:
        rec["error"] = err
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")


for s in range(0, len(rows), a.chunk):
    batch = rows[s:s + a.chunk]
    t = time.perf_counter()
    try:
        answers = decide_batch_wide([(r["state"], r["questions"]) for r in batch])
        ms = (time.perf_counter() - t) * 1000 / len(batch)
        for r, ans in zip(batch, answers):
            resp = {"answers": ans}
            try:
                validate(r["questions"], resp)
                emit(r, "ok", ms, resp)
            except Exception as exc:
                emit(r, "error", ms, err=str(exc))
    except Exception:
        for r in batch:
            t1 = time.perf_counter()
            try:
                ans, _ = m.decide(r["state"], r["questions"])
                resp = {"answers": ans}
                validate(r["questions"], resp)
                emit(r, "ok", (time.perf_counter() - t1) * 1000, resp)
            except Exception as exc:
                emit(r, "error", (time.perf_counter() - t1) * 1000, err=f"{type(exc).__name__}: {exc}")
            import torch; torch.cuda.empty_cache()
    out.flush()
    n += len(batch)
    if (s // a.chunk) % 20 == 0:
        el = time.time() - t0
        print(f"{n}/{len(rows)} {el:.0f}s {n/el:.1f} rows/s eta {(len(rows)-n)/(n/el)/60:.0f} min", flush=True)
out.close()
print("DIBATCH_DONE", a.shard, flush=True)
