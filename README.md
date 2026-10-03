# Torchcast Decision 27B

Serving and evaluation code for **Torchcast Decision 27B**, a typed-decision model developed by **Torchcast AI**,
fine-tuned from [StartLux-Decision-27B](https://huggingface.co/startlux-models/StartLux-Decision-27B) (itself based on
Qwen3.8-27B). It returns a probability for every option of choice, yes/no (`noul`) and ordinal score questions about a
state, through the TypeSafe-compatible `/v1/systemone` interface.

[Model card and weights](https://huggingface.co/torchcast-ai/torchcast-decision-27b), pinned revision
`08645834899cb8e00d370b77a3f4e0baa3241dd1`.

| Path | Contents |
|:---|:---|
| `eval/di_batch.py` | Decision Index runner: answers suite rows with the model's batched `decide_batch` and writes the kit's `results.jsonl` |
| `eval/merge_results.py` | Merges the results of runs that split the suite between GPUs |
| `runs/di-0.2.1-h100/` | Kit score files of the run reported below |
| `tests/` | Standard-library checks (no GPU) |

## Serve

The weights repository ships the inference package `torchcast_decision/` (Apache-2.0, derived from StartLux-Decision's
inference code; see its `NOTICE`). Linux with an NVIDIA GPU with at least 64 GB of memory (54.7 GB of BF16
weights). Tested with torch 2.11, transformers 5.8.1, flash-linear-attention 0.5.2 and causal-conv1d 1.7.0.

```bash
hf download torchcast-ai/torchcast-decision-27b --revision 08645834899cb8e00d370b77a3f4e0baa3241dd1 \
  --local-dir torchcast-decision-27b
cd torchcast-decision-27b
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                          # causal-conv1d may need --no-build-isolation
python -m torchcast_decision.check .                     # must print "fast kernels: active"
python -m torchcast_decision.server --model . --port 8090
```

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this ticket?",
               "criteria": {"billing": "Payments, refunds and invoices",
                            "shipping": "Delivery and tracking",
                            "technical": "App, login and account problems"}},
    "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"}
  }
}'
```

The server listens on 127.0.0.1 by default and serves one request at a time.

## Evaluate: Decision Index 0.2.1

Use the official kit [apolinario/decision-index](https://github.com/apolinario/decision-index) at release
"Version 0.2.1", commit `87d4650b42b377c0291a89c1f1a879f9b31082bf`. Build the suite as its README describes (it is not
redistributed), answer every row with `eval/di_batch.py`, then score with the kit:

```bash
git clone https://github.com/apolinario/decision-index && cd decision-index
git checkout 87d4650b42b377c0291a89c1f1a879f9b31082bf
pip install -e ".[transformers,rebuild]"                 # in its own environment
python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz
cat suite-0.2/selected-rows.jsonl.gz suite-0.2/added-rows.jsonl.gz > all-rows.jsonl.gz      # 155,390 rows
cd ..

# in the model's environment (see Serve)
DECISION_INDEX_DIR=$PWD/decision-index python eval/di_batch.py torchcast-decision-27b \
    decision-index/all-rows.jsonl.gz runs/full --chunk 64

# in the kit's environment
cd decision-index && python -m decision_index score --results ../runs/full/results.jsonl \
    --engine torchcast-decision-27b --out ../runs/full/score
```

To split the suite between two GPUs, run a second copy with `--reverse` into another directory, stop both when they
meet, and combine them with `python eval/merge_results.py merged.jsonl runs/full/results.jsonl runs/rev/results.jsonl`;
then pass `--results merged.jsonl` to `score` instead of `runs/full/results.jsonl`. The per-request times the runner records are amortised batch times, not latency.

### Result

**Author-run, not an official leaderboard result.**

| | Index | Knowledge | Language | Retrieval | Tools | Arts |
|:---|---:|---:|---:|---:|---:|---:|
| **Torchcast Decision 27B** | **65.00** | **45.58** | 73.81 | **67.54** | **84.58** | **51.51** |
| StartLux-Decision-27B, same pipeline | 63.99 | 44.65 | **74.52** | 66.81 | 82.16 | 47.98 |
| StartLux-Decision-27B, self-reported | 63.88 | | | | | |

Both models went through the pipeline above on all 155,390 rows with coverage 1.0, in BF16 on H100; absolute numbers
may differ slightly from other runners. The scores are single-run point estimates without uncertainty intervals. Score
files: [`runs/di-0.2.1-h100/`](runs/di-0.2.1-h100/).

## Training and benchmark exposure

A LoRA fine-tune of StartLux-Decision-27B, merged into the weights. Training inputs come from train and dev splits of
public datasets and benchmark-format decision data built from them, with the datasets' gold labels as targets. Training inputs came only from train and dev splits of
public datasets; no Decision Index test
rows were used, and all training data was screened against every test row. As expected for these datasets, about 1% of
training rows share a source document or question template with a test item, and one CLINC150 utterance appears in
both splits of that dataset.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Licence

Code in this repository: MIT (`LICENSE`); see `NOTICE.md`. Model weights (distributed separately): **CC BY-NC 4.0**,
inherited from StartLux-Decision-27B, non-commercial use only; commercial use of StartLux-Decision derivatives requires a
separate licence from StartLux Labs. The weights repository carries StartLux's licence and notices, including the
Apache-2.0 notice for the underlying Qwen model.

Credit Torchcast AI for this checkpoint, StartLux Labs for StartLux-Decision, and the Qwen team at Alibaba Cloud for the
underlying model. This project is independent of StartLux Labs, TypeSafe AI and the Decision Index maintainers.
