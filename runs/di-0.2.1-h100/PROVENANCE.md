# Decision Index 0.2.1, full suite on H100

Author-run, not an official leaderboard result.

- **Model:** `torchcast-ai/torchcast-decision-27b@08645834899cb8e00d370b77a3f4e0baa3241dd1`, BF16; the tensors are the ones used for this run (the repository history was squashed to one commit before release).
- **Run:** `eval/di_batch.py --chunk 64` over all 155,390 rows of the 0.2.1 suite, split between two H100 80 GB GPUs, each running a full copy of the model
  (forward and `--reverse`) and combined with `eval/merge_results.py`; every row returned a valid answer.
- **Inference code:** rows were answered with StartLux-Decision's inference package as of 2026-10-02, before it was
  renamed `torchcast_decision`; the runner was identical apart from the import name. The package shipped with the
  model is derived from a later upstream version with the same prompt, readout and temperatures but different forward
  code (shared-prefix reuse, longer context), so a re-run with it can differ in the last digits. The result records of
  this run carry the engine label `startlux-batch`; `eval/di_batch.py` now writes `torchcast-decision-27b`.
- **Scoring:** `python -m decision_index score --engine torchcast-decision-27b`, kit commit
  `87d4650b42b377c0291a89c1f1a879f9b31082bf` (release "Version 0.2.1"). The suite hashes are recorded in `scores.json`.
- **Result:** index 65.00; coverage 1.0 in every area.
- **Files:** `index.json`, `scores.json`, `benchmark-summary.json` as written by the kit. Latency fields are amortised
  batch times, not latency measurements. Per-item answers are not published.

## Weights

SHA-256 of the 12 shards at `0864583`, the tensors used for this run:

```
9300e4ebdd421c1d756f1ac06ecddbcfe5ec35b402541eebe4384099e1894fbb  model-00001-of-00012.safetensors
2ce1278149e531f02dbec7c29f4265c1cd47973876344e60735bdcef2d7d8090  model-00002-of-00012.safetensors
e03471beb16b3a7abf6646d67e83db0850222f2b95a24bc53662ea7127377f56  model-00003-of-00012.safetensors
2102d1dfe4c2a9e7f1972d19e352b3acbc2b3461b351facdc0b734825afbcd38  model-00004-of-00012.safetensors
a3ccc07cff39884329c9132821767498a20a21b0446891529b14cc0cff90bd34  model-00005-of-00012.safetensors
b2270a5f1e8df9fd3653f7218fea845dcca6447411476b6e63491553c39def30  model-00006-of-00012.safetensors
af3f86c9ccc825bf5be9bef6db9ee5f7531223263ad8192a5885499717276439  model-00007-of-00012.safetensors
87e42155477bba9161a66749163029b1235d622eb33603445f50e7e33b25133a  model-00008-of-00012.safetensors
246cd9d58226dd79ecc914857f7422dbf8c07714e7552fce16203ad70047dbf9  model-00009-of-00012.safetensors
859e225dde78fc1508f1074d2fa3f6796d5e013d55288b0bbeb4b25a369a9442  model-00010-of-00012.safetensors
de9651483405269259f1281dd78f563af7e49674fe0dba163ee4841ac438658d  model-00011-of-00012.safetensors
e8be3f2dea13c4f8fa698e085577a4ab96045722652e81822a62b4aceaa73c60  model-00012-of-00012.safetensors
```
