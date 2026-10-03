# Notices

Code in this repository: Copyright (c) 2026 Torchcast AI, MIT licence (`LICENSE`).

`eval/di_batch.py` loads the `torchcast_decision` package from the model folder (Apache License 2.0; derived from
StartLux-Decision's inference code, Copyright 2026 StartLux Labs, renamed and modified by Torchcast AI) and imports
the answer validator of the Decision Index kit
([apolinario/decision-index](https://github.com/apolinario/decision-index)); neither is included here, and each stays
under its own licence. The score files in `runs/` were produced by that kit's `score` command.

The model weights are distributed separately at
[torchcast-ai/torchcast-decision-27b](https://huggingface.co/torchcast-ai/torchcast-decision-27b) under CC BY-NC 4.0,
inherited from StartLux-Decision-27B, which is a modified version of a model released by Alibaba Cloud under the Apache
License 2.0. The weights repository carries the full licence texts and StartLux's NOTICE.
