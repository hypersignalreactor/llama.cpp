# Apollos LoopLM fork

An experimental fork of llama.cpp for looped language models (ByteDance Ouro), part of the Apollos project.

Base: llama.cpp PR #29823 ("model : add support for Ouro"), by BrandeisPatrick. On top of it:

- the Ouro early-exit gate is loaded when the GGUF carries it (as `cls.output`; see `add_gate.py`);
- probe mode (`OURO_PROBE=1`): every loop's gate logit and LM-head logits are exposed as graph nodes;
- `llama-ouro-probe`: a teacher-forced tool that records, per token and per loop, what the model would predict;
- `analyze.py`, `learned.py`, `results-phase1.txt`: the first exit-rule trials (Ouro-1.4B Q8_0, 6,634 tokens).

Phase 1 finding: exiting a token's loops once the top prediction's probability reaches 0.9 keeps 99.95% agreement with
full depth at 3.17 of 4 loops on average, ahead of the built-in gate (3.61 loops at 99.4%).

Not yet done: real loop skipping in the engine, KV sharing across loops. Research code; no warranty.
