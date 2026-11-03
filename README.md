# Org chart inference
```
generator/generate.py   truth-first dataset generator + mess machine -> out/data/*.csv
inference/infer.py      title parser, candidate scoring, global assignment, ghost nodes
inference/llm_parse.py  optional cached LLM tie-breaker (off by default)
eval/run_eval.py        calibration, abstention threshold, slices, ablations -> out/eval.json (a copy of the final run is in results/)
app/                    export.py builds a single-file static app (index.html) from held-out predictions
decisions/LOG.md        AI-assisted development log: what was tried, what failed
WRITEUP.md              approach, results, failures
```
Run: `pip install -r requirements.txt` then `python eval/run_eval.py && python app/export.py` (about 2 minutes). Host `app/index.html` on any static host (e.g. Netlify drop).
