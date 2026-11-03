# Org Chart Inference: write-up

**Live app:** (paste your hosted link here) · **Run it:** `python eval/run_eval.py` (about 90s) then `python app/export.py`

## Approach in one paragraph
I built the answer key before the data. A generator simulates 500 companies (100 to 700 people; startup, growth and enterprise shapes) with a hidden true org tree, then a "mess machine" turns it into the flat list: abbreviations, typos, missing function/team/location, stale seniority, dual-hat titles, and leaders deleted from the dataset entirely. Because the truth is known, every claim below is measured, not eyeballed. Inference then parses titles, scores candidate managers from weak signals, assigns globally, and abstains or attaches to a "ghost" leader when it can't resolve an edge.

## Key decisions
1. **Truth first, messiness as a dial.** Each company has a 0 to 1 messiness level, so accuracy can be sliced by it.
2. **"A wrong edge is worse than no edge" is a number, not a slogan.** Confidence is calibrated with isotonic regression on even-numbered companies; the emit threshold maximises (correct − 3 × wrong) on those same companies. Everything is reported on odd-numbered companies the model never saw.
3. **Layers are relative.** "Nearest layer above" is counted over layers that exist in that function, so flat companies that skip levels aren't penalised.
4. **Ghost nodes.** Someone whose team has no visible manager, or whose function's most senior person is lower than a function that size would normally have, attaches to "X leadership (not in dataset)". For sales this is a feature: a probable hidden buyer.
5. **Team vocabulary is learned per company** from repeated title suffixes, then typo-matched. No hand-written team list.
6. **Rules first, LLM as an optional hook.** Rules, fuzzy matching and per-company team vocabulary go a long way and are free and reproducible; an LLM tie-breaker exists but is off by default (see The LLM question).

## Results (held-out: 250 companies, 61,981 people)
| Measure | Result |
|---|---|
| Edges emitted (coverage) | 35.3% |
| Precision of emitted edges | 91.8% |
| Wrong edges, as share of all people | 2.9% |
| Calibration error (ECE) | 0.0065 |
| Leaders (levels 1 to 4): coverage / precision | 68% / 92% |
| ICs (levels 5 to 7): coverage / precision | 29% / 92% |
| Precision at low / mid / high messiness | 95.3% / 92.3% / 84.9% |
| Precision seed / growth / enterprise | 93.4% / 92.5% / 89.6% |

Raw top-1 accuracy is only 52%. That is expected: in most teams several managers have identical title, team and region, so the right manager is unknowable from this data. The system declines to guess those, which is why coverage is modest and precision is high. `out/eval.json` (copy in `results/`) has the risk-coverage and calibration curves.

### Ablations (160-company subset, re-calibrated each time)
| Variant | Coverage | Precision |
|---|---|---|
| Full | 34% | 92.7% |
| Naive baseline (function + level only) | 24% | 85.2% |
| No relative-layer logic | 27% | 83.1% |
| No team signal | 25% | 91.9% |
| No global solver | 37% | 90.5% |
| No ghost nodes | 39% | 89.0% |
| No region signal | 31% | 94.0% |

Honest reading: relative layers, the global solver and ghost nodes each buy 2 to 4 points of precision at the cost of some coverage (the full system abstains more where it is unsure). The team signal mainly buys coverage. **Region does not help** (precision is higher without it). Gaps under about 1 point are noise on a 160-company subset.

## Failure gallery
1. **Lookalike victims of a deleted leader.** "Sales Manager (SMB)" reports to a manager who isn't in the data, but a visible lookalike exists, so the model attaches to it at 100% confidence. Nothing in the data shows the missing node; only a span-of-control prior could catch it.
2. **Same-title sibling swaps.** "Sales Director - APAC" with two near-identical directors: scores tie and calibration can't know.
3. **Ghost detection is weak:** of 3,019 people whose true manager was deleted, 323 were flagged and 133 were emitted correctly.
4. **Bugs the gallery and error analysis caught and I fixed** (see `decisions/LOG.md`): ICs mistaken for function heads; managers with unknown function never offered as candidates; function words in the second half of a title ("Manager, Eng") ignored, which alone lifted coverage from 28% to 37%; titles with a team but no function now inherit the function from that team elsewhere in the same company.

## The LLM question
`inference/llm_parse.py` is a cached, batched, temperature-0 tie-breaker for titles still unresolved after rules and the company's team vocabulary (`cfg={"llm": make_llm_parser()}`). **It is not in the numbers above:** there was no API key in my build environment, so I verified only the plumbing with a stub. I also measured the headroom: after the rules fixes only about 3.5 unique unresolved-but-informative titles remain per company, so on this synthetic data an LLM can add little. It should matter more on real data, where titles are stranger than my generator's. Run it with the same harness before claiming a gain.

## Limitations
- **The data is synthetic and I wrote both sides.** Parts of the inference prior (e.g. function size implies head seniority) mirror how the generator works. Performance on real data will be lower; the harness is what transfers.
- Calibration is only as good as how realistic the mess is.
- No dotted-line or matrix reporting; ICs are assumed never to manage.

## What I'd do next
1. Run the LLM tie-breaker on real titles and measure it with this harness.
2. Add real-world signals the generator lacks (tenure, hiring dates, email threads) and a span-of-control prior to catch deleted leaders.
3. Emit "one of these 3 managers" sets instead of abstaining outright, since the team is often known when the manager isn't.
4. Run the eval on every change in CI and track the score over time.
