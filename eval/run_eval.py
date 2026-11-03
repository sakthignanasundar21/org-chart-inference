"""Eval harness: generate -> infer -> calibrate (even companies) -> abstain threshold -> score (odd companies) -> ablate."""
import sys, os, json, time, math, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.isotonic import IsotonicRegression
from generator.generate import generate, write_csv
from inference.infer import infer_company

LAM = 3.0  # a wrong edge costs 3x what a right edge earns -> emit only if calibrated P(correct) >= 0.75

def score_company(data, cid, cfg):
    recs = []
    for d in infer_company(data["rows"][cid], cfg):
        t = data["truth"][d["id"]]
        if t is None: continue
        to = d["to"]
        ok = (to == t[1]) if (t[0] == "M" and to not in (None, "GHOST")) else (to == "GHOST" and t[0] == "G" and d["fn"] == t[1])
        d.update(cid=cid, ok=bool(ok), truth=("MISSING" if t[0] == "G" else t[1]), tlvl=data["hidden"][d["id"]][1], tfn=data["hidden"][d["id"]][0])
        recs.append(d)
    return recs

def calibrate(train, test):
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit([r["p"] for r in train], [r["ok"] for r in train])
    for r in train + test: r["cal"] = float(iso.predict([r["p"]])[0])
    return iso

def pick_thr(train, lam=LAM):
    best, thr, u = 0, 1.01, 0
    for r in sorted(train, key=lambda r: -r["cal"]):
        u += 1 if r["ok"] else -lam
        if u > best: best, thr = u, r["cal"]
    return thr

def metrics(recs, thr):
    emit = [r for r in recs if r["cal"] >= thr]; n = max(1, len(recs)); tp = sum(r["ok"] for r in emit)
    return dict(n=len(recs), coverage=round(len(emit) / n, 4), precision=round(tp / max(1, len(emit)), 4), recall=round(tp / n, 4),
                wrong_edge_rate=round((len(emit) - tp) / n, 4), raw_accuracy=round(sum(r["ok"] for r in recs) / n, 4))

def ece(recs, bins=10):
    e = 0
    for b in range(bins):
        s = [r for r in recs if b / bins <= r["cal"] < (b + 1) / bins or (b == bins - 1 and r["cal"] == 1)]
        if s: e += len(s) / len(recs) * abs(np.mean([r["cal"] for r in s]) - np.mean([r["ok"] for r in s]))
    return round(float(e), 4)

def run(data, cids, cfg):
    tr = [r for c in cids if int(c[1:]) % 2 == 0 for r in score_company(data, c, cfg)]
    te = [r for c in cids if int(c[1:]) % 2 == 1 for r in score_company(data, c, cfg)]
    calibrate(tr, te); thr = pick_thr(tr)
    return tr, te, thr

if __name__ == "__main__":
    t0 = time.time(); data = generate(); write_csv(data, "out/data"); comp = {c["company_id"]: c for c in data["companies"]}
    print("people:", sum(len(r) for r in data["rows"].values()))
    cids = list(comp); tr, te, thr = run(data, cids, {})
    print(f"full run {time.time()-t0:.0f}s  threshold={thr:.3f}  train={len(tr)} test={len(te)}")
    M = metrics(te, thr); print("TEST", M, "ECE", ece(te))
    sl = {"archetype": {a: metrics([r for r in te if comp[r["cid"]]["archetype"] == a], thr) for a in ("seed", "growth", "enterprise")},
          "messiness": {f"{lo:.2f}-{hi:.2f}": metrics([r for r in te if lo <= comp[r["cid"]]["messiness"] < hi], thr) for lo, hi in [(0, .33), (.33, .66), (.66, 1.01)]},
          "level": {"leaders(1-4)": metrics([r for r in te if 1 <= r["tlvl"] <= 4], thr), "ICs(5-7)": metrics([r for r in te if r["tlvl"] >= 5], thr)}}
    for k, v in sl.items():
        for kk, m in v.items(): print(f"  {k:10s} {kk:14s} cov={m['coverage']:.2f} prec={m['precision']:.3f} raw={m['raw_accuracy']:.3f} wrong={m['wrong_edge_rate']:.3f}")
    s = sorted(te, key=lambda r: -r["cal"]); rc = []
    for q in range(5, 101, 5):
        k = int(len(s) * q / 100); rc.append(dict(coverage=q / 100, precision=round(float(np.mean([r["ok"] for r in s[:k]])), 4)))
    cal = []
    for b in range(10):
        x = [r for r in te if b / 10 <= r["cal"] < (b + 1) / 10 + (1e-9 if b == 9 else 0)]
        if x: cal.append(dict(bin=b / 10, n=len(x), mean_pred=round(float(np.mean([r["cal"] for r in x])), 3), mean_true=round(float(np.mean([r["ok"] for r in x])), 3)))
    sub = cids[:160]; ab = {}
    for name, cfg in {"full": {}, "no_global_solver": dict(glob=False), "no_scope": dict(scope=False, orphan=False), "no_region": dict(region=False),
                      "no_ghost_nodes": dict(ghost=False), "no_layer_logic": dict(layers=False),
                      "naive_baseline": dict(scope=False, region=False, ghost=False, glob=False, orphan=False, layers=False)}.items():
        a, b, th = run(data, sub, cfg); m = metrics(b, th); ab[name] = dict(**m, thr=round(th, 3))
        print(f"  ablation {name:18s} raw_acc={m['raw_accuracy']:.3f} cov={m['coverage']:.2f} prec={m['precision']:.3f} wrong={m['wrong_edge_rate']:.3f}")
    lam = {str(l): metrics(te, pick_thr(tr, l)) for l in (1, 3, 9)}
    json.dump(dict(headline=M, ece=ece(te), threshold=thr, lam=LAM, slices=sl, risk_coverage=rc, calibration=cal, ablations=ab, lambda_sweep=lam),
              open("out/eval.json", "w"), indent=1)
    gh = [r for r in te if r["truth"] == "MISSING"]; print("ghost truth n=", len(gh), "ghost found:", sum(r["to"] == "GHOST" for r in gh), "ghost emitted-correct:", sum(r["ok"] and r["cal"] >= thr for r in gh))
    for r in sorted([r for r in te if r["cal"] >= thr and not r["ok"]], key=lambda r: -r["cal"])[:6]:
        print("FAIL", r["title"], "| pred", r["to"], "truth", r["truth"], "| cal %.2f" % r["cal"], r["comps"])
    by = {}
    for r in tr + te:
        r["emit"] = r["cal"] >= thr; by.setdefault(r["cid"], []).append(r)
    pickle.dump(dict(by=by, thr=thr, comp=comp, rows=data["rows"]), open("out/preds.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
