"""Export held-out (odd) companies from out/preds.pkl into app data + a single-file index.html."""
import pickle, json, random, sys
d = pickle.load(open("out/preds.pkl", "rb")); by, thr, comp, rows = d["by"], d["thr"], d["comp"], d["rows"]
rng = random.Random(3)
odd = [c for c in comp if int(c[1:]) % 2 == 1]
pick = []
for arch, k in (("seed", 12), ("growth", 12), ("enterprise", 6)):
    pool = sorted([c for c in odd if comp[c]["archetype"] == arch], key=lambda c: comp[c]["messiness"])
    pick += [pool[int(i * (len(pool) - 1) / (k - 1))] for i in range(k)]
out = {}
for cid in pick:
    recs = {r["id"]: r for r in by[cid]}
    root = next(r for r in rows[cid] if r["person_id"] not in recs)
    n = [[r["id"], r["name"], r["title"], r["fn"], r["lvl"], r["to"], round(r["p"], 3) if False else round(r["cal"], 3), r.get("runner"), round(r.get("rp", 0), 2), r["truth"], r.get("comps", {}), r.get("rcomps", {})] for r in recs.values()]
    c = comp[cid]
    out[cid] = dict(name=c["name"], arch=c["archetype"], ind=c["industry"], mess=c["messiness"], root=dict(id=root["person_id"], name=root["name"], title=root["title"]), n=n)
data = dict(thr=round(thr, 3), c=out, ev=json.load(open("out/eval.json"))["headline"])
tpl = open("app/template.html").read()
open("app/index.html", "w").write(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
json.dump(data, open("app/data.json", "w"), separators=(",", ":"))
print(len(out), "companies,", sum(len(v["n"]) for v in out.values()), "people")
