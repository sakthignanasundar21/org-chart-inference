"""Org-chart inference: parse titles -> candidate managers -> weak-signal scores -> global solve -> ghost/abstain."""
import re, math, random, difflib
from collections import defaultdict, Counter

FN_RULES = [("Security", r"secur|infosec|\bciso\b"), ("Data", r"\bdata\b|analytic|\bml\b|\bbi\b|\bcdo\b|scientist"),
 ("Sales", r"sales|revenue|\bcro\b|\bsdr\b|\bbdr\b|account exec|account manager"), ("Design", r"design|\bux\b"),
 ("Product", r"product|\bprod\b|\bcpo\b"), ("Marketing", r"market|mktg|\bcmo\b"),
 ("Finance", r"financ|\bfin\b|\bcfo\b|accountant|accounting|controller"), ("People", r"people|\bhr\b|\bchro\b|recruiter"),
 ("Legal", r"legal|counsel|paralegal|\bgc\b"), ("Support", r"support|customer|\bcs\b|\bcco\b"),
 ("Operations", r"operat|\bops\b|\bcoo\b|coordinator"), ("Engineering", r"engineer|\beng\b|\bswe\b|develop|software|\bcto\b|r&d|technolog")]
FN_RE = [(f, re.compile(r)) for f, r in FN_RULES]
VOCAB = ["engineer", "engineering", "director", "manager", "security", "marketing", "operations", "product", "design", "support", "finance",
         "legal", "senior", "principal", "analyst", "developer", "scientist", "executive", "president", "officer", "customer", "technology",
         "account", "specialist", "associate", "junior", "generalist", "partner", "recruiter", "paralegal", "accountant", "coordinator", "designer"]
IC_MGR = re.compile(r"(?<!group )\b(?:product|program|project|account|customer success) manager")
REGION = {"na": "NA", "north america": "NA", "amer": "NA", "emea": "EMEA", "europe": "EMEA", "apac": "APAC", "asia pacific": "APAC"}
CITY = {c.lower(): r for r, cs in {"NA": ["New York", "San Francisco", "Austin", "Toronto"], "EMEA": ["London", "Berlin", "Dublin", "Paris"],
        "APAC": ["Bengaluru", "Singapore", "Sydney", "Tokyo"]}.items() for c in cs}
ADJ = defaultdict(set)
for a, b in [("Data", "Engineering"), ("Security", "Engineering"), ("Design", "Product"), ("Product", "Engineering"), ("Legal", "Finance"),
             ("People", "Operations"), ("People", "Finance"), ("Support", "Operations"), ("Support", "Sales"), ("Marketing", "Sales")]:
    ADJ[a].add(b); ADJ[b].add(a)
LAYER = [0, -1.3, -2.6, -4.0]
CAP = {0: 12, 1: 10, 2: 9, 3: 9, 4: 10}
DEFAULT = dict(layers=True, scope=True, region=True, orphan=True, ghost=True, glob=True, fn=True)


def fix_typos(t):
    out = []
    for w in re.findall(r"[a-z&']+|[^a-z&']+", t):
        if w.isalpha() and len(w) >= 5 and w not in VOCAB:
            m = difflib.get_close_matches(w, VOCAB, 1, .86); w = m[0] if m else w
        out.append(w)
    return "".join(out)


def level_of(t):
    if re.search(r"\bceo\b|chief executive|founder", t): return 0
    if re.search(r"\bc[a-z]{1,3}o\b|chief|general counsel|\bgc\b|\bsvp\b|\bevp\b|senior vice", t): return 1
    if re.search(r"\bvp\b|vice president", t): return 2
    if "head of" in t: return 2.5
    if re.search(r"director|\bdir\b", t): return 3
    t2 = IC_MGR.sub("", t)
    if re.search(r"manager|\bmgr\b", t2): return 4
    if re.search(r"\blead\b|staff|principal|senior|\bsr\b|architect", t2): return 5
    if re.search(r"junior|\bjr\b|associate|\bintern\b|trainee", t2): return 7
    return 6


def funcs_of(t):
    hits = sorted((m.start(), f) for f, r in FN_RE if (m := r.search(t)))
    fns = [f for _, f in hits]
    if "Engineering" in fns and len(fns) > 1 and not re.search(r"engineering|\beng\b|\bswe\b|\bcto\b|r&d|technolog", t):
        fns.remove("Engineering")  # 'Data Engineer', 'Security Engineer' are compound nouns, not dual-hats
    return fns


def parse_row(r):
    raw = r["title"] or ""; t = fix_typos(raw.lower())
    segs = [s.strip() for s in re.split(r"\s[-–—]\s|\s*,\s*|\(|\)", t) if s.strip()]
    base, rest = (segs[0] if segs else t), segs[1:]
    reg, sc = None, []
    for s in rest:
        if s in REGION: reg = REGION[s]
        elif not funcs_of(s) and not re.search(r"director|manager|\bvp\b", s): sc.append(s)
    lvl = level_of(t)
    fns = funcs_of(base) or [f for sg in rest for f in funcs_of(sg)][:1] or funcs_of((r["department"] or "").lower())
    loc = (r["location"] or "").lower()
    return dict(id=r["person_id"], name=r["name"], title=raw, dept=r["department"], loc=r["location"], lvl=lvl, key=math.ceil(lvl),
                fns=fns, scope_c=sc, region_t=reg, region_l=CITY.get(loc))


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; z = sum(e); return [v / z for v in e]


def infer_company(rows, cfg=None):
    cfg = {**DEFAULT, **(cfg or {})}; rng = random.Random(1)
    P = [parse_row(r) for r in rows]
    cnt = Counter(s for p in P for s in p["scope_c"]); valid = {s for s, c in cnt.items() if c >= 2}
    for p in P:  # company-specific scope vocabulary learned from the data itself
        sc = None
        for s in p["scope_c"]:
            if s in valid: sc = s; break
            m = difflib.get_close_matches(s, list(valid), 1, .8)
            if m: sc = m[0]; break
        p["scope"] = sc if cfg["scope"] else None
        p["region"] = (p["region_t"] or p["region_l"]) if cfg["region"] else None
        p["fn"] = p["fns"][0] if p["fns"] else None
    # learn team -> function from this company's own fully-parsed titles, then fill people whose function is missing
    tf = defaultdict(Counter)
    for p in P:
        if p["fn"] and p["scope"]: tf[p["scope"]][p["fn"]] += 1
    for p in P:
        if not p["fn"] and p["scope"] and cfg.get("team_fn", True) and tf[p["scope"]]:
            f, c = tf[p["scope"]].most_common(1)[0]
            if c >= 3 and c / sum(tf[p["scope"]].values()) >= .8: p["fn"] = f; p["fns"] = [f]; p["fn_inferred"] = True
    if cfg.get("llm"):  # optional tie-breaker for titles that are still unresolved after rules + team vocabulary
        un = sorted({p["title"] for p in P if not p["fn"] and p["lvl"] > 0 and re.search(r"[a-z]{4,}", re.sub(r"\b(senior|junior|manager|director|analyst|specialist|associate|lead|staff|principal)\b", "", p["title"].lower()))})
        if un:
            got = cfg["llm"](un, sorted(valid))
            for p in P:
                if not p["fn"] and got.get(p["title"]): p["fn"] = got[p["title"]]; p["fns"] = [p["fn"]]; p["fn_inferred"] = True
    ceos = [p for p in P if p["lvl"] == 0]
    root = min(ceos, key=lambda p: p["id"]) if ceos else None
    mgrs = [p for p in P if p["lvl"] <= 4]
    byfn = defaultdict(list)
    for m in mgrs:
        for f in m["fns"]: byfn[f].append(m)
    unk = [m for m in mgrs if not m["fns"] and m is not root]  # leaders whose function we couldn't parse: wildcard candidates
    size = Counter(p["fn"] for p in P if p is not root and p["fn"])
    top = {}
    for p in P:
        if p is not root and p["fn"]: top[p["fn"]] = min(top.get(p["fn"], 99), p["key"])
    S = {f: sorted({m["key"] for m in ms if m["key"] > 0} | {0}) for f, ms in byfn.items()}
    S[None] = sorted({m["key"] for m in mgrs})
    pres = Counter((f, m["scope"]) for m in mgrs for f in m["fns"] if m["scope"])
    exp_key = lambda f: 1 if size[f] >= 60 else 2 if size[f] >= 25 else 3

    def comps(p, m):
        c = {}; kp, km = p["key"], m["key"]; pf, mf = p["fns"], m["fns"]
        istop = bool(p["fn"]) and kp <= 4 and kp == top.get(p["fn"], 99)
        ex = max(0, kp - exp_key(p["fn"])) if istop else 0
        Sp = S.get(p["fn"], S[None])
        if m is root:
            c["function"] = .6 if kp <= 2 else -.3
            if istop: c["head_gap"] = -.5 * min(ex, 2)
            d = 0 if istop else sum(1 for k in Sp if 0 < k < kp)
        else:
            if not cfg["fn"]: c["function"] = 0.0
            elif pf and mf and set(pf) & set(mf): c["function"] = 2.0
            elif not pf or not mf: c["function"] = 0.0
            elif any(a in ADJ[f] for f in pf for a in mf): c["function"] = .2
            else: c["function"] = -3.0
            d = sum(1 for k in Sp if km < k < kp)
        lay = LAYER[min(d, 3)] if cfg["layers"] else -1.3 * abs(round(kp - km) - 1)
        c["layer"] = lay - (.8 if km == kp else 0)
        if cfg["scope"]:
            if p["scope"]:
                n_pres = pres[(p["fn"], p["scope"])]
                if m["scope"]: c["scope"] = 1.4 if m["scope"] == p["scope"] else (-2.4 if n_pres else -.4)
                elif m is not root and n_pres: c["scope"] = -.8
            elif m["scope"]: c["scope"] = -.2
        if cfg["region"] and p["region"] and m["region"]: c["region"] = .5 if p["region"] == m["region"] else -.7
        return c

    def ghost(p):
        g = -.9; c = {"prior": -.9}
        istop = bool(p["fn"]) and p["key"] <= 4 and p["key"] == top.get(p["fn"], 99)
        if cfg["orphan"] and p["scope"] and pres[(p["fn"], p["scope"])] == 0 and p["key"] > 2:
            g += 1.2; c["orphan_scope"] = 1.2
        if cfg["orphan"] and istop and p["key"] > exp_key(p["fn"]):
            b = 1.4 * min(p["key"] - exp_key(p["fn"]), 2); g += b; c["missing_head"] = b
        return g, c

    C = {}
    for p in P:
        if p is root: continue
        pool = {}
        for f in p["fns"]:
            for m in byfn[f]: pool[m["id"]] = m
            for a in ADJ[f]:
                for m in byfn[a]:
                    if m["lvl"] <= 3: pool[m["id"]] = m
        if not p["fns"]:
            for m in mgrs:
                if m["lvl"] <= 3: pool[m["id"]] = m
        for m in unk: pool[m["id"]] = m
        if root: pool[root["id"]] = root
        cl = []
        for m in pool.values():
            if m is p or not m["lvl"] < p["lvl"]: continue
            c = comps(p, m); cl.append((m["id"], sum(c.values()), c))
        if cfg["ghost"]:
            g, gc = ghost(p); cl.append(("GHOST", g, gc))
        C[p["id"]] = cl

    order = [p["id"] for p in P if p["id"] in C]; rng.shuffle(order)
    lvl_of = {p["id"]: p["key"] for p in P}; assign = {}; res = {}; load = Counter()
    for pas in range(2 if cfg["glob"] else 1):
        new = Counter()
        for pid in order:
            cl = C[pid]
            if not cl: continue
            adj = []
            for k, s, _ in cl:
                if k != "GHOST" and cfg["glob"]:
                    ld = new[k] if pas == 0 else load[k] - (1 if assign.get(pid) == k else 0)
                    s -= .5 * max(0, ld - CAP.get(lvl_of[k], 10))
                adj.append(s)
            pr = softmax(adj); j = max(range(len(pr)), key=pr.__getitem__)
            assign[pid] = cl[j][0]; res[pid] = (pr, j); new[cl[j][0]] += 1
        load = new
    pm = {p["id"]: p for p in P}; out = []
    for p in P:
        d = dict(id=p["id"], name=p["name"], title=p["title"], dept=p["dept"], loc=p["loc"], fn=p["fn"], lvl=p["lvl"], scope=p["scope"], region=p["region"])
        if p["id"] in res:
            pr, j = res[p["id"]]; cl = C[p["id"]]
            alt = sorted(range(len(pr)), key=lambda i: -pr[i])
            d.update(to=cl[j][0], p=pr[j], comps={k: round(v, 2) for k, v in cl[j][2].items()},
                     runner=cl[alt[1]][0] if len(alt) > 1 else None, rp=pr[alt[1]] if len(alt) > 1 else 0,
                     rcomps={k: round(v, 2) for k, v in cl[alt[1]][2].items()} if len(alt) > 1 else {}, ncand=len(cl))
        elif p is root: d.update(to="ROOT", p=1.0)
        else: d.update(to=None, p=0.0)
        out.append(d)
    return out
