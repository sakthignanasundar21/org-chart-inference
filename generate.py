"""Truth-first generator: simulate a hidden org tree, then run it through a 'mess machine'.
Because the true tree is known, inference can be scored exactly."""
import random, math, csv, os

REGIONS = {"NA": (["New York", "San Francisco", "Austin", "Toronto"], ["NA", "North America", "AMER"]),
           "EMEA": (["London", "Berlin", "Dublin", "Paris"], ["EMEA", "Europe"]),
           "APAC": (["Bengaluru", "Singapore", "Sydney", "Tokyo"], ["APAC", "Asia Pacific"])}

FN = {
 "Engineering": dict(share=.27, scopes=["Platform", "Backend", "Frontend", "Mobile", "Infrastructure", "Payments"],
                     ics=["Software Engineer", "Developer", "Engineer"], dept=["Engineering", "Eng", "R&D"], word=("Engineering", "Eng"), c=("CTO", "Chief Technology Officer")),
 "Data": dict(share=.08, scopes=["Pipelines", "Insights", "Experimentation", "Modeling"],
              ics=["Data Engineer", "Data Scientist", "Data Analyst"], dept=["Data", "Analytics"], word=("Data", "Data"), c=("CDO", "Chief Data Officer")),
 "Security": dict(share=.03, scopes=["AppSec", "GRC", "SecOps"],
                  ics=["Security Engineer", "Security Analyst"], dept=["Security", "InfoSec"], word=("Security", "InfoSec"), c=("CISO", "Chief Information Security Officer")),
 "Product": dict(share=.07, scopes=["Growth", "Core", "Enterprise", "Onboarding"],
                 ics=["Product Manager", "Product Analyst"], dept=["Product", "Prod"], word=("Product", "Prod"), c=("CPO", "Chief Product Officer")),
 "Design": dict(share=.04, scopes=["Brand", "Web", "Research"],
                ics=["UX Designer", "Visual Designer"], dept=["Design", "UX"], word=("Design", "UX"), c=(None, "Chief Design Officer")),
 "Sales": dict(share=.18, scopes=["Enterprise", "SMB", "MidMarket", "Partnerships"],
               ics=["Account Executive", "Account Manager", "SDR"], dept=["Sales", "Revenue"], word=("Sales", "Revenue"), c=("CRO", "Chief Revenue Officer")),
 "Marketing": dict(share=.07, scopes=["Content", "DemandGen", "Brand"],
                   ics=["Marketing Specialist", "Content Marketer"], dept=["Marketing", "Mktg"], word=("Marketing", "Mktg"), c=("CMO", "Chief Marketing Officer")),
 "Finance": dict(share=.04, scopes=["FP&A", "Treasury", "Tax"],
                 ics=["Financial Analyst", "Accountant"], dept=["Finance", "Fin"], word=("Finance", "Fin"), c=("CFO", "Chief Financial Officer")),
 "People": dict(share=.04, scopes=["Compensation", "Benefits", "Learning"],
                ics=["HR Generalist", "People Partner", "Recruiter"], dept=["People", "HR"], word=("People", "HR"), c=("CHRO", "Chief People Officer")),
 "Operations": dict(share=.07, scopes=["Facilities", "Procurement", "BizOps"],
                    ics=["Operations Analyst", "Ops Specialist", "Coordinator"], dept=["Operations", "Ops"], word=("Operations", "Ops"), c=("COO", "Chief Operating Officer")),
 "Legal": dict(share=.02, scopes=["Commercial", "IP", "Privacy"],
               ics=["Counsel", "Paralegal"], dept=["Legal", "Counsel"], word=("Legal", "Legal"), c=("GC", "General Counsel")),
 "Support": dict(share=.08, scopes=["Tier2", "Success", "Escalations"],
                 ics=["Support Specialist", "Customer Success Manager", "Support Engineer"], dept=["Customer Support", "Support", "CS"], word=("Customer Support", "CS"), c=("CCO", "Chief Customer Officer")),
}
# where small functions sometimes report instead of the CEO
ADJ_PARENT = {"Legal": "Finance", "People": "Operations", "Support": "Operations", "Design": "Product",
              "Security": "Engineering", "Data": "Engineering", "Marketing": "Sales"}
ARCH = {  # span of control, manager layers available, number of regions
 "seed": dict(n=(100, 150), span=5, mgr=[3, 4], regions=1),
 "growth": dict(n=(150, 350), span=6, mgr=[2, 3, 4], regions=2),
 "enterprise": dict(n=(350, 700), span=7, mgr=[2, 3, 4], regions=3)}
INDUSTRIES = ["SaaS", "Fintech", "Healthcare", "Retail", "Manufacturing", "Logistics", "Media", "Energy"]
FIRST = "Aarav Maya Liam Zoe Noah Priya Ethan Chen Sofia Omar Hana Lucas Amara Diego Ines Kofi Yuki Ravi Elena Tomas".split()
LAST = "Patel Smith Garcia Kim Nguyen Okafor Rossi Muller Tanaka Singh Cohen Silva Haddad Novak Brown Lopez Ivanov Khan Wong Dubois".split()
IC_ABBR = {"Software Engineer": "SWE", "Senior": "Sr.", "Manager": "Mgr"}
IC_W = [(5, .30), (6, .55), (7, .15)]


def build_company(rng, arch, cid):
    A = ARCH[arch]; N = rng.randint(*A["n"]); regions = rng.sample(list(REGIONS), A["regions"])
    nodes = {}
    def mk(fn, lv, scope, region, parent):
        nid = f"{cid}-{len(nodes):04d}"
        nodes[nid] = dict(id=nid, fn=fn, lv=lv, scope=scope, region=region, parent=parent)
        return nodes[nid]
    root = mk("Exec", 0, None, None, None)
    heads = {}
    fns = [f for f in FN if not (arch == "seed" and f in ("Legal", "Security", "Data") and rng.random() < .4)]
    tot = sum(FN[f]["share"] for f in fns)
    for fn in fns:
        h = max(2, round(N * FN[fn]["share"] / tot)); span = A["span"]
        head_lv = 1 if h >= 60 else 2 if h >= 25 else 3
        M = [l for l in A["mgr"] if l > head_lv]
        k = 0 if h <= span + 1 else (1 if h <= span * span else 2); k = min(k, len(M))
        lays = M[len(M) - k:] if k else []
        below, counts = h * .8, []
        for _ in lays:
            c = max(1, round(below / (span * rng.uniform(.8, 1.3)))); counts.append(c); below = c
        counts.reverse(); n_ic = h - 1 - sum(counts)
        if n_ic < 1: counts = [1] * len(counts); n_ic = max(1, h - 1 - sum(counts))
        reg = None if len(regions) > 1 else regions[0]
        head = mk(fn, head_lv, None, reg, None); heads[fn] = head
        prev = [head]; scopes = FN[fn]["scopes"][:]; rng.shuffle(scopes)
        def attach(specs, prev, first):
            order = list(range(len(specs))); rng.shuffle(order)
            w = [rng.lognormvariate(0, .6) for _ in prev]; out = []
            for j, i in enumerate(order):
                par = prev[j] if j < len(prev) else rng.choices(prev, w)[0]
                if first: sc = scopes[len(out) % len(scopes)] if par is head else par["scope"]
                else: sc = par["scope"] if (par["scope"] and rng.random() < .92) else (rng.choice(scopes) if par is head and rng.random() < .6 else par["scope"])
                rg = par["region"] if (par["region"] and rng.random() < .85) else rng.choice(regions)
                out.append(mk(fn, specs[i], sc, rg, par["id"]))
            return out
        first = True
        for lv, c in zip(lays, counts):
            prev = attach([lv] * c, prev, first); first = False
        ics = [rng.choices([a for a, _ in IC_W], [b for _, b in IC_W])[0] for _ in range(n_ic)]
        attach(ics, prev, first)
    for fn, hd in heads.items():  # function heads: usually CEO, sometimes an adjacent exec
        adj = heads.get(ADJ_PARENT.get(fn))
        hd["parent"] = adj["id"] if (adj and adj["lv"] < hd["lv"] and rng.random() < .35) else root["id"]
    return nodes, regions


def render_title(fn, lv, scope, region, m, rng, multi):
    if fn == "Exec":
        return rng.choice(["CEO", "Chief Executive Officer", "Founder & CEO", "Co-Founder & CEO"])
    F = FN[fn]; full, abbr = F["word"]
    fw = abbr if rng.random() < .15 + .5 * m else full
    if rng.random() < .08 * m and 2 <= lv <= 6:  # stale / shifted seniority
        lv = min(4, max(2, lv + rng.choice([-1, 1]))) if lv <= 4 else min(7, max(5, lv + rng.choice([-1, 1])))
    drop = rng.random() < .02 + .12 * m
    if lv == 1:
        ab, lg = F["c"]; t = rng.choice([lg, f"SVP {fw}", f"SVP, {fw}"] + ([ab] * 2 if ab else []))
    elif lv in (2, 3, 4) and drop:
        t = {2: "VP", 3: "Director", 4: "Manager"}[lv]
    elif lv == 2:
        t = rng.choice([f"VP {fw}", f"VP of {fw}", f"Vice President, {fw}"] + ([f"Head of {fw}"] if rng.random() < .3 else []))
    elif lv == 3:
        t = rng.choice([f"Director of {fw}", f"{fw} Director", f"Sr. Director, {fw}", f"Director, {fw}"] + ([f"Head of {fw}"] if rng.random() < .3 else []))
    elif lv == 4:
        t = rng.choice(["Group Product Manager", f"Manager, {fw}"]) if fn == "Product" else rng.choice([f"{fw} Manager", f"Manager, {fw}", f"Sr. Manager, {fw}", f"{fw} Mgr"])
    else:
        ic = rng.choice(F["ics"])
        if rng.random() < .2 + .4 * m: ic = IC_ABBR.get(ic, ic)
        if drop: ic = rng.choice(["Analyst", "Specialist", "Associate"])
        t = {5: rng.choice(["Senior ", "Sr. ", "Lead ", "Staff ", "Principal "]) + ic, 6: rng.choice([ic, ic, f"{ic} II", f"{ic} I"]),
             7: rng.choice(["Junior ", "Jr. ", "Associate "]) + ic}[lv]
    if lv in (1, 2, 3) and rng.random() < .06 * m:
        t += " & " + FN[rng.choice(list(FN))]["word"][1]
    if rng.random() < .12 * m:
        ws = t.split(); idx = [i for i, w in enumerate(ws) if len(w) >= 6]
        if idx:
            i = rng.choice(idx); j = rng.randrange(1, len(ws[i]) - 1); ws[i] = ws[i][:j] + ws[i][j + 1:]; t = " ".join(ws)
    if scope and rng.random() > .2 + .5 * m:
        t = rng.choice([f"{t}, {scope}", f"{t} - {scope}", f"{t} ({scope})"])
    if multi and region and rng.random() < .3:
        t += rng.choice([" - ", ", "]) + rng.choice(REGIONS[region][1])
    return t


def generate(n_companies=500, seed=7):
    rng = random.Random(seed); companies, rows, truth, hidden = [], {}, {}, {}
    for i in range(n_companies):
        cid = f"c{i:03d}"; arch = rng.choices(list(ARCH), [.4, .4, .2])[0]; m = rng.random()
        nodes, regions = build_company(rng, arch, cid)
        # ghost removal: some leaders simply aren't in the dataset
        elig = [n for n in nodes.values() if 1 <= n["lv"] <= 4 and n["parent"] and any(c["parent"] == n["id"] for c in nodes.values())]
        gone = {n["id"] for n in elig if rng.random() < .015 + .08 * m}
        kids = {}
        for n in nodes.values(): kids.setdefault(n["parent"], []).append(n)
        out = []; used = set()
        for n in nodes.values():
            if n["id"] in gone: continue
            while True:
                nm = f"{rng.choice(FIRST)} {chr(65 + rng.randrange(26))}. {rng.choice(LAST)}"
                if nm not in used: used.add(nm); break
            p = nodes.get(n["parent"]); t = None
            if p is None: t = None
            elif p["id"] in gone: t = ("G", p["fn"], p["lv"])
            else: t = ("M", p["id"])
            truth[n["id"]] = t; hidden[n["id"]] = (n["fn"], n["lv"])
            dept = None if rng.random() < .05 + .3 * m else (rng.choice(FN[n["fn"]]["dept"]) if n["fn"] != "Exec" else "Executive")
            reg = n["region"] or rng.choice(regions)
            loc = None if rng.random() < .05 + .3 * m else rng.choice(REGIONS[reg][0])
            out.append(dict(company_id=cid, person_id=n["id"], name=nm, title=render_title(n["fn"], n["lv"], n["scope"], n["region"], m, rng, len(regions) > 1),
                            department=dept, location=loc))
        rows[cid] = out
        companies.append(dict(company_id=cid, name=f"{rng.choice(['Nova','Apex','Quill','Lumen','Orbit','Vertex','Helix','Mosaic'])}{rng.choice(['ly','ware','ics','Labs','Works','Health'])} {i}",
                              archetype=arch, industry=rng.choice(INDUSTRIES), messiness=round(m, 3), headcount=len(out), ghosts=len(gone)))
    return dict(companies=companies, rows=rows, truth=truth, hidden=hidden)


def write_csv(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/people.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["company_id", "person_id", "name", "title", "department", "location"]); w.writeheader()
        for r in data["rows"].values(): w.writerows(r)
    with open(f"{outdir}/truth.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["person_id", "true_manager", "true_fn", "true_level"])
        for pid, t in data["truth"].items():
            w.writerow([pid, "" if t is None else (t[1] if t[0] == "M" else f"MISSING:{t[1]}:{t[2]}"), *data["hidden"][pid]])
    with open(f"{outdir}/companies.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(data["companies"][0])); w.writeheader(); w.writerows(data["companies"])


if __name__ == "__main__":
    d = generate(); write_csv(d, "out/data")
    print(len(d["companies"]), "companies,", sum(len(r) for r in d["rows"].values()), "people,", sum(c["ghosts"] for c in d["companies"]), "hidden leaders")
