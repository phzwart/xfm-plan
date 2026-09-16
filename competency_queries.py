"""Competency queries over xfm-gold.plan.yaml (networkx; Cypher/SPARQL equivalents are one-to-one)."""
import sys, yaml, networkx as nx
from collections import defaultdict
g = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "out/xfm-gold.plan.yaml"))
idx = {n["id"]: n for k, v in g.items() if isinstance(v, list) for n in v}
H = nx.DiGraph()
for t in g["transformations"]:
    for x in t.get("consumes", []): H.add_edge(x, t["id"])
    for x in t.get("produces", []): H.add_edge(t["id"], x)
def short(x): return x.split("/")[-1]

print("Q1 paths raw spectra -> co-localization claim (NB: consumes is AND; simple paths treat it as OR — see README)")
for p in nx.all_simple_paths(H, "xfm:ds/mca_spectrum", "xfm:ds/claim_colocalization"):
    ts = [x for x in p if x.startswith("xfm:t/")]
    print("  ", [short(t) for t in ts], "| assumes:", sorted({short(a) for t in ts for a in idx[t].get("assumes", [])}))

print("\nQ2 transformations introducing an external dependence")
for t in g["transformations"]:
    ext = [i for i in t.get("introduces", []) if any(w in i.lower() for w in ("standard", "matrix", "model", "table", "assignment"))]
    if ext: print("  ", short(t["id"]), "->", ext)

print("\nQ3 which assumption, if violated, invalidates the most goals")
wf_t = {w["id"]: {s["transformation"] for s in w["steps"]} for w in g["workflows"]}
wf_g = {w["id"]: set(w["serves"]) for w in g["workflows"]}
score = defaultdict(set)
for t in g["transformations"]:
    for a in t.get("assumes", []):
        for w, ts in wf_t.items():
            if t["id"] in ts: score[a] |= wf_g[w]
for a, gs in sorted(score.items(), key=lambda kv: -len(kv[1])): print("  ", short(a), len(gs), sorted(map(short, gs)))

print("\nQ4 which steps carry uncertainty forward and which discard it")
for t in g["transformations"]:
    ins = {idx[x].get("uncertainty", "none") for x in t["consumes"]}; outs = {idx[x].get("uncertainty", "none") for x in t["produces"]}
    if ins != outs: print("  ", short(t["id"]), sorted(ins), "->", sorted(outs))

print("\nQ5 decisions: human-required and rule status")
for d in g["decisions"]: print("  ", short(d["id"]), "| human:", d.get("requires_human"), "| rule:", d.get("has_rule"), "|", d.get("resolved_by", ""))

print("\nQ6 which claims are bounded by a validation outcome, and by what")
for v in g["validation_criteria"]:
    if v.get("derived_bound"): print("  ", short(v["id"]), "->", v["derived_bound"])

print("\nQ7 goals reachable only through a step with an inferred/unknown/absent support or an open gap")
for w in g["workflows"]:
    flagged = [short(s["transformation"]) for s in w["steps"] if idx[s["transformation"]].get("gaps")]
    print("  ", short(w["id"]), "serves", sorted(map(short, w["serves"])), "| steps with gaps:", flagged)

print("\nGap census")
kinds = defaultdict(list)
for n in idx.values():
    for x in n.get("gaps", []): kinds[x["kind"]].append((short(n["id"]), x["field"]))
for k, v in kinds.items(): print("  ", k, len(v), v)
