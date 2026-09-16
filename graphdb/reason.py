"""Goal-directed reasoning over a sciplan plan graph.

plan(goal, have, facts) answers: given what I have (DataStates) and what I know about the world
(assumptions held true/false), which routes reach the goal, which are blocked and by what, which
decisions I must make, and what single piece of information would unlock a blocked route.

Purely structural: it uses consumes/produces (as conjunctions), assumes, requires, alternative_to,
decisions/options, and validated_by. It does not evaluate governing relations or numbers.
"""
import sys, yaml
from collections import defaultdict

class Plan:
    def __init__(self, path):
        g = yaml.safe_load(open(path))
        self.g = g
        self.idx = {n["id"]: n for k, v in g.items() if isinstance(v, list) for n in v}
        self.T = {t["id"]: t for t in g["transformations"]}
        self.producers = defaultdict(list)
        for t in self.T.values():
            for o in t["produces"]: self.producers[o].append(t["id"])
        self.states_of_goal = {gl["id"]: gl.get("satisfied_by") or [gl.get("claim_type")] for gl in g["goals"]}
        self.decision_for = {}
        for d in g["decisions"]:
            for o in d["options"]:
                if o.get("selects_transformation"): self.decision_for[o["selects_transformation"]] = (d["id"], o["label"])

    def s(self, x): return x.split("/")[-1]

    def routes(self, target, have, facts, depth=0, seen=None):
        """Enumerate ways to obtain DataState `target`. Returns list of route dicts."""
        seen = seen or set()
        if target in have: return [dict(steps=[], blocked=[], decisions=[], unlock=[])]
        if target in seen or depth > 25: return []
        out = []
        for tid in self.producers.get(target, []):
            t = self.T[tid]
            # each input must be obtainable (conjunction) — take first feasible route per input, but record all blockers
            sub_steps, sub_blocked, sub_dec, sub_unlock, feasible = [], [], [], [], True
            for inp in t["consumes"]:
                rs = self.routes(inp, have, facts, depth + 1, seen | {target})
                if not rs: feasible = False; sub_blocked.append(f"no way to obtain {self.s(inp)}"); continue
                best = min(rs, key=lambda r: (len(r["blocked"]), len(r["steps"])))
                sub_steps += best["steps"]; sub_blocked += best["blocked"]; sub_dec += best["decisions"]; sub_unlock += best["unlock"]
            # this transformation's own assumptions against the facts
            for a in t.get("assumes", []):
                if facts.get(a) is False:
                    sub_blocked.append(f"{self.s(tid)} assumes {self.s(a)}, which is false here")
                    fm = self.idx[a].get("if_violated", [])
                    if fm: sub_blocked[-1] += f" -> failure mode {', '.join(self.s(f) for f in fm)}"
                    for alt in t.get("alternative_to", []):
                        at = self.T[alt]
                        need = [self.s(p) for p in at.get("parameters", [])]; asm = [self.s(x) for x in at.get("assumes", [])]
                        sub_unlock.insert(0, f"ALTERNATIVE {self.s(alt)}: needs parameters {need}; assumes {asm}; produces {[self.s(o) for o in at['produces']]}")
                elif a not in facts:
                    sub_unlock.append(f"confirm {self.s(a)} ({self.idx[a]['about']}; {self.idx[a].get('confirmable_by') or 'no confirmation route stated'}) for {self.s(tid)}")
            for pid in t.get("parameters", []):
                pd = self.idx[pid]
                if pd.get("kind") == "physical" and pd.get("default_value") is None and facts.get(pid) is not True:
                    sub_blocked.append(f"{self.s(tid)} needs physical parameter {self.s(pid)} ({pd.get('physical_meaning','')}), not available")
            for pre in t.get("requires", []):
                p = self.idx[pre]
                if facts.get(pre) is False: sub_blocked.append(f"{self.s(tid)} requires {self.s(pre)} ({p['checkable_by']}), which is false here")
                elif pre not in facts and not p.get("established_by"): sub_unlock.append(f"check {self.s(pre)} from {p['checkable_by']} for {self.s(tid)}")
            if tid in self.decision_for: sub_dec.append((self.s(self.decision_for[tid][0]), self.decision_for[tid][1]))
            out.append(dict(steps=sub_steps + [tid], blocked=sub_blocked, decisions=sorted(set(sub_dec)), unlock=sorted(set(sub_unlock))))
        return out

    def plan(self, goal, have, facts):
        rs = []
        for target in self.states_of_goal[goal]:
            for r in self.routes(target, set(have), facts):
                r["satisfies"] = target; rs.append(r)
        feasible = [r for r in rs if not r["blocked"]]; blocked = [r for r in rs if r["blocked"]]
        return feasible, blocked

def show(P, title, goal, have, facts):
    print("=" * 78); print(title); print("  goal:", P.s(goal), "| have:", [P.s(h) for h in have], "| facts:", {P.s(k): v for k, v in facts.items()})
    feasible, blocked = P.plan(goal, have, facts)
    if not feasible: print("  NO feasible route.")
    for i, r in enumerate(feasible, 1):
        print(f"  route {i} (yields {P.s(r['satisfies'])}): " + " -> ".join(P.s(x) for x in dict.fromkeys(r["steps"])))
        if r["decisions"]: print("     decisions to take:", r["decisions"])
        if r["unlock"]:
            print("     still unverified (route valid only if these hold):"); [print("       -", u) for u in r["unlock"]]
        t_last = P.T[r["steps"][-1]]
        v = [P.s(x) for x in t_last.get("validated_by", [])] + [P.s(x) for x in P.idx[goal].get("validated_by", [])]
        if v: print("     validate by:", sorted(set(v)))
    for r in blocked:
        print(f"  blocked (toward {P.s(r['satisfies'])}): " + " -> ".join(P.s(x) for x in dict.fromkeys(r["steps"])))
        for b in r["blocked"]: print("     x", b)
        for u in [u for u in r["unlock"] if u.startswith("ALTERNATIVE")]: print("     ->", u)

if __name__ == "__main__":
    P = Plan(sys.argv[1] if len(sys.argv) > 1 else "out/xfm-gold.plan.yaml")
    show(P, "S1  Absolute concentration from a thick tissue section, matrix unknown, thin-film standard measured at the same energy/distance/gain",
         "xfm:g/absolute_concentration", ["xfm:ds/mca_spectrum"],
         {"xfm:a/infinitely_thin": False, "xfm:a/same_energy": True, "xfm:a/same_distance": True, "xfm:a/same_gain": True,
          "xfm:p/std_conc": True, "xfm:p/i0_gain": True, "xfm:pre/standard_measured": True, "xfm:pre/i0_recorded": True})
    show(P, "S2  Same sample, but composition, density and thickness are now known",
         "xfm:g/absolute_concentration", ["xfm:ds/mca_spectrum"],
         {"xfm:a/infinitely_thin": False, "xfm:a/secondary_negligible": True, "xfm:p/sample_composition": True, "xfm:pre/standard_measured": True, "xfm:pre/i0_recorded": True})
    show(P, "S3  Co-localization claim, but only XRF spectra in hand (no FTIR, no optical)",
         "xfm:g/colocalization", ["xfm:ds/mca_spectrum"], {})
    show(P, "S4  Co-localization with all three inputs; detector known linear; sample rotated between modalities",
         "xfm:g/colocalization", ["xfm:ds/mca_spectrum", "xfm:ds/ftir_spectrum", "xfm:ds/optical_image"],
         {"xfm:a/detector_linear": True, "xfm:a/translation_only": False, "xfm:a/gaussian_response": True, "xfm:a/escape_geometry": True,
          "xfm:a/auger_neglected": True, "xfm:a/fixed_peak_shape": True})
