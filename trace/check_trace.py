"""Check an aims-leaf execution trace against a sciplan plan graph.

Usage: python check_trace.py plan.yaml trace.aimsleaf.yaml [--workflow xfm:w/...]

Checks
  1. every WorkflowRun.plan_step is a Transformation in the plan; every DataFile.plan_variable is a DataState
  2. per run: input files' plan_variables ⊆ Transformation.consumes; output files' ⊆ produces (and all consumes present)
  3. ordering: the run DAG (file produced by A, consumed by B ⇒ A < B) is consistent with the plan Workflow's preceded_by
  4. coverage: plan steps with no run — conditional/validation steps are expected absences, main steps are findings;
     runs whose transformation is not in the chosen plan Workflow
  5. parameters: values vs Parameter defaults; deviates_from_default flag consistent with the plan's default_value
  6. preconditions: for each run, Preconditions of its Transformation and whether an upstream run established them
"""
import sys, yaml
from collections import defaultdict

plan = yaml.safe_load(open(sys.argv[1])); trace = yaml.safe_load(open(sys.argv[2]))
wf_id = sys.argv[sys.argv.index("--workflow") + 1] if "--workflow" in sys.argv else None
idx = {n["id"]: n for k, v in plan.items() if isinstance(v, list) for n in v}
T = {t["id"]: t for t in plan["transformations"]}
DS = {d["id"]: d for d in plan["data_states"]}
P = {p["id"]: p for p in plan["parameters"]}
wf = next(w for w in plan["workflows"] if (wf_id is None or w["id"] == wf_id))
short = lambda x: x.split("/")[-1] if isinstance(x, str) else x

runs = {r["id"]: r for r in trace["workflow_runs"]}
files = {f["id"]: f for f in trace["data_files"]}
ins = defaultdict(list); outs = defaultdict(list)
for a in trace.get("workflow_input_associations", []): ins[a["workflow_id"]].append(a["file_id"])
for a in trace.get("workflow_output_associations", []): outs[a["workflow_id"]].append(a["file_id"])

findings, ok = [], []
def F(kind, msg): findings.append((kind, msg))

# 1 resolution
for r in runs.values():
    if r.get("plan_step") not in T: F("ERROR", f"{r['id']}: plan_step {r.get('plan_step')} is not a Transformation")
for f in files.values():
    if f.get("plan_variable") not in DS: F("ERROR", f"{f['id']}: plan_variable {f.get('plan_variable')} is not a DataState")

# 2 typed data flow
for rid, r in runs.items():
    t = T.get(r.get("plan_step"))
    if not t: continue
    in_vars = {files[f]["plan_variable"] for f in ins[rid]}; out_vars = {files[f]["plan_variable"] for f in outs[rid]}
    extra_in = in_vars - set(t["consumes"]); missing_in = set(t["consumes"]) - in_vars
    extra_out = out_vars - set(t["produces"])
    if extra_in: F("ERROR", f"{short(rid)} consumed {sorted(map(short, extra_in))} not in {short(t['id'])}.consumes")
    if missing_in: F("WARN", f"{short(rid)} did not consume {sorted(map(short, missing_in))} required by {short(t['id'])}.consumes")
    if extra_out: F("ERROR", f"{short(rid)} produced {sorted(map(short, extra_out))} not in {short(t['id'])}.produces")
    if not (extra_in or missing_in or extra_out): ok.append(f"dataflow {short(rid)} ⊆ {short(t['id'])}")

# 3 ordering
producer = {}
for rid in runs:
    for f in outs[rid]: producer.setdefault(f, []).append(rid)
run_before = defaultdict(set)  # b -> {a} : a must precede b
for rid in runs:
    for f in ins[rid]:
        for a in producer.get(f, []):
            if a != rid: run_before[rid].add(a)
steps = {s["step_id"]: s for s in wf["steps"]}
step_of_t = defaultdict(list)
for s in wf["steps"]: step_of_t[s["transformation"]].append(s["step_id"])
def plan_preds(sid, seen=None):
    seen = seen or set()
    for p in steps[sid].get("preceded_by", []):
        if p not in seen: seen.add(p); plan_preds(p, seen)
    return seen
for b, As in run_before.items():
    tb = runs[b]["plan_step"]
    for a in As:
        ta = runs[a]["plan_step"]
        # is there some plan step for ta that precedes some plan step for tb?
        consistent = any(sa in plan_preds(sb) for sb in step_of_t.get(tb, []) for sa in step_of_t.get(ta, []))
        (ok.append if consistent else (lambda m: F("WARN", m)))(f"order {short(a)} < {short(b)} matches plan" if consistent else f"trace has {short(ta)} before {short(tb)} but plan workflow has no preceded_by path between them")

# 4 coverage
run_ts = {r["plan_step"] for r in runs.values()}
for s in wf["steps"]:
    if s["transformation"] not in run_ts:
        role = s.get("role", "main")
        if role in ("conditional", "validation", "loop"): F("INFO", f"plan step {s['step_id']} ({short(s['transformation'])}, {role}, option={s.get('condition_option')}) has no run — expected absence")
        else: F("WARN", f"plan step {s['step_id']} ({short(s['transformation'])}, main) has no run")
plan_ts = {s["transformation"] for s in wf["steps"]}
for r in runs.values():
    if r["plan_step"] in T and r["plan_step"] not in plan_ts: F("WARN", f"{short(r['id'])} executes {short(r['plan_step'])}, which is not a step of {short(wf['id'])}")

# 5 parameters
for r in runs.values():
    for pv in r.get("plan_parameter_values", []):
        p = P.get(pv["parameter"])
        if not p: F("ERROR", f"{short(r['id'])}: parameter {pv['parameter']} not in plan"); continue
        default = p.get("default_value")
        if default is not None:
            dev_actual = str(pv["value"]).strip() != str(default).split(" ")[0].strip()
            if "deviates_from_default" in pv and pv["deviates_from_default"] != dev_actual:
                F("WARN", f"{short(r['id'])}: {short(p['id'])}={pv['value']} vs default {default}: flag says deviates={pv['deviates_from_default']}, computed {dev_actual}")
            else: ok.append(f"param {short(r['id'])} {short(p['id'])}={pv['value']} (default {default}, deviates={dev_actual})")
        else: ok.append(f"param {short(r['id'])} {short(p['id'])}={pv['value']} (no plan default)")

# 6 preconditions
for r in runs.values():
    t = T.get(r["plan_step"])
    if not t: continue
    for pre in t.get("requires", []):
        est = set(idx[pre].get("established_by", []))
        upstream = set()
        stack = [r["id"]]; seen = set()
        while stack:
            x = stack.pop()
            for a in run_before.get(x, []):
                if a not in seen: seen.add(a); upstream.add(runs[a]["plan_step"]); stack.append(a)
        if est and not (est & upstream): F("WARN", f"{short(r['id'])} requires {short(pre)} (established by {sorted(map(short, est))}); no such upstream run in the trace")
        elif est: ok.append(f"precondition {short(pre)} for {short(r['id'])} established upstream")
        else: F("INFO", f"{short(r['id'])} requires {short(pre)} ({idx[pre]['checkable_by']}); not establishable by a plan step — check metadata")

print(f"plan {plan['field']} {plan['version']} workflow {wf['id']} | trace {trace['id']}: {len(runs)} runs, {len(files)} files")
print(f"\n{len(ok)} checks passed")
for kind in ("ERROR", "WARN", "INFO"):
    items = [m for k, m in findings if k == kind]
    if items:
        print(f"\n{kind} ({len(items)})"); [print("  ", m) for m in items]
sys.exit(1 if any(k == "ERROR" for k, _ in findings) else 0)
