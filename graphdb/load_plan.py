"""Load a sciplan PlanGraph (YAML) and a pygrits bundle (JSON-LD) into a property graph.

Backends: kuzu (embedded, default) or neo4j (via neo4j driver, --neo4j bolt://... user pass).
Node labels = sciplan class names; relationship types = sciplan slot names, upper-cased.
Support/Gap/Conflict are materialised as nodes so evidence is traversable.
"""
import argparse, json, sys, yaml
from pathlib import Path

REL_LISTS = {  # slot -> relationship type (list-valued refs)
    "consumes": "CONSUMES", "produces": "PRODUCES", "parameters": "HAS_PARAMETER",
    "requires": "REQUIRES", "assumes": "ASSUMES", "justified_by": "JUSTIFIED_BY",
    "validated_by": "VALIDATED_BY", "failure_modes": "HAS_FAILURE_MODE", "alternative_to": "ALTERNATIVE_TO",
    "serves": "SERVES", "starts_from": "STARTS_FROM", "ends_at": "ENDS_AT", "established_by": "ESTABLISHED_BY",
    "caused_by": "CAUSED_BY", "detected_by": "DETECTED_BY", "if_violated": "IF_VIOLATED", "bounded_by": "BOUNDED_BY",
}
REL_SCALARS = {"on_state": "ON_STATE", "set_by_decision": "SET_BY_DECISION", "claim_type": "CLAIM_TYPE"}
KIND_LABEL = {"data_states": "DataState", "parameters": "Parameter", "transformations": "Transformation", "workflows": "Workflow",
              "goals": "Goal", "preconditions": "Precondition", "assumptions": "Assumption", "decisions": "Decision",
              "rationales": "Rationale", "validation_criteria": "ValidationCriterion", "failure_modes": "FailureMode"}
SCALAR_PROPS = ["name", "description", "statement", "purpose", "governing_relation", "quantity", "units", "uncertainty",
                "kind", "default_value", "default_scope", "question", "has_rule", "requires_human", "resolved_by",
                "observable", "threshold", "derived_bound", "about", "checkable_by", "claim_statement", "version", "field"]


def extract(plan, bundle):
    nodes, rels = [], []
    def N(label, id_, props): nodes.append((label, id_, {k: v for k, v in props.items() if v is not None}))
    def R(src, typ, dst, props=None): rels.append((src, typ, dst, props or {}))
    # plan nodes
    for kind, label in KIND_LABEL.items():
        for n in plan.get(kind, []):
            props = {k: n.get(k) for k in SCALAR_PROPS if k in n}
            for lk in ("qualifiers", "indexed_by", "known_up_to", "adds_qualifiers", "removes_qualifiers", "preserves", "changes", "discards", "introduces"):
                if n.get(lk): props[lk] = json.dumps(n[lk])
            props["n_consumes"] = len(n.get("consumes", [])) if label == "Transformation" else None
            N(label, n["id"], props)
            for slot, typ in REL_LISTS.items():
                for x in n.get(slot, []): R(n["id"], typ, x)
            for slot, typ in REL_SCALARS.items():
                if n.get(slot): R(n["id"], typ, n[slot])
            for a in n.get("aliases", []):
                aid = f"{n['id']}#alias#{a['term']}@{a['used_by']}"
                N("Term", aid, {"term": a["term"], "used_by": a["used_by"], "note": a.get("note")}); R(aid, "DENOTES", n["id"])
            for i, s in enumerate(n.get("support", [])):
                sid = f"{n['id']}#support#{i}"
                N("Support", sid, {"field": s["field"], "how": s["how"], "note": s.get("note")}); R(n["id"], "HAS_SUPPORT", sid); R(sid, "EVIDENCED_BY", s["grit"])
            for i, g in enumerate(n.get("gaps", [])):
                gid = f"{n['id']}#gap#{i}"
                N("Gap", gid, {"field": g["field"], "kind": g["kind"], "note": g.get("note")}); R(n["id"], "HAS_GAP", gid)
                if g.get("search_grit"): R(gid, "SEARCHED_BY", g["search_grit"])
            for i, c in enumerate(n.get("conflicts", [])):
                cid = f"{n['id']}#conflict#{i}"
                N("Conflict", cid, {"field": c["field"], "would_settle": c.get("would_settle")}); R(n["id"], "HAS_CONFLICT", cid)
                for j, p in enumerate(c["positions"]):
                    pid = f"{cid}#pos#{j}"; N("Position", pid, {"statement": p["statement"]}); R(cid, "HAS_POSITION", pid); R(pid, "EVIDENCED_BY", p["grit"])
            for i, im in enumerate(n.get("implemented_in", [])):
                iid = f"{n['id']}#impl#{i}"
                N("Implementation", iid, {"software": im["software"], "version": im.get("version"), "component": im["component"], "note": im.get("note")}); R(n["id"], "IMPLEMENTED_IN", iid)
                if im.get("grit"): R(iid, "EVIDENCED_BY", im["grit"])
            if label == "Decision":
                for i, o in enumerate(n.get("options", [])):
                    oid = f"{n['id']}#opt#{i}"
                    N("Option", oid, {"label": o["label"], "sets_value": o.get("sets_value"), "consequence": o.get("consequence")}); R(n["id"], "HAS_OPTION", oid)
                    if o.get("selects_transformation"): R(oid, "SELECTS", o["selects_transformation"])
                    if o.get("sets_parameter"): R(oid, "SETS", o["sets_parameter"])
            if label == "Workflow":
                for s in n["steps"]:
                    sid = f"{n['id']}#step#{s['step_id']}"
                    N("WorkflowStep", sid, {"step_id": s["step_id"], "role": s.get("role", "main"), "loop_group": s.get("loop_group"), "condition_option": s.get("condition_option")})
                    R(n["id"], "HAS_STEP", sid); R(sid, "EXECUTES", s["transformation"])
                    for p in s.get("preceded_by", []): R(sid, "PRECEDED_BY", f"{n['id']}#step#{p}")
                    if s.get("condition"): R(sid, "CONDITIONED_ON", s["condition"])
                    if s.get("loop_exit"): R(sid, "LOOP_EXIT", s["loop_exit"])
                    for pv in s.get("parameter_values", []):
                        R(sid, "USES_VALUE", pv["parameter"], {"value": str(pv["value"]), "deviates": bool(pv.get("deviates_from_default", False))})
    # grits
    for g in bundle["@graph"]:
        t = g["@type"].split(":")[-1]
        props = {k: g.get(k) for k in ("how", "kind", "result", "summary", "rationale", "name")}
        if g.get("target"): props["exact"] = g["target"]["selector"].get("exact")
        if g.get("source"): props["source_uri"] = g["source"]["uri"]; props["source_sha256"] = g["source"]["sha256"]
        N("Grit" if t == "Entity" else t, g["@id"], props)
        if t == "Activity":
            for u in g.get("used", []): R(g["@id"], "USED", u)
            for o in g.get("generated", []): R(g["@id"], "GENERATED", o)
    return nodes, rels


def load_kuzu(db_path, nodes, rels):
    import kuzu
    db = kuzu.Database(db_path); con = kuzu.Connection(db)
    con.execute("CREATE NODE TABLE IF NOT EXISTS Node(id STRING PRIMARY KEY, label STRING, props STRING)")
    con.execute("CREATE REL TABLE IF NOT EXISTS Rel(FROM Node TO Node, type STRING, props STRING)")
    ids = set()
    for label, id_, props in nodes:
        ids.add(id_)
        con.execute("MERGE (n:Node {id: $id}) SET n.label = $label, n.props = $props", {"id": id_, "label": label, "props": json.dumps(props)})
    for s, t, d, p in rels:
        for x in (s, d):
            if x not in ids:
                con.execute("MERGE (n:Node {id: $id}) SET n.label = 'Unresolved', n.props = '{}'", {"id": x}); ids.add(x)
        con.execute("MATCH (a:Node {id:$s}),(b:Node {id:$d}) CREATE (a)-[:Rel {type:$t, props:$p}]->(b)", {"s": s, "d": d, "t": t, "p": json.dumps(p)})
    return con


def load_neo4j(uri, user, pw, nodes, rels):
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(uri, auth=(user, pw))
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        for label, id_, props in nodes:
            s.run(f"MERGE (n:{label} {{id:$id}}) SET n += $props", id=id_, props=props)
        for src, typ, dst, p in rels:
            s.run(f"MATCH (a {{id:$s}}),(b {{id:$d}}) MERGE (a)-[r:{typ}]->(b) SET r += $p", s=src, d=dst, p=p)
    return drv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("plan"); ap.add_argument("bundle")
    ap.add_argument("--kuzu", default="xfm.kuzu"); ap.add_argument("--neo4j", nargs=3, metavar=("URI", "USER", "PASS"))
    a = ap.parse_args()
    plan = yaml.safe_load(open(a.plan)); bundle = json.load(open(a.bundle))
    nodes, rels = extract(plan, bundle)
    print(f"{len(nodes)} nodes, {len(rels)} relationships")
    if a.neo4j: load_neo4j(*a.neo4j, nodes, rels); print("loaded into neo4j")
    else:
        import shutil, os
        for path in (a.kuzu, a.kuzu + ".wal"):
            if os.path.isdir(path): shutil.rmtree(path)
            elif os.path.exists(path): os.remove(path)
        load_kuzu(a.kuzu, nodes, rels); print("loaded into", a.kuzu)
