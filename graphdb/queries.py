"""Competency queries in Cypher against the loaded plan graph.

Runs on Kùzu (generic Node/Rel tables; label/type are properties) by default.
`queries.cypher` holds the same queries in native Neo4j form (labels and relationship types).

Q1 is the AND-reachability query. `consumes` is a conjunction, so reachability is a fixpoint:
a Transformation fires only when every input DataState is already reachable. Neo4j has no
recursive-fixpoint construct without APOC, so Q1 is driven from Python in both backends:
one Cypher per round, iterate until no new DataStates appear. Everything else is a single Cypher.
"""
import json, sys
import kuzu

def q(con, cypher, **params):
    r = con.execute(cypher, params if params else None)
    cols = r.get_column_names(); out = []
    while r.has_next(): out.append(dict(zip(cols, r.get_next())))
    return out

def lit(xs):
    """Cypher list literal; Kùzu 0.11 cannot bind list parameters inside list predicates, so inline them."""
    return "[" + ",".join("'" + str(x).replace("'", "\\'") + "'" for x in xs) + "]" if xs else "['__none__']"

def short(x): return x.split("/")[-1] if isinstance(x, str) else x

# ---- Q1: AND-reachability from start states to a target state, with the firing sequence and assumptions
def q1_and_reachability(con, starts, target):
    reachable, fired, rounds = set(starts), [], 0
    while True:
        rows = q(con, f"""
            MATCH (t:Node {{label:'Transformation'}})-[:Rel {{type:'CONSUMES'}}]->(d:Node)
            WITH t, collect(d.id) AS ins
            WHERE NOT t.id IN {lit(fired)} AND size(ins) > 0 AND all(x IN ins WHERE x IN {lit(reachable)})
            MATCH (t)-[:Rel {{type:'PRODUCES'}}]->(o:Node)
            RETURN t.id AS t, collect(DISTINCT o.id) AS outs
        """)
        if not rows: break
        rounds += 1
        for r in rows:
            fired.append(r["t"]); reachable |= set(r["outs"])
    ok = target in reachable
    # assumptions on the fired steps that lie on a producer chain to the target (backward slice)
    slice_ = set(); frontier = {target}
    while frontier:
        rows = q(con, f"""
            MATCH (t:Node {{label:'Transformation'}})-[:Rel {{type:'PRODUCES'}}]->(d:Node)
            WHERE d.id IN {lit(frontier)} AND t.id IN {lit(fired)}
            MATCH (t)-[:Rel {{type:'CONSUMES'}}]->(i:Node)
            RETURN t.id AS t, collect(DISTINCT i.id) AS ins
        """)
        new = set()
        for r in rows:
            if r["t"] not in slice_:
                slice_.add(r["t"]); new |= set(r["ins"])
        frontier = new - set(starts)
        if not new: break
    assumptions = q(con, f"""
        MATCH (t:Node)-[:Rel {{type:'ASSUMES'}}]->(a:Node) WHERE t.id IN {lit(slice_)}
        RETURN DISTINCT a.id AS a ORDER BY a
    """)
    return dict(reachable=ok, rounds=rounds, fired_in_order=fired, backward_slice=sorted(slice_), assumptions=[r["a"] for r in assumptions])

# ---- Q2: transformations that introduce an external dependence
Q2 = """
MATCH (t:Node {label:'Transformation'})
WHERE t.props CONTAINS '"introduces"' AND (t.props CONTAINS 'standard' OR t.props CONTAINS 'matrix' OR t.props CONTAINS 'model' OR t.props CONTAINS 'table' OR t.props CONTAINS 'assignment')
RETURN t.id AS t, t.props AS props
"""
# ---- Q3: assumption -> goals reached through workflows whose steps assume it
Q3 = """
MATCH (w:Node {label:'Workflow'})-[:Rel {type:'HAS_STEP'}]->(:Node)-[:Rel {type:'EXECUTES'}]->(t:Node)-[:Rel {type:'ASSUMES'}]->(a:Node),
      (w)-[:Rel {type:'SERVES'}]->(g:Node)
RETURN a.id AS assumption, collect(DISTINCT g.id) AS goals, count(DISTINCT g) AS n
ORDER BY n DESC, assumption
"""
# ---- Q4: steps that change the uncertainty representation
Q4 = """
MATCH (i:Node)<-[:Rel {type:'CONSUMES'}]-(t:Node {label:'Transformation'})-[:Rel {type:'PRODUCES'}]->(o:Node)
WITH t, collect(DISTINCT i.props) AS ip, collect(DISTINCT o.props) AS op
RETURN t.id AS t, ip, op
"""
# ---- Q5: decisions, human-required, rule status
Q5 = """
MATCH (d:Node {label:'Decision'})
OPTIONAL MATCH (d)-[:Rel {type:'HAS_OPTION'}]->(o:Node)-[:Rel {type:'SELECTS'}]->(t:Node)
RETURN d.id AS decision, d.props AS props, collect(DISTINCT t.id) AS selects
"""
# ---- Q6: validation outcomes that bound claims
Q6 = """
MATCH (v:Node {label:'ValidationCriterion'})
WHERE v.props CONTAINS 'derived_bound'
OPTIONAL MATCH (x:Node)-[:Rel {type:'VALIDATED_BY'}]->(v)
RETURN v.id AS criterion, v.props AS props, collect(DISTINCT x.id) AS validates
"""
# ---- Q7: goals reachable only through steps carrying a gap or non-quote support
Q7 = """
MATCH (w:Node {label:'Workflow'})-[:Rel {type:'SERVES'}]->(g:Node),
      (w)-[:Rel {type:'HAS_STEP'}]->(:Node)-[:Rel {type:'EXECUTES'}]->(t:Node)
OPTIONAL MATCH (t)-[:Rel {type:'HAS_GAP'}]->(gap:Node)
OPTIONAL MATCH (t)-[:Rel {type:'HAS_SUPPORT'}]->(s:Node) WHERE s.props CONTAINS '"inferred"' OR s.props CONTAINS '"unknown"'
RETURN g.id AS goal, w.id AS workflow, collect(DISTINCT CASE WHEN gap IS NULL THEN NULL ELSE t.id END) AS steps_with_gaps,
       collect(DISTINCT CASE WHEN s IS NULL THEN NULL ELSE t.id END) AS steps_with_inferred_support
"""
# ---- Q8: evidence audit — every claim field and the grit kind behind it, per transformation
Q8 = """
MATCH (t:Node {label:'Transformation'})-[:Rel {type:'HAS_SUPPORT'}]->(s:Node)-[:Rel {type:'EVIDENCED_BY'}]->(gr:Node)
RETURN t.id AS t, s.props AS support, gr.label AS grit_kind, gr.props AS grit
"""
# ---- Q9: alias table
Q9 = """
MATCH (a:Node {label:'Term'})-[:Rel {type:'DENOTES'}]->(n:Node)
RETURN n.id AS canonical, collect(a.props) AS terms
"""

if __name__ == "__main__":
    con = kuzu.Connection(kuzu.Database(sys.argv[1] if len(sys.argv) > 1 else "xfm.kuzu"))
    P = lambda s: json.loads(s) if isinstance(s, str) else s

    print("Q1 AND-reachability: raw spectra + FTIR + optical -> co-localization claim")
    r = q1_and_reachability(con, ["xfm:ds/mca_spectrum", "xfm:ds/ftir_spectrum", "xfm:ds/optical_image"], "xfm:ds/claim_colocalization")
    print("  reachable:", r["reachable"], "| rounds:", r["rounds"])
    print("  fired:", [short(t) for t in r["fired_in_order"]])
    print("  backward slice:", [short(t) for t in r["backward_slice"]])
    print("  assumptions on slice:", [short(a) for a in r["assumptions"]])
    print("Q1b AND-reachability from raw spectra ALONE -> co-localization claim")
    r = q1_and_reachability(con, ["xfm:ds/mca_spectrum"], "xfm:ds/claim_colocalization")
    print("  reachable:", r["reachable"], "| fired:", [short(t) for t in r["fired_in_order"]])

    print("\nQ2 external dependence")
    for row in q(con, Q2): print("  ", short(row["t"]), "->", P(row["props"]).get("introduces"))

    print("\nQ3 assumption -> goals")
    for row in q(con, Q3): print("  ", short(row["assumption"]), row["n"], sorted(short(g) for g in row["goals"]))

    print("\nQ4 uncertainty representation changes")
    for row in q(con, Q4):
        ip = sorted({P(p).get("uncertainty", "none") for p in row["ip"]}); op = sorted({P(p).get("uncertainty", "none") for p in row["op"]})
        if ip != op: print("  ", short(row["t"]), ip, "->", op)

    print("\nQ5 decisions")
    for row in q(con, Q5):
        p = P(row["props"]); print("  ", short(row["decision"]), "| human:", p.get("requires_human"), "| rule:", p.get("has_rule"), "| selects:", [short(t) for t in (row["selects"] or []) if t])

    print("\nQ6 validation outcomes that bound claims")
    for row in q(con, Q6): print("  ", short(row["criterion"]), "->", P(row["props"]).get("derived_bound"), "| validates:", [short(x) for x in (row["validates"] or []) if x])

    print("\nQ7 goals through gapped / inferred steps")
    for row in q(con, Q7): print("  ", short(row["goal"]), "via", short(row["workflow"]), "| gaps:", sorted(short(x) for x in (row["steps_with_gaps"] or []) if x), "| inferred:", sorted(short(x) for x in (row["steps_with_inferred_support"] or []) if x))

    print("\nQ8 evidence audit (counts per transformation by grit kind)")
    from collections import Counter, defaultdict
    audit = defaultdict(Counter)
    for row in q(con, Q8): audit[short(row["t"])][P(row["grit"]).get("how") or row["grit_kind"]] += 1
    for t, c in sorted(audit.items()): print("  ", t, dict(c))

    print("\nQ9 alias table")
    for row in q(con, Q9): print("  ", short(row["canonical"]), "<-", [(P(x)["term"], P(x)["used_by"]) for x in row["terms"]])
