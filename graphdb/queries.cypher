// Competency queries — Neo4j-native form (real labels and relationship types, as load_plan.py --neo4j creates them).
// Q1 (AND-reachability) is a fixpoint and is driven from Python; see queries.py. With APOC it can be a single call:
//   CALL apoc.periodic.commit(...) is overkill; the Python loop is four lines and backend-neutral.

// Q1 — one round of the fixpoint: transformations whose every input is already reachable, and what they produce.
// Parameters: $reach (list of DataState ids reached so far), $fired (transformation ids already fired)
MATCH (t:Transformation)-[:CONSUMES]->(d:DataState)
WITH t, collect(d.id) AS ins
WHERE NOT t.id IN $fired AND all(x IN ins WHERE x IN $reach)
MATCH (t)-[:PRODUCES]->(o:DataState)
RETURN t.id AS t, collect(DISTINCT o.id) AS outs;

// Q2 — transformations that introduce an external dependence
MATCH (t:Transformation)
WHERE t.introduces IS NOT NULL AND any(w IN ['standard','matrix','model','table','assignment'] WHERE toLower(t.introduces) CONTAINS w)
RETURN t.id, t.introduces;

// Q3 — which assumption, if violated, invalidates the most goals
MATCH (w:Workflow)-[:HAS_STEP]->(:WorkflowStep)-[:EXECUTES]->(t:Transformation)-[:ASSUMES]->(a:Assumption),
      (w)-[:SERVES]->(g:Goal)
RETURN a.id AS assumption, collect(DISTINCT g.id) AS goals, count(DISTINCT g) AS n
ORDER BY n DESC, assumption;

// Q4 — steps that change the uncertainty representation
MATCH (i:DataState)<-[:CONSUMES]-(t:Transformation)-[:PRODUCES]->(o:DataState)
WITH t, collect(DISTINCT coalesce(i.uncertainty,'none')) AS ip, collect(DISTINCT coalesce(o.uncertainty,'none')) AS op
WHERE ip <> op
RETURN t.id, ip, op;

// Q5 — decisions: human-required, rule status, what each option selects
MATCH (d:Decision)
OPTIONAL MATCH (d)-[:HAS_OPTION]->(o:Option)-[:SELECTS]->(t:Transformation)
RETURN d.id, d.requires_human, d.has_rule, d.resolved_by, collect(DISTINCT t.id) AS selects;

// Q6 — validation outcomes that bound claims, and what they validate
MATCH (v:ValidationCriterion) WHERE v.derived_bound IS NOT NULL
OPTIONAL MATCH (x)-[:VALIDATED_BY]->(v)
RETURN v.id, v.derived_bound, collect(DISTINCT x.id) AS validates;

// Q7 — goals reached through steps carrying a gap or inferred/unknown support
MATCH (w:Workflow)-[:SERVES]->(g:Goal), (w)-[:HAS_STEP]->(:WorkflowStep)-[:EXECUTES]->(t:Transformation)
OPTIONAL MATCH (t)-[:HAS_GAP]->(gap:Gap)
OPTIONAL MATCH (t)-[:HAS_SUPPORT]->(s:Support) WHERE s.how IN ['inferred','unknown']
RETURN g.id AS goal, w.id AS workflow,
       collect(DISTINCT CASE WHEN gap IS NULL THEN NULL ELSE t.id END) AS steps_with_gaps,
       collect(DISTINCT CASE WHEN s IS NULL THEN NULL ELSE t.id END) AS steps_with_inferred_support;

// Q8 — evidence audit: per transformation, how many supports rest on quote / derived / inferred / unknown
MATCH (t:Transformation)-[:HAS_SUPPORT]->(s:Support)-[:EVIDENCED_BY]->(gr)
RETURN t.id, gr.how AS how, count(*) AS n ORDER BY t.id, how;

// Q9 — alias table
MATCH (a:Term)-[:DENOTES]->(n)
RETURN n.id AS canonical, collect({term: a.term, used_by: a.used_by}) AS terms;

// Q10 — full provenance of one claim field: transformation -> support -> grit -> (activity -> used grits) -> source hash
MATCH (t:Transformation {id:'xfm:t/deadtime_correction'})-[:HAS_SUPPORT]->(s:Support)-[:EVIDENCED_BY]->(g)
OPTIONAL MATCH (act:Activity)-[:GENERATED]->(g)
OPTIONAL MATCH (act)-[:USED]->(u)
RETURN s.field, s.how, g.id, g.how, g.exact, g.source_sha256, collect(DISTINCT u.id) AS derived_from;
