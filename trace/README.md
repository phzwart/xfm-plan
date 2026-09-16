# trace/ — execution trace of the pistachio analysis, checked against the plan graph

    linkml-validate -s trace/schema/aims_leaf_ber_schema.yaml -C Dataset trace/pistachio-run.aimsleaf.yaml
    python trace/check_trace.py graph/xfm-gold.plan.yaml trace/pistachio-run.aimsleaf.yaml --workflow xfm:w/pistachio_multimodal

`pistachio-run.aimsleaf.yaml` is the analysis as an aims-leaf Dataset with `plan_step` / `plan_variable` links into xfm-gold. `check_trace.py` checks resolution, typed data flow, ordering, coverage, parameter defaults and preconditions; `check_results.txt` is the current result (40 pass, one expected absence). `schema/` is a vendored, patched aims-leaf schema — see its README.
