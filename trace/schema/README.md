# Vendored aims-leaf schema (patched)

`aims_leaf_ber_schema.yaml` is upstream aims-leaf at commit `967340b` with `aims-leaf-plan-hook.patch` applied (adds `plan_step`, `plan_graph`, `plan_parameter_values` on WorkflowRun; `plan_variable` on DataFile; `PlanParameterValue`; `pplan`/`sciplan` prefixes). The other YAML files are its unmodified imports.

This copy exists only so the trace validates without an upstream change. When aims-leaf carries these slots, delete this folder and validate against the installed package instead.
