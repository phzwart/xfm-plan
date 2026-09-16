# xfm-gold 0.2.1 — gold graph for the pistachio XRF+FTIR workflow

Hand-built, machine-checked. Two files carry the content; everything else reproduces or interrogates them.

`out/xfm-gold.grits.jsonld` — pygrits 0.6.2 bundle, 126 nodes: 1 plan, 91 quote entities (59 into papers and guides, 32 into SMAK 3.0.11 source) (each a `TextQuoteSelector` into a hashed source), 4 absence entities (searches actually run), 14 inferred/derived entities with rationale, 15 derivation activities, and the plan-graph entity with `payload_ref` to the YAML below. Passes `pygrits.validate()`; expands under the live context to 18 distinct predicates with nothing dropped.

`out/xfm-gold.plan.yaml` — sciplan 0.1.0 `PlanGraph`: 24 DataStates, 22 Transformations, 17 Parameters, 11 Rationales, 14 Assumptions, 4 Preconditions, 5 Decisions, 5 ValidationCriteria, 3 FailureModes, 3 Goals, 3 Workflows. Passes `linkml-validate` and loads through the generated Pydantic models. Every `Support.grit` resolves to a bundle node whose `how` matches the support label; every cross-reference resolves.

`build_gold.py` rebuilds both from the source texts and fails on any quote that is not found verbatim in the normalized source text, any absence that a word-bounded search contradicts, any dangling reference, and any label mismatch. `competency_queries.py` runs seven queries by traversal; `out/competency_results.txt` is its output.

## Sources and hashes

Published papers and the manuscript are not redistributed; `sources/SHA256SUMS` gives their hashes so the `source.sha256` fields can be checked against your copies. The pistachio manuscript is unpublished: its 30 quotes are in the bundle for the private graph only and must be stripped before any public release.

The three SMAK pages could not be fetched as raw HTML from the build host. `sources/*.txt` are the sentence-level extractions actually quoted; their hashes are what the bundle carries, and each file says so in its first line. At registration, re-fetch the raw HTML, hash it, and replace the `ContentRef`s — the `exact` strings will still match.

## What the build found

**A source-register error.** Pushie 2022 does not discuss dead-time correction anywhere in its text (only detector saturation and pile-up at double energy). The register said it did. The absence search `abs:pushie-deadtime` records this; the dead-time cluster is now SMAK (rarely, with a stated rule), XRF-Maps (carries the data), PyXRF (nothing), pistachio (silent, inferred not applied), Pushie (saturation only).

**The trace's two silences, handled differently.** Dead-time correction is in the workflow as a conditional step, with the trace's status recorded as `inferred: not applied` (`inf:pist-deadtime-not-applied`) resting on the absence in the manuscript plus SMAK's stated policy. Flux normalization is `unknown` (`inf:pist-i0-unknown`): the manuscript is silent, and the one route by which SMAK would have applied it automatically (quantification) was not taken. It is therefore not a step in the as-run workflow, and the workflow carries a `human_judgment` Gap saying so. Only the authors can close it.

**One Conflict.** On `spectrum_fit.parameters`: Solé 2007 fits ZERO/GAIN/NOISE/FANO globally; SMAK practice holds them fixed on a first pass. `would_settle` names the code read that would resolve it.

**Three quantification paths, none taken.** `quant_ratio_standard` (SMAK; assumes infinitely thin, same energy, distance, gain) and `quant_fp` (PyMca; assumes secondary fluorescence negligible; needs composition, density, thickness, geometry) are `alternative_to` each other under Decision `quantify`, whose third option, "relative only", is what the trace did. Q3 shows the four SMAK assumptions and the FP assumption each reach exactly one goal — `absolute_concentration` — which is the correct answer.

**Parameter default vs. used.** `blur_sigma` default 0.8 (SMAK Basics, quoted) with `deviates_from_default: true` at value 0.75 on step s3 (pistachio, quoted). The model carries both.

**Registration bound propagates.** `v/registration_rmse` has `derived_bound: ~80 µm` from `der:reg-bound`; `manual_registration.introduces` names it; goal `colocalization` is `validated_by` it. Q6 returns it.

## What the gold graph broke, or nearly

**`consumes` is a conjunction; simple-path queries treat it as a disjunction.** Q1 finds a path that reaches `channel_component_correlation` without passing `multifile_weighted_fa`, because that step consumes both `fa_scores` and `registered_stack` and the query follows either. The graph is a hypergraph on the input side; a correct reachability query must require all `consumes` of each step to be satisfied. This is the first thing to fix in the query layer, and Cypher will have the same problem unless written as an AND over inputs.

**Granularity had to be decided twice.** `ftir_preprocess` collapses four conceptual steps (atmospheric correction, SG denoise, rubberband baseline, mean normalization) with three stated rationales into one Transformation, because the trace applies them as a fixed sequence and no registered source separates them. Recorded as a Gap on `alternative_to`. The opposite case — `spectrum_fit` implemented as one PyMca call, one SMAK menu, and five XRF-Maps routines — is carried on `implemented_in` with a note that the five routines are not yet modelled as alternatives. The rule "smallest unit with its own Rationale" held, but it needed a note both times.

**Free-string qualifiers survived, barely.** `flux_normalized`, `dead_time_corrected`, `peak_fitted`, `registered` are consistent because one person wrote them. The moment XRF-Maps' `Elapsed_Livetime` and SMAK's `-blur` suffix are ingested by an extractor, an alias table on qualifiers (not just on nodes) will be needed. Flagged in the schema README; confirmed here.

**Validation steps as WorkflowSteps.** The 2900 cm⁻¹ re-run is modelled as step `s7v` with `role: validation` and a condition option on `norm_wavenumber`. It works, but a validation run that re-executes a transformation with different parameters is really a loop with an exit criterion of "results unchanged", and `loop_exit` on the main step would express that better. Either is defensible; noting the choice.

**`Support.field` for multivalued slots** was not a problem here — supports on `assumes`, `options`, `steps` are read as "supports the list". Index syntax can wait.


## 0.2.0 — what the SMAK 3.0.11 source settled

The code tier was applied as an overlay in `build_gold.py` (search for "code-tier overlay"): nine source files hashed individually inside the tarball (`0d368011…`), 32 quotes verified against the code text, nine derived entities each with a stated chain from quotes to conclusion.

**Dead time.** `#DT: corFF=FF*exp(tau*1e-6*ICR)` — the paralyzable model. Gap on `governing_relation` closed. More important: `dodt` is a Tk IntVar, default 0, applied when a channel is rendered or analysed. Dead-time correction in SMAK is a view-time toggle that leaves no record in saved channels. The trace's silence is therefore expected behaviour, not an omission; `inf:pist-deadtime-not-applied` stands, now with the default as evidence.

**I0.** Same pattern: `doI0c`, default 0, view-time. `Set I0 Channel` prefers `I0STRM`. Quantification divides by the chosen channel separately, and that channel may be `None`. `inf:pist-i0-unknown` stands — the manuscript reported scaled counts, and a view-time toggle would not have appeared in it either way.

**Time normalization.** `newdata=ad/Tdat*80000000`: the guide's 0.0000125 is 1/80 MHz. New Transformation `time_normalization`, gap closed.

**The fitting Conflict.** `activePyMCAConfig.cfg` ships with `fixedzero`, `fixedgain`, `fixednoise`, `fixedfano`, `fixedsum` all `= 1`, `stripalgorithm = 1` (SNIP, width 50), `hypermetflag = 7`, and the wrapper calls `FastXRFLinearFit.fitMultipleSpectra(weight=None, refit=True)`. SMAK's per-pixel fit is an unweighted linear least-squares of line areas under a fixed peak model; Solé's global non-linear optimisation is available but not the default. The Conflict now carries three positions and a `would_settle` that says it is settled for this version — it is a method-paper-versus-default difference, not a contradiction. New Assumption `fixed_peak_shape`; Q3 now ranks it with the three detector-physics assumptions as reaching all three goals.

**Quantification.** `mod=[fw.slope,fw.intc]; pred=np.poly1d(mod)` after dividing by I0: a linear calibration C = slope·(A/I0) + intercept, which is a ratio only when the intercept is zero. Governing relation replaced; the Transformation renamed accordingly.

**Registration.** Automatic stack alignment is `register_translation` (phase correlation) on high-pass-filtered images — translation only, which is exactly why the trace rejected it (rotation, 45° vs 90° geometry). Manual registration is `tps_warp2D`: a thin-plate-spline warp on landmark pairs, non-rigid, not affine. A third option exists (`align_images`: ORB features + homography, 2023) that neither guide nor trace mentions. No RMSE computation exists in the source, so the manuscript's RMSE was computed elsewhere; `v/registration_rmse` now carries a Gap saying so.

**Factor analysis.** Weighting is n_g = mean over the group's channels of Σ|loading|, w_g = max n / n_g, re-estimated for three cycles, with component sign flipped so the largest-magnitude loading is positive. Governing relation filled, gap closed. FA itself is `sklearn.decomposition.FactorAnalysis`.

**FTIR granularity.** `IR_MathClass.DoNorm` applies atmospheric correction, an optional resonant-Mie (RMieS) correction with a reference spectrum, Savitzky-Golay, a selectable baseline (default Rubberband; Concave Rubberband, AsLS, arPLS available), and normalization (default Mean) as five separately switchable calls. The granularity rule now has code behind it: `ftir_preprocess` is split into four Transformations on the as-run path plus `ftir_rmies_correction` as an offered step the trace did not use. The workflow steps are s7a–s7d with s7v as the 2900 cm⁻¹ validation re-run of s7d. One new `contested` gap: the trace attributes atmospheric correction to Omnic export, SMAK implements it too, and which ran is not stated.

Net: three gaps closed, four opened (all smaller), the Conflict settled for this version, five Transformations added, one split. The as-run workflow is now 16 steps.

## graphdb/ — loaded and queried (0.2.1)

`graphdb/load_plan.py` loads the plan and the bundle into a property graph — Kùzu (embedded) by default, Neo4j with `--neo4j bolt://host user pass`. Node labels are sciplan classes; Support, Gap, Conflict, Position, Implementation, Option, WorkflowStep and Term are materialised as nodes so evidence and aliases are traversable; grits and activities load beside them, so a claim field can be walked to its quote and its source hash (Q10). 463 nodes, 765 relationships for the gold graph.

`graphdb/queries.py` runs nine queries on Kùzu; `graphdb/queries.cypher` is the same set in Neo4j-native form. `graphdb/results.txt` is the output.

**Q1 is fixed.** AND-reachability is computed as a fixpoint: a Transformation fires only when every DataState it consumes is already reachable. Driven from Python, one Cypher per round (six rounds for the gold graph). The first run found a modelling error the simple-path version had hidden: from raw XRF spectra alone the co-localization claim was reachable, because `stack_alignment` was written as consuming only the XRF map while producing the multimodal registered stack. It now consumes both channel sets, and raw-spectra-alone reaches nine transformations and stops — the correct answer. The backward slice from the claim to the inputs carries seven assumptions.

Two Kùzu-specific notes: 0.11 cannot bind list parameters inside `all(x IN … WHERE x IN $list)`, so Q1 inlines list literals; and the database is a single file, so cleanup must remove a file, not a directory. Neither applies to Neo4j.

## Gap census

14 gaps: 11 `not_in_corpus` (mostly missing rationales for steps the trace applies without saying why, and the FA and dead-time governing relations), 1 `tacit` (pixels treated as independent samples in the effect-size statistics), 1 `human_judgment` (flux-normalization status of the trace), plus the FTIR granularity note. Each `not_in_corpus` gap names what would fill it; the Handbook and the Webb 2011 SMAK paper would close most.

## Next

1. Fix the reachability query to honour `consumes` as AND; re-run Q1.
2. Load the bundle and the plan into Neo4j (n10s from the JSON-LD expansion, or a small loader from the YAML) and port the seven queries to Cypher.
3. Register the raw SMAK HTML and re-hash.
4. aims-leaf hook: express the pistachio run as `ExperimentRun` + `WorkflowRun` with `plan_step` → these Transformation IRIs.
5. Gap-driven retrieval: the nine `not_in_corpus` gaps are the next corpus additions.
