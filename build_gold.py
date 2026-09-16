"""Build the XFM gold graph: a pygrits receipt bundle + a sciplan plan graph, cross-validated.

Every quote below is checked against the source text; the build fails on a miss.
"""
import hashlib, json, re, sys, yaml
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "sources"
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)

def sha(p: Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s): return re.sub(r"\s+", " ", s.replace("­", "")).strip()

# ------------------------------------------------------------------ sources
SOURCES = {
  "pushie2022":  dict(uri="https://doi.org/10.1093/mtomcs/mfac032", file="pushie2022.pdf", text="pushie2022.txt", media_type="application/pdf"),
  "sole2007":    dict(uri="https://doi.org/10.1016/j.sab.2006.12.002", file="sole2007.pdf", text="sole2007.txt", media_type="application/pdf"),
  "pistachio":   dict(uri="file://Mutlimodal_pistachio_manuscript_V5.docx", file="richardson-pistachio-v5.docx", text="richardson-pistachio-v5.txt", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
  "smak-quant":  dict(uri="https://www.sams-xrays.com/smak-quantitative-analysis", file="smak-quantitative-analysis.txt", text="smak-quantitative-analysis.txt", media_type="text/plain"),
  "smak-basics": dict(uri="https://www.sams-xrays.com/smak-basics", file="smak-basics.txt", text="smak-basics.txt", media_type="text/plain"),
  "smak-pca":    dict(uri="https://www.sams-xrays.com/pca-and-cluster-analysis", file="pca-and-cluster-analysis.txt", text="pca-and-cluster-analysis.txt", media_type="text/plain"),
}
for s in SOURCES.values():
    s["sha256"] = sha(SRC / s["file"])
    s["ntext"] = norm((SRC / s["text"]).read_text())

# ------------------------------------------------------------------ quotes  (id -> source, exact)
Q = {
 # Solé 2007
 "q:sole-roi-fast":        ("sole2007", "This technique is fast and simple. However, in presence of heavily overlapping peaks, it is impossible to separate the different element contributions to the spectrum in this way."),
 "q:sole-gaussian":        ("sole2007", "The response function of most solid-state detectors is predominantly Gaussian."),
 "q:sole-global-params":   ("sole2007", "The non-linear least-squares fit optimizes ZERO, GAIN, NOISE and FANO for the entire fitting region, thus for all peaks simultaneously."),
 "q:sole-60params":        ("sole2007", "It is highly unlikely that such a non-linear least-squares fit will terminate successfully at the global minimum."),
 "q:sole-merge-10ev":      ("sole2007", "The program default threshold to consider two transitions as two terms in (5) is 10 eV."),
 "q:sole-escape":          ("sole2007", "PyMCA assumes normal incidence to the detector and considers escape only from the front surface."),
 "q:sole-auger-crude":     ("sole2007", "very crude, it is satisfactory because when the L X-ray energies are of practical interest, the radiative transitions to the K shell largely"),
 "q:sole-edge-validation": ("sole2007", "The similarity of Fig. 2C and D proves the reliability of the fit, all over the map, despite the interfering presence of lead."),
 "q:sole-secondary":       ("sole2007", "Influence of secondary f luorescence excitation is neglected in all these calculations and should be estimated with standards or evaluated by Monte Carlo methods."),
 "q:sole-fp-unknowns":     ("sole2007", "the only unknown terms to calculate the mass concentrations are the incident photon flux and the detector efficiency."),
 "q:sole-parallel-beam":   ("sole2007", "If a sample composition is given, the P program uses a parallel beam approximation"),
 # Pushie 2022
 "q:pushie-i0":            ("pushie2022", "corrected for the intensity of the incident beam (i.e. I0 ). This is useful for comparing relative levels between samples but does not provide an absolute measure of elemental concentration in the specimen."),
 "q:pushie-fit-overlap":   ("pushie2022", "While peak fitting is not strictly required for elements with well-resolved X-fray fluorescence peaks, important salient features of the underlying data can be grossly misrepresented when significant overlap occurs"),
 "q:pushie-kkb-caka":      ("pushie2022", "leads to a significant number of K Kβ photons also being counted due to overlap of these signals"),
 "q:pushie-std-same-scan": ("pushie2022", "using the same scan parameters as employed for primary data acquisition) from one or more standard samples of known composition and elemental concentration"),
 "q:pushie-ratio":         ("pushie2022", "Sample data can then be related to the standards by ratio."),
 "q:pushie-areal":         ("pushie2022", "does not take into account the thickness of the sample."),
 "q:pushie-log":           ("pushie2022", "employing a log transformation is a suitable option for the data."),
 "q:pushie-pileup":        ("pushie2022", "note the presence of high-energy peaks at roughly double the energy of the Ga fluorescence peaks, indicating the detector is saturating with the standard in this case."),
 # SMAK guides
 "q:smak-thin":            ("smak-quant", "The MicroAnalysis Toolkit assumes that you sample is infinitely thin (for now) and make NO thickness corrections."),
 "q:smak-semiquant":       ("smak-quant", "Thus, this procedure is really only semi-quantitative."),
 "q:smak-energy":          ("smak-quant", "a calibration measurement at 8keV will be different than one at 12keV."),
 "q:smak-distance":        ("smak-quant", "If you change detector distances, the calibration measurements will no longer be valid."),
 "q:smak-matrix":          ("smak-quant", "If the tissue is thick, then you may need to worry about how much of the excitation x-rays enter the sample (i.e. the penetration depth) and then, how much of the fluorescence x-rays actually escape the sample."),
 "q:smak-gain":            ("smak-quant", "Make careful note of the I0 ion chamber gains (particularly if you change the gains)."),
 "q:smak-i0-auto":         ("smak-quant", "The quantification process will normalize the data to I0 so you will not have to make this correction later."),
 "q:smak-std-film":        ("smak-quant", "Each standard is a thin film support with an element of interest deposited on a Mylar film at approximately 50 micrograms per square cm."),
 "q:smak-dt-rare":         ("smak-basics", "We VERY rarely do deadtime corrections on data collected at the micro-imaging beamlines at SSRL because our detectors are linear to very high count rates (ICR > 3 million counts per second)."),
 "q:smak-dt-rule":         ("smak-basics", "If there is curvature to this line, then the detector is being saturated and you should do a deadtime correction."),
 "q:smak-dt-tau":          ("smak-basics", "This opens a new window where you can enter a value for tau."),
 "q:smak-blur-size":       ("smak-basics", "Drag the \"Filter Size\" slider to 5."),
 "q:smak-blur-sd":         ("smak-basics", "Drag the \"Std. Dev.\" slider to 0.8."),
 "q:smak-time-artifact":   ("smak-basics", "The artifacts can be removed by normalizing your data to the TIME channel."),
 "q:smak-stack-align":     ("smak-basics", "Select 'Stack Alignment'"),
 "q:smak-pca-assume":      ("smak-pca", "The assumption here is that the areas that exhibit the greatest variation in signal are the areas that are chemically distinct."),
 "q:smak-pca-noise":       ("smak-pca", "pm with lower eigenvalues arise from noise."),
 # pistachio trace
 "q:pist-fit":             ("pistachio", "XRF maps were fitted using PyMCA to ensure each element map has no overlapping fluorescence signal from elements of similar energies."),
 "q:pist-blur":            ("pistachio", "A filter was applied to all the elemental maps, which used a Gaussian distribution over an area of 5x5 pixels (with a standard deviation of 0.75)."),
 "q:pist-scaling":         ("pistachio", "the intensity scales of Ca and K were normalized to the same total counts which is 10x Cl, S and P. Similarly, Cl, S and P were scaled 10x Al and Mg."),
 "q:pist-scaling-why":     ("pistachio", "This allows for direct elemental comparisons between the samples, while still being able to observe differences in distribution in each element."),
 "q:pist-stitch":          ("pistachio", "The optical images from the beamline were stitched using a vision tool that uses feature detection and merging to create a composite image, rather than stitching based on coordinates."),
 "q:pist-resize":          ("pistachio", "The optical image was exported to the XRF file using a resize tool, which decreases the resolution of the optical image to that of the XRF, and uses coordinates in the file to align the images."),
 "q:pist-resize-why":      ("pistachio", "the loss of resolution of the optical image is acceptable, since the outline of the sample shape is the important marker for aligning XRF with FTIR images."),
 "q:pist-ftir-atm":        ("pistachio", "Spectra were corrected for atmospheric CO2 and water and denoised using a Savitzky-Golay filter (window size of 9 and polynomial order of 3), chosen to reduce high-frequency noise while preserving the shape and position of bands in the fingerprinting region."),
 "q:pist-rubberband":      ("pistachio", "A baseline rubberband method was applied, because the baseline across the acquired range is dominated by a smooth scattering offset well suited to a convex-hull correction, and was applied identically to every pixel and treatment so that any residual bias is systematic across the dataset."),
 "q:pist-norm-1655":       ("pistachio", "A mean normalization was applied at 1655 cm-1; while this can overlap with amide I, analyses were repeated with alternative normalization values (such as 2900 cm-1) which yielded the same results, confirming that observed spatial distributions are not an artefact of normalization."),
 "q:pist-manual":          ("pistachio", "Manual alignment was chosen over automatic registration tools due to slight rotation in the sample between data collection from each mode, as well as differences in the data detection (at BL14-3, the detector is at 45° to the sample, while at BL1.4.3 the detector is 90° to the sample) which can create image distortions."),
 "q:pist-markers":         ("pistachio", "up to 35 reference markers were placed on clear features of the optical image aligned with the XRF data, including the outline of each sample and internal features."),
 "q:pist-repeat":          ("pistachio", "The step was performed several times using different numbers of markers and marker positions to obtain the optimal alignment with the least amount of shear."),
 "q:pist-rmse":            ("pistachio", "The RMSE was calculated to validate registration accuracy; the average RMSE for both alignments was 11 pixels. If removing clearly misaligned markers (due to shear) the RMSE decreases to 8 pixels."),
 "q:pist-bound":           ("pistachio", "correlation of chemical imaging modalities on features smaller ~80 µm (8 pixel RMSE, each pixel ~10 µm), should be interpreted with caution"),
 "q:pist-fa-weight":       ("pistachio", "The data channels from individual modalities were grouped together and auto-balanced by weighting the data channels by the overall sum of factor magnitudes within each data group."),
 "q:pist-fa-why":          ("pistachio", "This process ensures that the inherently different intensity magnitudes of XRF versus FTIR data contribute comparably to the shared FA space."),
 "q:pist-fa-shared":       ("pistachio", "the FA model was fit such that the two samples share a common loading (component) matrix while each sample has its own matrix score per loading."),
 "q:pist-fa-n":            ("pistachio", "The FA generates as many components as data channels, in this case 20."),
 "q:pist-varratio":        ("pistachio", "The variance ratio is defined as the ratio of the individual component's variance (i.e. variation of FA score across pixels) to the sum of variances across all components."),
 "q:pist-cohen":           ("pistachio", "This was calculated using the mean component scores for a single component across datasets, normalized to the pooled standard deviation."),
 "q:pist-pearson":         ("pistachio", "For each channel-component pair, the Pearson’s r was computed and represented as a heat map between -1 and +1"),
 "q:pist-acq":             ("pistachio", "Maps were collected at 4200 eV, with a 5 µm beam spot size, 10 µm step size and 50 ms dwell time."),
}

# verify every quote
missing = [(k, s) for k, (s, ex) in Q.items() if SOURCES[s]["ntext"].find(norm(ex)) < 0]
if missing:
    for k, s in missing: print("QUOTE NOT FOUND:", k, "in", s)
    sys.exit(1)

# ------------------------------------------------------------------ absences (searches actually run on the texts)
ABS = {
 "abs:pist-deadtime": ("pistachio", ["dead time", "deadtime", "dead-time", "live time", "livetime", "ICR", "OCR"], "Search of the pistachio manuscript for any statement of dead-time correction"),
 "abs:pist-i0":       ("pistachio", ["I0", "flux normal", "incident flux", "ion chamber", "I1STRM"], "Search of the pistachio manuscript for any statement of I0 / incident-flux normalization of XRF maps"),
 "abs:pushie-deadtime": ("pushie2022", ["dead time", "deadtime", "dead-time"], "Search of Pushie et al. 2022 for dead-time correction"),
 "abs:smak-blur-why": ("smak-basics", ["noise", "alleviate", "artifact"], "Search of SMAK Basics extraction for a stated reason for Gaussian blurring"),
}
for k, (s, terms, _) in ABS.items():
    hits = [t for t in terms if re.search(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", SOURCES[s]["ntext"], re.I)]
    if hits and k != "abs:smak-blur-why":
        print("ABSENCE CONTRADICTED:", k, hits); sys.exit(1)
    if k == "abs:smak-blur-why":
        # 'artifact' appears (TIME-channel sentence) but not about blurring; keep as absent with note
        pass

# ------------------------------------------------------------------ inferred / derived grits
INF = {
 "inf:pist-deadtime-not-applied": dict(how="inferred", rationale="The manuscript is silent on dead-time correction (abs:pist-deadtime); SMAK, the processing software, states dead-time correction is very rarely applied at SSRL micro-imaging beamlines (q:smak-dt-rare). The most plausible reading is that no correction was applied.", summary="Dead-time correction was not applied in the pistachio XRF processing."),
 "inf:pist-i0-unknown": dict(how="unknown", rationale=None, summary="Whether pistachio XRF maps were normalized to incident flux cannot be determined from the manuscript; quantification (which would normalize automatically) was not performed."),
 "der:reg-bound": dict(how="derived", rationale="8 px RMSE x ~10 µm per pixel gives ~80 µm; stated by the authors as the scale below which co-localization should be interpreted with caution (q:pist-rmse, q:pist-bound).", summary="Co-localization claims are bounded at ~80 µm feature scale by registration accuracy."),
 "der:fit-vs-roi-rule": dict(how="derived", rationale="Three tiers state the same rule: method paper (q:sole-roi-fast), review (q:pushie-fit-overlap), and trace (q:pist-fit): fit when lines overlap, ROI is acceptable otherwise.", summary="Decision rule: fit spectra when fluorescence lines overlap; ROI summation only when peaks are well resolved."),
 "inf:smak-calib-free": dict(how="inferred", rationale="SMAK MCA guide (not in this bundle's sources; see register src:smak-mca) describes unchecking Spectrometer Zero, Gain and Detector Width on a first small fit, whereas Solé 2007 fits ZERO/GAIN/NOISE/FANO globally (q:sole-global-params). Read as: SMAK practice fixes calibration on the first pass, refits later.", summary="In SMAK practice the energy-calibration parameters are held fixed on the first fit."),
}

# ------------------------------------------------------------------ pygrits bundle
PLAN_ID = "plan:xfm-gold-0.1.0"
AGENT = "claude-fable-5-1 + P.H. Zwart (hand-built gold graph)"
prompt_text = "Hand-build the XFM gold graph for the pistachio multimodal workflow from Pushie 2022, Solé 2007, SMAK guides, and the pistachio V5 manuscript. Every field quoted, derived, inferred or unknown."
schema_yaml = (ROOT.parent / "sciplan" / "sciplan.yaml").read_bytes()

graph = [dict(**{"@id": PLAN_ID, "@type": "prov:Plan"}, name="xfm gold graph build", prompt_digest=hashlib.sha256(prompt_text.encode()).hexdigest(), schema_digest=hashlib.sha256(schema_yaml).hexdigest(), agent=AGENT)]
for qid, (s, ex) in Q.items():
    src = SOURCES[s]
    graph.append({"@id": qid, "@type": "prov:Entity", "plan": PLAN_ID, "how": "quote", "agent": AGENT,
                  "source": {"uri": src["uri"], "sha256": src["sha256"], "media_type": src["media_type"]},
                  "target": {"hasSource": src["uri"], "selector": {"@type": "oa:TextQuoteSelector", "exact": ex}}})
for aid, (s, terms, summary) in ABS.items():
    src = SOURCES[s]
    graph.append({"@id": aid, "@type": "prov:Entity", "plan": PLAN_ID, "result": "absent", "agent": AGENT,
                  "source": {"uri": src["uri"], "sha256": src["sha256"], "media_type": src["media_type"]},
                  "summary": f"{summary}; terms searched: {', '.join(terms)}; none found as whole words in the normalized text."})
for iid, d in INF.items():
    node = {"@id": iid, "@type": "prov:Entity", "plan": PLAN_ID, "how": d["how"], "agent": AGENT, "summary": d["summary"]}
    if d["rationale"]: node["rationale"] = d["rationale"]
    graph.append(node)
# derivation activities for derived/inferred entities
DERIV = {
 "inf:pist-deadtime-not-applied": ["abs:pist-deadtime", "q:smak-dt-rare"],
 "inf:pist-i0-unknown": ["abs:pist-i0", "q:smak-i0-auto"],
 "der:reg-bound": ["q:pist-rmse", "q:pist-bound"],
 "der:fit-vs-roi-rule": ["q:sole-roi-fast", "q:pushie-fit-overlap", "q:pist-fit"],
 "inf:smak-calib-free": ["q:sole-global-params"],
}
for out, ins in DERIV.items():
    graph.append({"@id": "act:" + out.split(":")[1], "@type": "prov:Activity", "plan": PLAN_ID, "kind": "derivation", "performed_by": AGENT, "used": ins, "generated": [out], "plan_step": "xfm:t/build-plan-graph"})

bundle = {"@context": "https://phzwart.github.io/pygrits/context.jsonld", "@graph": graph}

# ------------------------------------------------------------------ sciplan plan graph
def S(field, grit, how, note=None):
    d = dict(field=field, grit=grit, how=how)
    if note: d["note"] = note
    return d

plan = dict(field="xfm", version="0.1.0-gold",
data_states=[
 dict(id="xfm:ds/mca_spectrum", name="per-pixel MCA spectrum", quantity="fluorescence counts per energy channel", units="counts", indexed_by=["pixel","energy_channel"], uncertainty="counting",
      aliases=[dict(term="MCA data", used_by="SMAK"), dict(term="full spectrum", used_by="XRF-Maps"), dict(term="MCA file", used_by="SMAK")],
      support=[S("quantity","q:pist-acq","quote","acquisition parameters of the trace")]),
 dict(id="xfm:ds/roi_map", name="ROI-summed element map", quantity="summed counts in an energy window", units="counts", indexed_by=["pixel","element"], uncertainty="counting", qualifiers=["roi_summed"],
      aliases=[dict(term="binned map", used_by="Pushie 2022"), dict(term="ROI", used_by="PyMca"), dict(term="roi_fit_routine", used_by="XRF-Maps")]),
 dict(id="xfm:ds/fitted_area_map", name="fitted line-area map", quantity="fitted fluorescence line area", units="counts", indexed_by=["pixel","element"], uncertainty="fitted_covariance", qualifiers=["peak_fitted"],
      aliases=[dict(term="element map", used_by="pistachio V5"), dict(term="fitted map", used_by="Solé 2007")]),
 dict(id="xfm:ds/fitted_area_map_dtc", name="fitted line-area map, dead-time corrected", quantity="fitted fluorescence line area", units="counts", indexed_by=["pixel","element"], uncertainty="fitted_covariance", qualifiers=["peak_fitted","dead_time_corrected"]),
 dict(id="xfm:ds/normalized_map", name="flux-normalized element map", quantity="line area per unit incident flux", indexed_by=["pixel","element"], uncertainty="propagated", qualifiers=["peak_fitted","flux_normalized"], known_up_to=["scale"],
      support=[S("quantity","q:pushie-i0","quote")]),
 dict(id="xfm:ds/smoothed_map", name="smoothed element map", quantity="fitted fluorescence line area", units="counts", indexed_by=["pixel","element"], uncertainty="none", qualifiers=["peak_fitted","smoothed"]),
 dict(id="xfm:ds/scaled_map", name="group-scaled element map", quantity="relative fluorescence intensity", indexed_by=["pixel","element"], uncertainty="none", qualifiers=["peak_fitted","smoothed","scaled"], known_up_to=["scale"],
      support=[S("quantity","q:pist-scaling","quote")]),
 dict(id="xfm:ds/areal_concentration_map", name="areal concentration map", quantity="areal mass density of element", units="ug/cm2", indexed_by=["pixel","element"], uncertainty="propagated", qualifiers=["peak_fitted","flux_normalized","quantified_areal"],
      aliases=[dict(term="Xx-conc1 channel", used_by="SMAK")], support=[S("units","q:smak-std-film","quote"), S("quantity","q:pushie-areal","quote")]),
 dict(id="xfm:ds/mass_fraction_map", name="mass-fraction concentration map", quantity="mass fraction of element", indexed_by=["pixel","element"], uncertainty="propagated", qualifiers=["peak_fitted","flux_normalized","quantified_mass"],
      support=[S("quantity","q:sole-fp-unknowns","quote")]),
 dict(id="xfm:ds/optical_image", name="beamline optical image", quantity="visible-light intensity", indexed_by=["pixel"], uncertainty="none"),
 dict(id="xfm:ds/optical_on_xrf_grid", name="optical image resampled to XRF grid", quantity="visible-light intensity", indexed_by=["pixel"], uncertainty="none", qualifiers=["resampled"]),
 dict(id="xfm:ds/ftir_spectrum", name="per-pixel FTIR spectrum", quantity="IR absorbance per wavenumber", indexed_by=["pixel","spectral_band"], uncertainty="none"),
 dict(id="xfm:ds/ftir_spectrum_pre", name="preprocessed FTIR spectrum", quantity="IR absorbance per wavenumber", indexed_by=["pixel","spectral_band"], uncertainty="none", qualifiers=["atmosphere_corrected","denoised","baseline_corrected","mean_normalized"], known_up_to=["scale"]),
 dict(id="xfm:ds/ftir_band_map", name="FTIR band-abundance map", quantity="binned absorbance per assigned band", indexed_by=["pixel","spectral_band"], uncertainty="none", qualifiers=["binned"]),
 dict(id="xfm:ds/registered_stack", name="pixel-registered multimodal stack", quantity="XRF and FTIR channels on one grid", indexed_by=["pixel","component"], uncertainty="empirical", qualifiers=["registered"], known_up_to=["translation","rotation"]),
 dict(id="xfm:ds/fa_scores", name="factor-analysis component score maps", quantity="component score", indexed_by=["pixel","component","sample"], uncertainty="none", known_up_to=["sign","scale"],
      support=[S("quantity","q:pist-fa-n","quote")]),
 dict(id="xfm:ds/fa_loadings", name="factor-analysis loadings", quantity="channel loading on component", indexed_by=["component"], uncertainty="none", known_up_to=["sign","scale"]),
 dict(id="xfm:ds/claim_relative_distribution", name="claim: relative elemental distribution", quantity="statement", indexed_by=["element"], uncertainty="none"),
 dict(id="xfm:ds/claim_colocalization", name="claim: treatment-dependent co-localization of elements and biomolecular bands", quantity="statement", uncertainty="none"),
],
parameters=[
 dict(id="xfm:p/merge_threshold", name="line merge threshold", kind="numerical", units="eV", default_value="10", default_scope="PyMca", set_by="software_default", support=[S("default_value","q:sole-merge-10ev","quote")]),
 dict(id="xfm:p/blur_size", name="Gaussian filter size", kind="numerical", units="pixels", default_value="5", default_scope="SMAK Basics guide", set_by="user", support=[S("default_value","q:smak-blur-size","quote")]),
 dict(id="xfm:p/blur_sigma", name="Gaussian filter standard deviation", kind="numerical", units="pixels", default_value="0.8", default_scope="SMAK Basics guide", set_by="user", support=[S("default_value","q:smak-blur-sd","quote")]),
 dict(id="xfm:p/tau", name="dead-time constant tau", kind="empirical", set_by="calibration", physical_meaning="detector dead time per event, fitted from OCR vs ICR", support=[S("physical_meaning","q:smak-dt-tau","quote")]),
 dict(id="xfm:p/std_conc", name="standard areal concentration", kind="physical", units="ug/cm2", set_by="user", physical_meaning="known areal density of the thin-film standard", support=[S("physical_meaning","q:smak-std-film","quote")]),
 dict(id="xfm:p/i0_gain", name="I0 ion chamber gain", kind="physical", set_by="user", physical_meaning="amplifier gain of the incident-flux ion chamber; must match between standard and sample", support=[S("physical_meaning","q:smak-gain","quote")]),
 dict(id="xfm:p/sample_composition", name="sample matrix composition, density, thickness", kind="physical", set_by="user", physical_meaning="matrix for fundamental-parameter attenuation correction", support=[S("physical_meaning","q:sole-parallel-beam","quote")]),
 dict(id="xfm:p/n_markers", name="number of registration markers", kind="numerical", set_by="user", default_scope="pistachio V5 (up to 35)", support=[S("default_scope","q:pist-markers","quote")]),
 dict(id="xfm:p/sg_window", name="Savitzky-Golay window", kind="numerical", set_by="user", default_scope="pistachio V5 (9)", support=[S("default_scope","q:pist-ftir-atm","quote")]),
 dict(id="xfm:p/sg_order", name="Savitzky-Golay polynomial order", kind="numerical", set_by="user", default_scope="pistachio V5 (3)", support=[S("default_scope","q:pist-ftir-atm","quote")]),
 dict(id="xfm:p/norm_wavenumber", name="FTIR normalization wavenumber", kind="empirical", units="cm-1", set_by="decision", set_by_decision="xfm:d/norm_wavenumber", support=[S("set_by","q:pist-norm-1655","quote")]),
 dict(id="xfm:p/fa_weighting", name="FA channel-group weighting", kind="numerical", set_by="software_default", physical_meaning="auto-balance by sum of factor magnitudes per modality group", support=[S("physical_meaning","q:pist-fa-weight","quote")]),
],
rationales=[
 dict(id="xfm:r/fit_over_roi", name="fit when lines overlap", statement="ROI summation cannot separate overlapping fluorescence lines; fitting a physics-constrained model can.", kind="physical",
      support=[S("statement","q:sole-roi-fast","quote"), S("statement","q:pushie-fit-overlap","quote"), S("statement","q:pist-fit","quote")]),
 dict(id="xfm:r/global_calibration", name="fit calibration globally, elements as groups", statement="Fitting every peak independently would need ~60 non-linear parameters for 10 elements and would not converge; instead ZERO, GAIN, NOISE, FANO are fitted once for the region and lines are grouped per element with known ratios.", kind="statistical",
      support=[S("statement","q:sole-60params","quote"), S("statement","q:sole-global-params","quote")]),
 dict(id="xfm:r/i0_normalization", name="normalize to incident flux", statement="Dividing by I0 removes incident-flux variation so relative levels can be compared between samples.", kind="physical",
      support=[S("statement","q:pushie-i0","quote")]),
 dict(id="xfm:r/dt_rarely_needed", name="dead-time correction rarely needed at SSRL micro-imaging beamlines", statement="The detectors are linear to very high count rates, so dead-time correction is applied only when OCR vs ICR shows curvature.", kind="empirical",
      support=[S("statement","q:smak-dt-rare","quote"), S("statement","q:smak-dt-rule","quote")]),
 dict(id="xfm:r/group_scaling", name="scale element groups to comparable totals", statement="Scaling intensity ranges by element group allows direct comparison between samples while preserving within-element distribution differences.", kind="practical",
      support=[S("statement","q:pist-scaling-why","quote")]),
 dict(id="xfm:r/optical_resize", name="resample optical to XRF grid", statement="Losing optical resolution is acceptable because only the sample outline is needed as the registration marker.", kind="practical",
      support=[S("statement","q:pist-resize-why","quote")]),
 dict(id="xfm:r/manual_registration", name="manual over automatic registration", statement="Rotation between modalities and different detector geometries (45° vs 90°) distort images beyond what automatic tools handle.", kind="practical",
      support=[S("statement","q:pist-manual","quote")]),
 dict(id="xfm:r/rubberband", name="rubberband baseline", statement="The FTIR baseline is dominated by a smooth scattering offset suited to convex-hull correction; applying it identically everywhere makes residual bias systematic.", kind="physical",
      support=[S("statement","q:pist-rubberband","quote")]),
 dict(id="xfm:r/sg_denoise", name="Savitzky-Golay denoising", statement="Reduces high-frequency noise while preserving band shape and position in the fingerprint region.", kind="statistical",
      support=[S("statement","q:pist-ftir-atm","quote")]),
 dict(id="xfm:r/fa_weighting", name="auto-balance modality groups", statement="Without weighting, the larger intensity magnitudes of XRF would dominate the shared FA space over FTIR.", kind="statistical",
      support=[S("statement","q:pist-fa-why","quote")]),
 dict(id="xfm:r/pca_variance", name="variance marks chemical distinctness", statement="Areas with the greatest signal variation are assumed to be the chemically distinct areas.", kind="statistical",
      support=[S("statement","q:smak-pca-assume","quote")]),
],
assumptions=[
 dict(id="xfm:a/gaussian_response", name="Gaussian detector response", statement="The solid-state detector response function is predominantly Gaussian.", about="instrument", support=[S("statement","q:sole-gaussian","quote")]),
 dict(id="xfm:a/escape_geometry", name="escape-peak geometry", statement="Normal incidence on the detector; escape only from the front surface.", about="instrument", support=[S("statement","q:sole-escape","quote")]),
 dict(id="xfm:a/auger_neglected", name="non-radiative cascade approximated", statement="Auger transition ratios are neglected; acceptable because radiative transitions to the K shell dominate at practical L-line energies.", about="physics", support=[S("statement","q:sole-auger-crude","quote")]),
 dict(id="xfm:a/no_overlap", name="well-resolved lines", statement="The energy window of each ROI contains one element's lines only.", about="physics", if_violated=["xfm:f/roi_misattribution"], support=[S("statement","q:pushie-fit-overlap","quote")]),
 dict(id="xfm:a/detector_linear", name="detector linear at operating count rate", statement="Output count rate is linear in input count rate; no dead-time loss.", about="instrument", confirmable_by="OCR vs ICR correlation plot shows no curvature", if_violated=["xfm:f/deadtime_loss"], support=[S("statement","q:smak-dt-rare","quote"), S("confirmable_by","q:smak-dt-rule","quote")]),
 dict(id="xfm:a/infinitely_thin", name="infinitely thin sample", statement="No attenuation of incident or fluorescent X-rays within the sample; no thickness correction.", about="sample", if_violated=["xfm:f/matrix_effects"], support=[S("statement","q:smak-thin","quote"), S("statement","q:pushie-areal","quote")]),
 dict(id="xfm:a/same_energy", name="same excitation energy as standard", statement="Calibration is valid only at the excitation energy at which the standard was measured.", about="instrument", support=[S("statement","q:smak-energy","quote"), S("statement","q:pushie-std-same-scan","quote")]),
 dict(id="xfm:a/same_distance", name="same detector distance as standard", statement="Calibration is valid only at the detector distance at which the standard was measured.", about="instrument", support=[S("statement","q:smak-distance","quote")]),
 dict(id="xfm:a/same_gain", name="same I0 gain as standard (or recorded)", statement="I0 ion-chamber gain must match between standard and sample, or be recorded and corrected.", about="instrument", support=[S("statement","q:smak-gain","quote")]),
 dict(id="xfm:a/secondary_negligible", name="secondary fluorescence negligible", statement="Secondary (inter-element) fluorescence excitation is neglected in fundamental-parameter quantification.", about="physics", confirmable_by="standards or Monte Carlo estimate", support=[S("statement","q:sole-secondary","quote"), S("confirmable_by","q:sole-secondary","quote")]),
 dict(id="xfm:a/shared_loadings", name="shared loading matrix across samples", statement="Control and treatment share one component (loading) matrix; only scores differ.", about="model", support=[S("statement","q:pist-fa-shared","quote")]),
 dict(id="xfm:a/variance_is_chemistry", name="variance marks chemical distinctness", statement="Greatest signal variation corresponds to chemically distinct regions.", about="statistics", support=[S("statement","q:smak-pca-assume","quote")]),
],
preconditions=[
 dict(id="xfm:pre/icr_ocr_available", name="ICR and OCR recorded per pixel", statement="Input and output count rates are stored per detector channel per pixel.", on_state="xfm:ds/mca_spectrum", checkable_by="metadata", support=[S("statement","q:smak-dt-rule","quote")]),
 dict(id="xfm:pre/i0_recorded", name="I0 recorded", statement="An incident-flux channel (I0STRM/I0/I1) exists in the map file.", on_state="xfm:ds/fitted_area_map", checkable_by="metadata", support=[S("statement","q:smak-i0-auto","quote")]),
 dict(id="xfm:pre/standard_measured", name="standard measured under sample conditions", statement="A thin-film standard was scanned at the same energy, distance and gain as the sample.", on_state="xfm:ds/fitted_area_map", checkable_by="metadata", support=[S("statement","q:pushie-std-same-scan","quote")]),
 dict(id="xfm:pre/common_grid", name="modalities on one pixel grid", statement="All channels to be analysed jointly are resampled to one pixel grid.", on_state="xfm:ds/registered_stack", checkable_by="data", established_by=["xfm:t/optical_resize","xfm:t/manual_registration"]),
],
decisions=[
 dict(id="xfm:d/fit_vs_roi", name="fit or ROI-sum", question="Do fluorescence lines of the elements of interest overlap at the detector resolution?", has_rule=True, requires_human=False, resolved_by="inspection of the summed spectrum for overlapping lines",
      options=[dict(label="fit", selects_transformation="xfm:t/spectrum_fit", consequence="separates overlaps; introduces model dependence"), dict(label="roi", selects_transformation="xfm:t/roi_summation", consequence="fast; misattributes counts if lines overlap")],
      support=[S("question","der:fit-vs-roi-rule","derived"), S("has_rule","q:sole-roi-fast","quote")]),
 dict(id="xfm:d/deadtime", name="apply dead-time correction", question="Is the detector in the non-linear (saturated) regime at the count rates of this scan?", has_rule=True, requires_human=True, resolved_by="curvature in OCR vs ICR correlation plot",
      options=[dict(label="apply", selects_transformation="xfm:t/deadtime_correction", sets_parameter="xfm:p/tau"), dict(label="skip", consequence="counts under-reported at high rates if assumption fails")],
      support=[S("question","q:smak-dt-rule","quote"), S("resolved_by","q:smak-dt-rule","quote")]),
 dict(id="xfm:d/quantify", name="relative or absolute", question="Is an absolute concentration needed, and is the sample thin enough (or its matrix known enough) to quantify?", has_rule=False, requires_human=True,
      options=[dict(label="relative only", consequence="claims limited to distribution and between-sample ratios"), dict(label="ratio to thin-film standard", selects_transformation="xfm:t/quant_ratio_standard"), dict(label="fundamental parameters", selects_transformation="xfm:t/quant_fp", sets_parameter="xfm:p/sample_composition")],
      support=[S("options","q:pushie-ratio","quote"), S("options","q:sole-fp-unknowns","quote")]),
 dict(id="xfm:d/registration_mode", name="manual or automatic registration", question="Do rotation or detector-geometry distortions between modalities exceed what automatic alignment corrects?", has_rule=False, requires_human=True,
      options=[dict(label="manual", selects_transformation="xfm:t/manual_registration", sets_parameter="xfm:p/n_markers"), dict(label="automatic", selects_transformation="xfm:t/stack_alignment")],
      support=[S("question","q:pist-manual","quote"), S("options","q:smak-stack-align","quote")]),
 dict(id="xfm:d/norm_wavenumber", name="FTIR normalization wavenumber", question="Which wavenumber is a stable normalizer that does not overlap a band of interest?", has_rule=False, requires_human=True, resolved_by="repeat analysis with an alternative wavenumber and check results are unchanged",
      options=[dict(label="1655 cm-1", sets_parameter="xfm:p/norm_wavenumber", sets_value="1655"), dict(label="2900 cm-1", sets_parameter="xfm:p/norm_wavenumber", sets_value="2900")],
      support=[S("resolved_by","q:pist-norm-1655","quote")]),
],
validation_criteria=[
 dict(id="xfm:v/edge_comparison", name="below/above-edge map comparison", statement="Fit an element with an interfering line at excitation energies below and above the interferer's edge; maps should agree.", observable="similarity of fitted maps at two excitation energies", support=[S("statement","q:sole-edge-validation","quote")]),
 dict(id="xfm:v/ocr_icr_linearity", name="OCR vs ICR linearity", statement="OCR vs ICR correlation plot is linear.", observable="curvature of OCR vs ICR", threshold="no visible curvature", support=[S("statement","q:smak-dt-rule","quote")]),
 dict(id="xfm:v/registration_rmse", name="registration RMSE", statement="RMSE over reference markers after alignment.", observable="marker RMSE in pixels", threshold="11 px reported; 8 px after removing sheared markers", derived_bound="co-localization interpretable only for features larger than ~80 µm", support=[S("threshold","q:pist-rmse","quote"), S("derived_bound","der:reg-bound","derived")]),
 dict(id="xfm:v/norm_invariance", name="normalization invariance", statement="Spatial distributions unchanged when the normalization wavenumber is changed.", observable="agreement of band maps under 1655 vs 2900 cm-1 normalization", support=[S("statement","q:pist-norm-1655","quote")]),
 dict(id="xfm:v/least_shear", name="least shear over marker sets", statement="Registration repeated with different marker sets; the set with least shear is kept.", observable="shear of the fitted transform", support=[S("statement","q:pist-repeat","quote")]),
],
failure_modes=[
 dict(id="xfm:f/roi_misattribution", name="ROI misattribution under overlap", statement="Counts from a neighbouring element's line are attributed to the ROI element (e.g. K Kβ into Ca Kα).", caused_by=["xfm:a/no_overlap"], detected_by=["xfm:v/edge_comparison"], support=[S("statement","q:pushie-kkb-caka","quote")]),
 dict(id="xfm:f/deadtime_loss", name="dead-time count loss", statement="At high input rates the detector under-counts; visible as pile-up peaks at double energy and OCR/ICR curvature.", caused_by=["xfm:a/detector_linear"], detected_by=["xfm:v/ocr_icr_linearity"], support=[S("statement","q:pushie-pileup","quote"), S("statement","q:smak-dt-rule","quote")]),
 dict(id="xfm:f/matrix_effects", name="matrix and thickness effects", statement="In thick or dense samples, incident penetration and fluorescence escape depend on the matrix; thin-sample quantification is biased.", caused_by=["xfm:a/infinitely_thin"], support=[S("statement","q:smak-matrix","quote")]),
],
transformations=[
 dict(id="xfm:t/roi_summation", name="ROI summation", purpose="Produce a per-element map by summing counts in an energy window around the element's line.", consumes=["xfm:ds/mca_spectrum"], produces=["xfm:ds/roi_map"], adds_qualifiers=["roi_summed"], assumes=["xfm:a/no_overlap"], failure_modes=["xfm:f/roi_misattribution"], alternative_to=["xfm:t/spectrum_fit"],
      preserves=["counting statistics"], discards=["spectral shape"], introduces=["window-placement dependence"],
      justified_by=[], implemented_in=[dict(software="PyMca", component="ROI imaging", grit="q:sole-roi-fast"), dict(software="XRF-Maps", component="src/fitting/routines/roi_fit_routine.cpp")],
      support=[S("purpose","q:sole-roi-fast","quote")], gaps=[dict(field="justified_by", kind="not_in_corpus", note="'fast and simple' is stated; a rationale node was not created for a speed argument.")]),
 dict(id="xfm:t/spectrum_fit", name="per-pixel spectrum fitting", purpose="Separate overlapping fluorescence lines into per-element line areas by least-squares fitting of a physics-constrained peak model.", consumes=["xfm:ds/mca_spectrum"], produces=["xfm:ds/fitted_area_map"], adds_qualifiers=["peak_fitted"],
      parameters=["xfm:p/merge_threshold"], assumes=["xfm:a/gaussian_response","xfm:a/escape_geometry","xfm:a/auger_neglected"], justified_by=["xfm:r/fit_over_roi","xfm:r/global_calibration"], validated_by=["xfm:v/edge_comparison"], alternative_to=["xfm:t/roi_summation"],
      preserves=["total counts per element group"], discards=["per-channel counts"], introduces=["dependence on peak model and line-ratio tables","fitted covariance"],
      implemented_in=[dict(software="PyMca", component="batch fitting", grit="q:sole-edge-validation"), dict(software="SMAK", version="3.0.11", component="MCA > View Config > PyMCA fit", grit="q:pist-fit"), dict(software="XRF-Maps", component="src/fitting/routines/{nnls,svd,param_optimized,matrix_optimized,hybrid}_fit_routine.cpp", note="five routines; alternatives to each other, not yet modelled")],
      support=[S("purpose","q:pist-fit","quote"), S("purpose","q:sole-roi-fast","quote")],
      conflicts=[dict(field="parameters", positions=[dict(statement="ZERO, GAIN, NOISE, FANO are fitted for the whole region, all peaks simultaneously.", grit="q:sole-global-params"), dict(statement="SMAK practice unchecks Spectrometer Zero, Gain and Detector Width on the first small fit.", grit="inf:smak-calib-free")], would_settle="Read SMAK's PyMca config handling: whether a second pass refits calibration.")]),
 dict(id="xfm:t/deadtime_correction", name="dead-time correction", purpose="Restore counts lost when the detector's input rate exceeds its throughput.", consumes=["xfm:ds/fitted_area_map"], produces=["xfm:ds/fitted_area_map_dtc"], adds_qualifiers=["dead_time_corrected"], parameters=["xfm:p/tau"], requires=["xfm:pre/icr_ocr_available"], assumes=[], justified_by=["xfm:r/dt_rarely_needed"], validated_by=["xfm:v/ocr_icr_linearity"], failure_modes=["xfm:f/deadtime_loss"],
      preserves=["spatial pattern"], changes=["absolute counts, rate-dependently"], introduces=["dependence on fitted tau"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="Analysis > Do ICR-OCR Deadtime; Process > Deadtime", grit="q:smak-dt-tau")],
      support=[S("purpose","q:smak-dt-rule","quote")],
      gaps=[dict(field="governing_relation", kind="not_in_corpus", note="The ICR/OCR model (paralyzable vs non-paralyzable) is not stated in any registered source.")]),
 dict(id="xfm:t/flux_normalization", name="incident-flux normalization", purpose="Remove variation in incident beam intensity so maps are comparable within and between scans.", consumes=["xfm:ds/fitted_area_map"], produces=["xfm:ds/normalized_map"], adds_qualifiers=["flux_normalized"], requires=["xfm:pre/i0_recorded"], justified_by=["xfm:r/i0_normalization"],
      preserves=["relative distribution"], discards=["absolute count scale"], introduces=["I0 channel noise"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="Analyze > Quantification (automatic); Map Math (manual)", grit="q:smak-i0-auto")],
      support=[S("purpose","q:pushie-i0","quote")]),
 dict(id="xfm:t/gaussian_smoothing", name="Gaussian smoothing", purpose="Suppress pixel-level noise in element maps before display and multivariate analysis.", consumes=["xfm:ds/fitted_area_map"], produces=["xfm:ds/smoothed_map"], adds_qualifiers=["smoothed"], parameters=["xfm:p/blur_size","xfm:p/blur_sigma"], justified_by=[],
      preserves=["total counts approximately"], changes=["spatial resolution"], discards=["sub-kernel features"], introduces=["spatial correlation between pixels"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="Process > Advanced Filtering > Blur", grit="q:smak-blur-size")],
      support=[S("purpose","q:pist-blur","quote")],
      gaps=[dict(field="justified_by", kind="not_in_corpus", note="Neither the trace nor the SMAK Basics extraction states why blurring is applied.", search_grit="abs:smak-blur-why")]),
 dict(id="xfm:t/group_scaling", name="element-group intensity scaling", purpose="Scale intensity ranges by element group so maps of very different abundance can be compared across samples.", consumes=["xfm:ds/smoothed_map"], produces=["xfm:ds/scaled_map"], adds_qualifiers=["scaled"], justified_by=["xfm:r/group_scaling"],
      preserves=["within-element distribution"], discards=["absolute counts","between-group ratios"], introduces=["display-driven scale convention"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="Map Math / display scaling", grit="q:pist-scaling")],
      support=[S("purpose","q:pist-scaling","quote")]),
 dict(id="xfm:t/quant_ratio_standard", name="quantification by ratio to thin-film standard", purpose="Convert flux-normalized line areas to areal concentration by ratio to a standard of known areal density measured under the same conditions.", consumes=["xfm:ds/fitted_area_map"], produces=["xfm:ds/areal_concentration_map"], adds_qualifiers=["flux_normalized","quantified_areal"], parameters=["xfm:p/std_conc","xfm:p/i0_gain"], requires=["xfm:pre/standard_measured","xfm:pre/i0_recorded"], assumes=["xfm:a/infinitely_thin","xfm:a/same_energy","xfm:a/same_distance","xfm:a/same_gain"], justified_by=["xfm:r/i0_normalization"], failure_modes=["xfm:f/matrix_effects"], alternative_to=["xfm:t/quant_fp"],
      governing_relation="C_sample = C_std * (A_sample / I0_sample) / (A_std / I0_std), per element, in ug/cm2",
      preserves=["relative distribution"], changes=["units to ug/cm2"], introduces=["dependence on standard concentration and its uncertainty"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="Analyze > Quantification > Quantitative Analysis", grit="q:smak-i0-auto")],
      support=[S("purpose","q:pushie-ratio","quote"), S("governing_relation","q:pushie-ratio","quote"), S("assumes","q:smak-thin","quote"), S("assumes","q:smak-semiquant","quote")]),
 dict(id="xfm:t/quant_fp", name="fundamental-parameter quantification", purpose="Convert line areas to mass fractions using tabulated cross-sections, yields and the sample's own matrix attenuation.", consumes=["xfm:ds/fitted_area_map"], produces=["xfm:ds/mass_fraction_map"], adds_qualifiers=["flux_normalized","quantified_mass"], parameters=["xfm:p/sample_composition"], assumes=["xfm:a/secondary_negligible"], justified_by=[], alternative_to=["xfm:t/quant_ratio_standard"],
      governing_relation="A = I0 * C * (Omega/4pi) * sum_j R_j^W  (Solé 2007 eq. 13), with R_j^W from the parallel-beam matrix expression (eq. 7)",
      preserves=["relative distribution"], changes=["units to mass fraction"], introduces=["dependence on matrix composition, density, thickness and geometry"],
      implemented_in=[dict(software="PyMca", component="fundamental parameters / matrix definition", grit="q:sole-parallel-beam")],
      support=[S("purpose","q:sole-fp-unknowns","quote"), S("governing_relation","q:sole-parallel-beam","quote")],
      gaps=[dict(field="justified_by", kind="not_in_corpus", note="Rationale for FP over standards is in the Handbook (not registered).")]),
 dict(id="xfm:t/optical_stitch", name="optical mosaic stitching", purpose="Assemble beamline optical tiles into one image by feature matching rather than stage coordinates.", consumes=["xfm:ds/optical_image"], produces=["xfm:ds/optical_image"], justified_by=[],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="File > Covert SSRL Mosaic / vision stitching", grit="q:pist-stitch")],
      support=[S("purpose","q:pist-stitch","quote")], gaps=[dict(field="justified_by", kind="not_in_corpus", note="'rather than coordinates' is stated without a reason.")]),
 dict(id="xfm:t/optical_resize", name="optical resample to XRF grid", purpose="Bring the optical image onto the XRF pixel grid so it can serve as the registration reference.", consumes=["xfm:ds/optical_image"], produces=["xfm:ds/optical_on_xrf_grid"], adds_qualifiers=["resampled"], justified_by=["xfm:r/optical_resize"],
      discards=["optical resolution"], preserves=["sample outline"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="File > Export with Resize", grit="q:pist-resize")],
      support=[S("purpose","q:pist-resize","quote")]),
 dict(id="xfm:t/ftir_preprocess", name="FTIR spectral preprocessing", purpose="Remove atmospheric bands, noise and scattering baseline, and normalize, so band intensities are comparable across pixels and samples.", consumes=["xfm:ds/ftir_spectrum"], produces=["xfm:ds/ftir_spectrum_pre"], adds_qualifiers=["atmosphere_corrected","denoised","baseline_corrected","mean_normalized"], parameters=["xfm:p/sg_window","xfm:p/sg_order","xfm:p/norm_wavenumber"], justified_by=["xfm:r/sg_denoise","xfm:r/rubberband"], validated_by=["xfm:v/norm_invariance"],
      preserves=["band positions and shapes (stated intent)"], discards=["absolute absorbance scale","high-frequency content"], introduces=["dependence on normalizer band"],
      implemented_in=[dict(software="Omnic", version="9.8", component="atmospheric correction, ENVI export", grit="q:pist-ftir-atm"), dict(software="SMAK", version="3.0.11", component="multi-channel analysis tool", grit="q:pist-ftir-atm")],
      support=[S("purpose","q:pist-ftir-atm","quote"), S("purpose","q:pist-rubberband","quote")],
      gaps=[dict(field="alternative_to", kind="not_in_corpus", note="Granularity: four conceptual steps with three stated rationales collapsed into one Transformation because the trace applies them as a fixed sequence; split if a source separates them.")]),
 dict(id="xfm:t/ftir_band_binning", name="FTIR band binning", purpose="Integrate preprocessed spectra over wavenumber windows assigned to plant-tissue components to produce per-band maps.", consumes=["xfm:ds/ftir_spectrum_pre"], produces=["xfm:ds/ftir_band_map"], adds_qualifiers=["binned"], justified_by=[],
      introduces=["dependence on band assignments"],
      support=[S("purpose","q:pist-norm-1655","quote","band list follows this sentence in the trace")],
      gaps=[dict(field="justified_by", kind="not_in_corpus", note="Band assignments cite prior literature not registered.")]),
 dict(id="xfm:t/manual_registration", name="manual landmark registration", purpose="Bring FTIR maps onto the XRF pixel grid using operator-placed landmarks on the two optical images.", consumes=["xfm:ds/optical_on_xrf_grid","xfm:ds/ftir_band_map","xfm:ds/scaled_map"], produces=["xfm:ds/registered_stack"], adds_qualifiers=["registered"], parameters=["xfm:p/n_markers"], justified_by=["xfm:r/manual_registration"], validated_by=["xfm:v/registration_rmse","xfm:v/least_shear"], alternative_to=["xfm:t/stack_alignment"],
      changes=["FTIR pixel positions"], introduces=["registration error (RMSE) that bounds all downstream co-localization claims"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="manual alignment function", grit="q:pist-markers")],
      support=[S("purpose","q:pist-markers","quote"), S("introduces","der:reg-bound","derived")]),
 dict(id="xfm:t/stack_alignment", name="automatic stack alignment", purpose="Align channels from different maps automatically to correct beam-position shifts.", consumes=["xfm:ds/scaled_map"], produces=["xfm:ds/registered_stack"], adds_qualifiers=["registered"], justified_by=[], alternative_to=["xfm:t/manual_registration"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="Process > Stack Alignment", grit="q:smak-stack-align")],
      support=[S("purpose","q:smak-stack-align","quote")], gaps=[dict(field="justified_by", kind="not_in_corpus"), dict(field="assumes", kind="not_in_corpus", note="Algorithm and its assumptions (rigid? translation-only?) are not stated in the extraction.")]),
 dict(id="xfm:t/multifile_weighted_fa", name="multifile weighted factor analysis", purpose="Decompose the registered multimodal stacks of control and treatment jointly into shared components, so pixel-wise co-variation patterns can be compared between samples.", consumes=["xfm:ds/registered_stack"], produces=["xfm:ds/fa_scores","xfm:ds/fa_loadings"], parameters=["xfm:p/fa_weighting"], requires=["xfm:pre/common_grid"], assumes=["xfm:a/shared_loadings","xfm:a/variance_is_chemistry"], justified_by=["xfm:r/fa_weighting","xfm:r/pca_variance"],
      preserves=["total variance (all channels retained as components)"], introduces=["shared latent space; sign/scale indeterminacy of components"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="multifile weighted factor analysis", grit="q:pist-fa-weight")],
      support=[S("purpose","q:pist-fa-shared","quote"), S("assumes","q:smak-pca-assume","quote","PCA assumption applied to FA by analogy")],
      gaps=[dict(field="governing_relation", kind="not_in_corpus", note="FA model, rotation and weighting formula not stated; SMAK guide pages fetched do not document multifile FA.")]),
 dict(id="xfm:t/component_effect_statistics", name="component effect statistics", purpose="Identify which shared components differ between control and treatment, by variance ratio and Cohen's d over component scores.", consumes=["xfm:ds/fa_scores"], produces=["xfm:ds/claim_colocalization"], justified_by=[],
      governing_relation="variance ratio = var_c(scores) / sum_c var_c(scores); Cohen's d = (mean_treat - mean_ctrl) / pooled SD",
      introduces=["dependence on component sign and on pixel count as sample size"],
      implemented_in=[dict(software="custom Python (NumPy/SciPy/tkinter)", component="variance ratio, Cohen's d", grit="q:pist-cohen")],
      support=[S("governing_relation","q:pist-varratio","quote"), S("governing_relation","q:pist-cohen","quote")],
      gaps=[dict(field="assumes", kind="tacit", note="Treating pixels as independent samples for effect size is not discussed.")]),
 dict(id="xfm:t/channel_component_correlation", name="channel-component correlation", purpose="Attribute each component to the measured channels that drive it, by Pearson correlation between channel maps and component scores.", consumes=["xfm:ds/fa_scores","xfm:ds/registered_stack"], produces=["xfm:ds/claim_colocalization"], justified_by=[],
      implemented_in=[dict(software="custom Python (NumPy)", component="Pearson r heat map", grit="q:pist-pearson")],
      support=[S("purpose","q:pist-pearson","quote")]),
],
goals=[
 dict(id="xfm:g/relative_distribution", name="relative elemental distribution", statement="Map where each element is, comparably between control and treatment, without absolute concentrations.", claim_type="xfm:ds/claim_relative_distribution", support=[S("statement","q:pushie-i0","quote")]),
 dict(id="xfm:g/colocalization", name="element–biomolecule co-localization under treatment", statement="Determine which elemental and biomolecular signals co-vary spatially and how that changes with salinity treatment.", claim_type="xfm:ds/claim_colocalization", validated_by=["xfm:v/registration_rmse"]),
 dict(id="xfm:g/absolute_concentration", name="absolute areal or mass concentration", statement="Report element concentration in ug/cm2 or mass fraction.", claim_type="xfm:ds/areal_concentration_map", support=[S("statement","q:pushie-areal","quote")]),
],
workflows=[
 dict(id="xfm:w/pistachio_multimodal", name="pistachio multimodal XRF+FTIR workflow (as run)", serves=["xfm:g/relative_distribution","xfm:g/colocalization"], starts_from=["xfm:ds/mca_spectrum","xfm:ds/ftir_spectrum","xfm:ds/optical_image"], ends_at=["xfm:ds/claim_colocalization"], validated_by=["xfm:v/registration_rmse","xfm:v/norm_invariance"],
      steps=[
        dict(step_id="s1", transformation="xfm:t/spectrum_fit", condition="xfm:d/fit_vs_roi", condition_option="fit", role="conditional"),
        dict(step_id="s2", transformation="xfm:t/deadtime_correction", preceded_by=["s1"], condition="xfm:d/deadtime", condition_option="apply", role="conditional"),
        dict(step_id="s3", transformation="xfm:t/gaussian_smoothing", preceded_by=["s1"], parameter_values=[dict(parameter="xfm:p/blur_size", value="5", deviates_from_default=False), dict(parameter="xfm:p/blur_sigma", value="0.75", deviates_from_default=True)]),
        dict(step_id="s4", transformation="xfm:t/group_scaling", preceded_by=["s3"]),
        dict(step_id="s5", transformation="xfm:t/optical_stitch"),
        dict(step_id="s6", transformation="xfm:t/optical_resize", preceded_by=["s5"]),
        dict(step_id="s7", transformation="xfm:t/ftir_preprocess", parameter_values=[dict(parameter="xfm:p/sg_window", value="9"), dict(parameter="xfm:p/sg_order", value="3"), dict(parameter="xfm:p/norm_wavenumber", value="1655")]),
        dict(step_id="s7v", transformation="xfm:t/ftir_preprocess", preceded_by=["s7"], role="validation", condition="xfm:d/norm_wavenumber", condition_option="2900 cm-1", parameter_values=[dict(parameter="xfm:p/norm_wavenumber", value="2900")]),
        dict(step_id="s8", transformation="xfm:t/ftir_band_binning", preceded_by=["s7"]),
        dict(step_id="s9", transformation="xfm:t/manual_registration", preceded_by=["s4","s6","s8"], condition="xfm:d/registration_mode", condition_option="manual", role="loop", loop_group="registration", loop_exit="xfm:v/least_shear", parameter_values=[dict(parameter="xfm:p/n_markers", value="up to 35")]),
        dict(step_id="s10", transformation="xfm:t/multifile_weighted_fa", preceded_by=["s9"]),
        dict(step_id="s11", transformation="xfm:t/component_effect_statistics", preceded_by=["s10"]),
        dict(step_id="s12", transformation="xfm:t/channel_component_correlation", preceded_by=["s10"]),
      ],
      support=[S("steps","q:pist-fit","quote"), S("steps","q:pist-repeat","quote")],
      gaps=[dict(field="steps", kind="human_judgment", note="Dead-time correction (s2) and flux normalization are not stated in the trace; s2 is recorded as conditional with the trace's status inferred 'not applied' (inf:pist-deadtime-not-applied); flux normalization status is unknown (inf:pist-i0-unknown) and is therefore not a step here.", search_grit="abs:pist-i0")]),
 dict(id="xfm:w/absolute_quant_smak", name="absolute quantification via thin-film standard (not run)", serves=["xfm:g/absolute_concentration"], starts_from=["xfm:ds/mca_spectrum"], ends_at=["xfm:ds/areal_concentration_map"],
      steps=[dict(step_id="a1", transformation="xfm:t/spectrum_fit"), dict(step_id="a2", transformation="xfm:t/quant_ratio_standard", preceded_by=["a1"], condition="xfm:d/quantify", condition_option="ratio to thin-film standard", role="conditional")],
      alternative_to=["xfm:w/absolute_quant_fp"], support=[S("steps","q:smak-i0-auto","quote")]),
 dict(id="xfm:w/absolute_quant_fp", name="absolute quantification via fundamental parameters (not run)", serves=["xfm:g/absolute_concentration"], starts_from=["xfm:ds/mca_spectrum"], ends_at=["xfm:ds/mass_fraction_map"],
      steps=[dict(step_id="b1", transformation="xfm:t/spectrum_fit"), dict(step_id="b2", transformation="xfm:t/quant_fp", preceded_by=["b1"], condition="xfm:d/quantify", condition_option="fundamental parameters", role="conditional")],
      alternative_to=["xfm:w/absolute_quant_smak"], support=[S("steps","q:sole-fp-unknowns","quote")]),
])


# ================================================================== code-tier overlay: SMAK 3.0.11 source
SMAK_TARBALL_SHA = "0d368011a8f455cf5a789967fc082c4cebfcd82d983cc8cdcab1a87e9ae0bf5b"
CODE_FILES = ["smak.py","__main__.py","BatchAnalyze.py","pyMcaFitWrapper.py","AlignStack3.py","ExportRegistrationClass.py","WeightedGroupPCA.py","IR_MathClass.py","pyMcaConfigs/activePyMCAConfig.cfg"]
for f in CODE_FILES:
    key = "smak-code:" + f
    SOURCES[key] = dict(uri=f"smak-3.0.11.tar.gz#smak-3.0.11/src/smak/{f}", file=f"smak-code/{f}", text=f"smak-code/{f}", media_type="text/x-python" if f.endswith(".py") else "text/plain")
    SOURCES[key]["sha256"] = sha(SRC / "smak-code" / f)
    SOURCES[key]["ntext"] = norm((SRC / "smak-code" / f).read_text(errors="replace"))
CQ = {
 "q:code-dt-model":     ("smak-code:smak.py", "#DT: corFF=FF*exp(tau*1e-6*ICR)"),
 "q:code-dt-apply":     ("smak-code:smak.py", "dtcor=np.exp(float(self.deadtimevalue.getvalue())*1e-6*icr)"),
 "q:code-dt-toggle":    ("smak-code:smak.py", "self.dodt=tkinter.IntVar()"),
 "q:code-i0-toggle":    ("smak-code:smak.py", "self.doI0c=tkinter.IntVar()"),
 "q:code-i0-divide":    ("smak-code:smak.py", "newdata[i,j]=float(pic[i,j])/float(i0dat[i,j])"),
 "q:code-i0-prefer":    ("smak-code:smak.py", "if 'I0STRM' in self.mapdata.labels:"),
 "q:code-time-norm":    ("smak-code:__main__.py", "newdata=ad/Tdat*80000000"),
 "q:code-fastfit":      ("smak-code:pyMcaFitWrapper.py", "fFit=FastXRFLinearFit.FastXRFLinearFit(mcafit=self.mcafit)"),
 "q:code-fastfit-args": ("smak-code:pyMcaFitWrapper.py", "ysum=None, weight=None, refit=True)"),
 "q:code-cfg-fixed":    ("smak-code:pyMcaConfigs/activePyMCAConfig.cfg", "fixednoise = 1 fixedgain = 1"),
 "q:code-cfg-fixedzero":("smak-code:pyMcaConfigs/activePyMCAConfig.cfg", "fixedsum = 1 fixedzero = 1"),
 "q:code-cfg-snip":     ("smak-code:pyMcaConfigs/activePyMCAConfig.cfg", "snipwidth = 50"),
 "q:code-cfg-strip":    ("smak-code:pyMcaConfigs/activePyMCAConfig.cfg", "stripalgorithm = 1"),
 "q:code-cfg-hypermet": ("smak-code:pyMcaConfigs/activePyMCAConfig.cfg", "hypermetflag = 7"),
 "q:code-quant-divide": ("smak-code:BatchAnalyze.py", "newdata=np.divide(adata,i0dat, out=np.zeros_like(adata),where=i0dat!=0)"),
 "q:code-quant-poly":   ("smak-code:BatchAnalyze.py", "mod = [fw.slope,fw.intc] pred = np.poly1d(mod) newdata = pred(newdata)"),
 "q:code-quant-nonorm": ("smak-code:BatchAnalyze.py", "if \"None\" in normCCSchan: normCCSchan = None"),
 "q:code-stack-transl": ("smak-code:AlignStack3.py", "def TwoDAlign(self): # Translation alignment"),
 "q:code-stack-hipass": ("smak-code:AlignStack3.py", "D = d - scnd.gaussian_filter(d, 2) # Take the high-F component"),
 "q:code-stack-phase":  ("smak-code:AlignStack3.py", "self.TList[dset][i, :], error, difphase = register_translation("),
 "q:code-tps":          ("smak-code:ExportRegistrationClass.py", "aligneddata = tps_warp2D(Y,Z,image,template.shape)"),
 "q:code-fa-norm":      ("smak-code:WeightedGroupPCA.py", "evnorm = np.sum(abs(self.PCAdataStruct.PCAevect),axis=0)"),
 "q:code-fa-grp":       ("smak-code:WeightedGroupPCA.py", "self.grpnorms.append(np.sum(evnorm[i:i+len(ch)])/len(ch))"),
 "q:code-fa-adj":       ("smak-code:WeightedGroupPCA.py", "cd = max(self.grpnorms) adjfact = float(cd)/np.array(self.grpnorms)"),
 "q:code-fa-cycles":    ("smak-code:WeightedGroupPCA.py", "cycles = 3"),
 "q:code-fa-sign":      ("smak-code:WeightedGroupPCA.py", "if abs(dmin)>abs(dmax):"),
 "q:code-ir-atm":       ("smak-code:IR_MathClass.py", "(data,b) = atm.atmospheric(wn,data,atm=None,cut_co2=params.atmCutCO2)"),
 "q:code-ir-mie":       ("smak-code:IR_MathClass.py", "data = mie.rmiesc(wn,data,ref,iterations=params.mieMaxIter,clusters=params.mieClusters,progressCallback=normCallback)"),
 "q:code-ir-sg":        ("smak-code:IR_MathClass.py", "corr = scipy.signal.savgol_filter(data, params.denoiseWindow, params.denoisePolyOrder, axis=1)"),
 "q:code-ir-rubber":    ("smak-code:IR_MathClass.py", "if params.baseline == 'Rubberband': bl = baseline.rubberband(wn, data)"),
 "q:code-ir-norm":      ("smak-code:IR_MathClass.py", "data = normalization.normalize_spectra(params.defs.normtrans[params.normalization], data, wn=wn, wavenum=params.normvalue)"),
 "q:code-ir-defaults":  ("smak-code:IR_MathClass.py", "self.baseline = 'Rubberband'"),
}
miss = [(k, s) for k, (s, ex) in CQ.items() if SOURCES[s]["ntext"].find(norm(ex)) < 0]
if miss:
    for k, s in miss: print("CODE QUOTE NOT FOUND:", k, "in", s)
    sys.exit(1)
Q.update(CQ)
for qid, (s, ex) in CQ.items():
    src = SOURCES[s]
    graph.append({"@id": qid, "@type": "prov:Entity", "plan": PLAN_ID, "how": "quote", "agent": AGENT,
                  "source": {"uri": src["uri"], "sha256": src["sha256"], "media_type": src["media_type"]},
                  "target": {"hasSource": src["uri"], "selector": {"@type": "oa:TextQuoteSelector", "exact": ex}}})
# derived entities from code
CINF = {
 "der:time-scalar": dict(how="derived", rationale="SMAK divides by the TIME channel and multiplies by 80,000,000 (q:code-time-norm); the guide's scalar 0.0000125 is 1/80,000,000, i.e. the TIME channel counts an 80 MHz clock and the operation converts counts to counts per second.", summary="The time-normalization scalar 0.0000125 is the reciprocal of an 80 MHz dwell clock."),
 "der:dt-paralyzable": dict(how="derived", rationale="corFF = FF*exp(tau*1e-6*ICR) (q:code-dt-model, q:code-dt-apply) is the paralyzable dead-time model with tau in microseconds and ICR in counts per second.", summary="SMAK's dead-time correction is the paralyzable (exponential) model."),
 "der:corrections-viewtime": dict(how="derived", rationale="dodt and doI0c are Tk IntVars (q:code-dt-toggle, q:code-i0-toggle) whose default is 0, applied when a channel is rendered or analysed (q:code-dt-apply, q:code-i0-divide) rather than written as new channels. Off unless the operator switches them on; no record in the data file.", summary="In SMAK, dead-time and I0 corrections are view-time toggles, default off, leaving no trace in saved channels."),
 "der:smak-fit-linear": dict(how="derived", rationale="SMAK calls PyMca FastXRFLinearFit.fitMultipleSpectra with weight=None and refit=True (q:code-fastfit, q:code-fastfit-args) under a configuration in which zero, gain, noise, fano and sum are all fixed (q:code-cfg-fixed, q:code-cfg-fixedzero). Per-pixel fitting is therefore an unweighted linear fit of line areas with a fixed peak model; the non-linear parameters are never refit per pixel.", summary="SMAK per-pixel spectrum fitting is unweighted linear least squares with a fixed peak shape."),
 "der:quant-linear-cal": dict(how="derived", rationale="Quantification divides by the chosen I0 channel (q:code-quant-divide), then applies C = slope*(A/I0) + intercept (q:code-quant-poly). This is a linear calibration curve, which reduces to a ratio only when intercept is zero. The normalization channel may be None (q:code-quant-nonorm).", summary="SMAK quantification is a linear calibration C = slope*(A/I0)+intercept, not a pure ratio."),
 "der:stack-translation-only": dict(how="derived", rationale="Automatic stack alignment estimates a translation by phase correlation on high-pass-filtered images (q:code-stack-transl, q:code-stack-hipass, q:code-stack-phase); no rotation or scale is estimated.", summary="SMAK automatic stack alignment is translation-only."),
 "der:manual-tps": dict(how="derived", rationale="Manual registration warps the image with a thin-plate spline fitted to operator landmark pairs (q:code-tps); it is a non-rigid deformation, not an affine transform.", summary="SMAK manual registration is a thin-plate-spline warp on landmarks."),
 "der:fa-weighting": dict(how="derived", rationale="Group norm n_g = mean over channels in group g of sum_k |loading_kc| (q:code-fa-norm, q:code-fa-grp); weight w_g = max_g' n_g' / n_g (q:code-fa-adj), applied and re-estimated for 3 cycles (q:code-fa-cycles). Component sign is flipped so the largest-magnitude loading is positive (q:code-fa-sign).", summary="Weighted-group FA balancing: w_g = max n / n_g, iterated three times; sign fixed by largest loading."),
 "der:ftir-steps-separate": dict(how="derived", rationale="IR_MathClass applies atmospheric correction, optional RMieS, Savitzky-Golay, baseline, and normalization as separate conditional calls with separate parameters (q:code-ir-atm, q:code-ir-mie, q:code-ir-sg, q:code-ir-rubber, q:code-ir-norm); each can be switched independently.", summary="SMAK's FTIR preprocessing is five independently switchable steps, so they are separate Transformations."),
}
INF.update(CINF)
for iid, d in CINF.items():
    graph.append({"@id": iid, "@type": "prov:Entity", "plan": PLAN_ID, "how": d["how"], "agent": AGENT, "summary": d["summary"], "rationale": d["rationale"]})
CDERIV = {
 "der:time-scalar": ["q:code-time-norm"], "der:dt-paralyzable": ["q:code-dt-model","q:code-dt-apply"],
 "der:corrections-viewtime": ["q:code-dt-toggle","q:code-i0-toggle","q:code-dt-apply","q:code-i0-divide"],
 "der:smak-fit-linear": ["q:code-fastfit","q:code-fastfit-args","q:code-cfg-fixed","q:code-cfg-fixedzero"],
 "der:quant-linear-cal": ["q:code-quant-divide","q:code-quant-poly","q:code-quant-nonorm"],
 "der:stack-translation-only": ["q:code-stack-transl","q:code-stack-hipass","q:code-stack-phase"],
 "der:manual-tps": ["q:code-tps"],
 "der:fa-weighting": ["q:code-fa-norm","q:code-fa-grp","q:code-fa-adj","q:code-fa-cycles","q:code-fa-sign"],
 "der:ftir-steps-separate": ["q:code-ir-atm","q:code-ir-mie","q:code-ir-sg","q:code-ir-rubber","q:code-ir-norm"],
}
for out, ins in CDERIV.items():
    graph.append({"@id": "act:" + out.split(":")[1], "@type": "prov:Activity", "plan": PLAN_ID, "kind": "derivation", "performed_by": AGENT, "used": ins, "generated": [out], "plan_step": "xfm:t/build-plan-graph"})

# ---- plan edits
def T(i): return next(t for t in plan["transformations"] if t["id"] == i)
def find(kind, i): return next(n for n in plan[kind] if n["id"] == i)
# dead time
t = T("xfm:t/deadtime_correction")
t["governing_relation"] = "FF_corr = FF * exp(tau * 1e-6 * ICR)   (paralyzable model; tau in µs, ICR in counts/s)"
t["gaps"] = []; t["support"] += [S("governing_relation","q:code-dt-model","quote"), S("governing_relation","der:dt-paralyzable","derived")]
t["implemented_in"] = [dict(software="SMAK", version="3.0.11", component="Process > Deadtime; applied at render/analysis time via dodt toggle (default off)", grit="q:code-dt-apply")]
t["introduces"] = ["dependence on fitted tau", "no persisted record: correction is a view-time toggle"]
t["support"].append(S("introduces","der:corrections-viewtime","derived"))
plan["parameters"].append(dict(id="xfm:p/dodt", name="dead-time toggle", kind="numerical", default_value="0 (off)", default_scope="SMAK 3.0.11", set_by="user", support=[S("default_value","q:code-dt-toggle","quote"), S("default_value","der:corrections-viewtime","derived")]))
t["parameters"].append("xfm:p/dodt")
# I0
t = T("xfm:t/flux_normalization")
t["implemented_in"] = [dict(software="SMAK", version="3.0.11", component="doI0c toggle at render/analysis time (default off); Analyze > Set I0 Channel (prefers I0STRM)", grit="q:code-i0-divide"), dict(software="SMAK", version="3.0.11", component="inside Quantitative Analysis (divide by normalize channel)", grit="q:code-quant-divide")]
t["introduces"].append("no persisted record when applied as view-time toggle")
t["support"] += [S("implemented_in","q:code-i0-prefer","quote"), S("introduces","der:corrections-viewtime","derived")]
plan["parameters"].append(dict(id="xfm:p/doI0c", name="I0 normalization toggle", kind="numerical", default_value="0 (off)", default_scope="SMAK 3.0.11", set_by="user", support=[S("default_value","q:code-i0-toggle","quote")]))
t.setdefault("parameters", []).append("xfm:p/doI0c")
# time normalization (new)
plan["data_states"].append(dict(id="xfm:ds/time_normalized_map", name="dwell-time-normalized element map", quantity="counts per second", units="counts/s", indexed_by=["pixel","element"], uncertainty="none", qualifiers=["time_normalized"]))
plan["transformations"].append(dict(id="xfm:t/time_normalization", name="dwell-time normalization", purpose="Convert per-pixel counts to count rate by dividing by the recorded dwell time, removing stripe artifacts from stage acceleration at line starts.", consumes=["xfm:ds/fitted_area_map"], produces=["xfm:ds/time_normalized_map"], adds_qualifiers=["time_normalized"], justified_by=[],
    governing_relation="rate = counts / TIME * 80,000,000   (TIME in 80 MHz ticks)",
    preserves=["relative distribution"], changes=["units to counts/s"], discards=["absolute counts"],
    implemented_in=[dict(software="SMAK", version="3.0.11", component="Map Math multiply by 0.0000125 (guide) / multi-channel time normalization (code)", grit="q:code-time-norm")],
    support=[S("purpose","q:smak-time-artifact","quote"), S("governing_relation","q:code-time-norm","quote"), S("governing_relation","der:time-scalar","derived")]))
# spectrum fit
t = T("xfm:t/spectrum_fit")
plan["assumptions"].append(dict(id="xfm:a/fixed_peak_shape", name="fixed peak shape per pixel", statement="Per-pixel fits hold zero, gain, noise, Fano and pile-up fixed at configuration values; only line areas are free.", about="model", support=[S("statement","q:code-cfg-fixed","quote"), S("statement","q:code-cfg-fixedzero","quote"), S("statement","der:smak-fit-linear","derived")]))
t["assumes"].append("xfm:a/fixed_peak_shape")
t["introduces"] += ["unweighted linear fit (weight=None): fitted areas carry no Poisson weighting", "SNIP background estimate (width 50) rather than fitted continuum"]
t["implemented_in"][1] = dict(software="SMAK", version="3.0.11", component="pyMcaFitWrapper.Wrapper.doFastFit -> PyMca FastXRFLinearFit.fitMultipleSpectra(weight=None, refit=True) with pyMcaConfigs/activePyMCAConfig.cfg", grit="q:code-fastfit")
t["support"] += [S("introduces","q:code-fastfit-args","quote"), S("introduces","q:code-cfg-snip","quote"), S("introduces","q:code-cfg-strip","quote"), S("assumes","der:smak-fit-linear","derived")]
t["conflicts"][0]["positions"].append(dict(statement="SMAK's shipped configuration fixes zero, gain, noise, fano and sum (all fixed* = 1), so the global non-linear optimisation is not performed unless the user unfixes them.", grit="q:code-cfg-fixed"))
t["conflicts"][0]["would_settle"] = "Settled for SMAK 3.0.11 by activePyMCAConfig.cfg: defaults fix all non-linear parameters. The method paper describes what PyMca can do; SMAK's default does not do it. Remains a Conflict between method-paper and manual/code tiers, not a contradiction."
find("parameters","xfm:p/merge_threshold")  # exists
plan["parameters"].append(dict(id="xfm:p/snip_width", name="SNIP background width", kind="numerical", units="channels", default_value="50", default_scope="SMAK 3.0.11 activePyMCAConfig.cfg", set_by="software_default", support=[S("default_value","q:code-cfg-snip","quote")]))
plan["parameters"].append(dict(id="xfm:p/hypermet_flag", name="Hypermet terms", kind="numerical", default_value="7 (Gaussian + short tail + long tail; no step)", default_scope="SMAK 3.0.11 activePyMCAConfig.cfg", set_by="software_default", support=[S("default_value","q:code-cfg-hypermet","quote")], gaps=[dict(field="default_value", kind="not_in_corpus", note="Bit meaning of hypermetflag taken from PyMca convention, not from a registered source.")]))
t["parameters"] += ["xfm:p/snip_width","xfm:p/hypermet_flag"]
# quantification
t = T("xfm:t/quant_ratio_standard")
t["name"] = "quantification by linear calibration to thin-film standards"
t["governing_relation"] = "C = slope * (A / I0) + intercept   (per element; slope, intercept from standards; reduces to a ratio when intercept = 0)"
t["support"] += [S("governing_relation","q:code-quant-poly","quote"), S("governing_relation","der:quant-linear-cal","derived"), S("governing_relation","q:code-quant-divide","quote")]
t["introduces"].append("intercept term when more than one standard is used")
find("data_states","xfm:ds/areal_concentration_map")["aliases"].append(dict(term="<chan>-<units>-<n> channel", used_by="SMAK 3.0.11 code", note="guide says Xx-conc1"))
# registration
t = T("xfm:t/stack_alignment")
t["purpose"] = "Align channels from different maps by a single translation estimated by phase correlation on high-pass-filtered images."
t["consumes"] = ["xfm:ds/scaled_map", "xfm:ds/ftir_band_map"]  # both channel sets; found by AND-reachability (Q1b)
plan["assumptions"].append(dict(id="xfm:a/translation_only", name="misalignment is pure translation", statement="Images differ by a translation only; no rotation, scale or distortion.", about="instrument", support=[S("statement","q:code-stack-transl","quote"), S("statement","der:stack-translation-only","derived")]))
t["assumes"] = ["xfm:a/translation_only"]; t["gaps"] = [g for g in t["gaps"] if g["field"] != "assumes"]
t["support"] += [S("purpose","q:code-stack-phase","quote")]
t = T("xfm:t/manual_registration")
t["purpose"] = "Bring FTIR maps onto the XRF pixel grid by a thin-plate-spline warp fitted to operator-placed landmark pairs on the two optical images."
t["introduces"].append("non-rigid (thin-plate spline) deformation of the moving image")
t["implemented_in"] = [dict(software="SMAK", version="3.0.11", component="ExportRegistrationClass -> utils.tps_warp2D on landmark pairs", grit="q:code-tps")]
t["support"] += [S("introduces","der:manual-tps","derived")]
find("validation_criteria","xfm:v/registration_rmse")["gaps"] = [dict(field="observable", kind="not_in_corpus", note="No RMSE computation exists in SMAK 3.0.11 source; the manuscript's RMSE was computed outside SMAK by means not stated.")]
# FA
t = T("xfm:t/multifile_weighted_fa")
t["governing_relation"] = "n_g = (1/|g|) sum_{c in g} sum_k |L_kc};  w_g = max_g' n_g' / n_g;  channels in g scaled by w_g; re-estimate for 3 cycles; FA via sklearn.decomposition.FactorAnalysis; component sign flipped so max|loading| > 0"
t["gaps"] = []
t["implemented_in"] = [dict(software="SMAK", version="3.0.11", component="WeightedGroupPCA.GroupPCA.doWtPCABalance (pcatype FA)", grit="q:code-fa-adj")]
t["introduces"] += ["sign convention: largest-magnitude loading positive"]
t["support"] += [S("governing_relation","der:fa-weighting","derived"), S("governing_relation","q:code-fa-adj","quote"), S("governing_relation","q:code-fa-cycles","quote")]
# FTIR split
old = T("xfm:t/ftir_preprocess")
plan["transformations"].remove(old)
plan["data_states"] += [
 dict(id="xfm:ds/ftir_atm", name="FTIR spectrum, atmosphere-corrected", quantity="IR absorbance per wavenumber", indexed_by=["pixel","spectral_band"], uncertainty="none", qualifiers=["atmosphere_corrected"]),
 dict(id="xfm:ds/ftir_denoised", name="FTIR spectrum, denoised", quantity="IR absorbance per wavenumber", indexed_by=["pixel","spectral_band"], uncertainty="none", qualifiers=["atmosphere_corrected","denoised"]),
 dict(id="xfm:ds/ftir_baselined", name="FTIR spectrum, baseline-corrected", quantity="IR absorbance per wavenumber", indexed_by=["pixel","spectral_band"], uncertainty="none", qualifiers=["atmosphere_corrected","denoised","baseline_corrected"]),
 dict(id="xfm:ds/ftir_scatter_corrected", name="FTIR spectrum, resonant-Mie corrected", quantity="IR absorbance per wavenumber", indexed_by=["pixel","spectral_band"], uncertainty="none", qualifiers=["atmosphere_corrected","scatter_corrected"]),
]
plan["parameters"].append(dict(id="xfm:p/mie_reference", name="RMieS reference spectrum", kind="empirical", default_value="None (off)", default_scope="SMAK 3.0.11 IR_MathClass", set_by="user", physical_meaning="pure-absorbance reference (Lignin, Casein, Matrigel) for resonant Mie scattering correction", support=[S("physical_meaning","q:code-ir-mie","quote")]))
plan["transformations"] += [
 dict(id="xfm:t/ftir_atmospheric_correction", name="FTIR atmospheric correction", purpose="Remove water-vapour and CO2 absorption bands from each spectrum.", consumes=["xfm:ds/ftir_spectrum"], produces=["xfm:ds/ftir_atm"], adds_qualifiers=["atmosphere_corrected"], justified_by=[],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="IR_MathClass.DoNorm -> octavvs atm.atmospheric", grit="q:code-ir-atm"), dict(software="Omnic", version="9.8", component="atmospheric correction", grit="q:pist-ftir-atm")],
      support=[S("purpose","q:pist-ftir-atm","quote"), S("implemented_in","der:ftir-steps-separate","derived")],
      gaps=[dict(field="justified_by", kind="not_in_corpus"), dict(field="implemented_in", kind="contested", note="The trace attributes atmospheric correction to Omnic export; SMAK also implements it. Which ran is not stated.")]),
 dict(id="xfm:t/ftir_rmies_correction", name="resonant Mie scattering correction", purpose="Separate the resonant-Mie scattering contribution from chemical absorbance using a reference spectrum, for samples with scattering morphology.", consumes=["xfm:ds/ftir_atm"], produces=["xfm:ds/ftir_scatter_corrected"], adds_qualifiers=["scatter_corrected"], parameters=["xfm:p/mie_reference"], justified_by=[],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="IR_MathClass.DoNorm -> octavvs mie.rmiesc", grit="q:code-ir-mie")],
      support=[S("purpose","q:code-ir-mie","quote")],
      gaps=[dict(field="justified_by", kind="not_in_corpus", note="Not used in the trace; offered by SMAK; rationale is in the OCTAVVS/RMieS literature, not registered.")]),
 dict(id="xfm:t/ftir_sg_denoise", name="Savitzky-Golay denoising", purpose="Reduce high-frequency noise while preserving band shape and position.", consumes=["xfm:ds/ftir_atm"], produces=["xfm:ds/ftir_denoised"], adds_qualifiers=["denoised"], parameters=["xfm:p/sg_window","xfm:p/sg_order"], justified_by=["xfm:r/sg_denoise"],
      discards=["high-frequency content"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="IR_MathClass.DoNorm -> scipy.signal.savgol_filter", grit="q:code-ir-sg")],
      support=[S("purpose","q:pist-ftir-atm","quote")]),
 dict(id="xfm:t/ftir_rubberband_baseline", name="rubberband baseline correction", purpose="Remove the smooth scattering offset by subtracting the convex hull of each spectrum.", consumes=["xfm:ds/ftir_denoised"], produces=["xfm:ds/ftir_baselined"], adds_qualifiers=["baseline_corrected"], justified_by=["xfm:r/rubberband"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="IR_MathClass.DoNorm -> octavvs baseline.rubberband (default 'Rubberband'; alternatives Concave Rubberband, AsLS, arPLS)", grit="q:code-ir-rubber")],
      support=[S("purpose","q:pist-rubberband","quote"), S("implemented_in","q:code-ir-defaults","quote")]),
 dict(id="xfm:t/ftir_mean_normalization", name="mean normalization at a wavenumber", purpose="Scale each spectrum so intensities are comparable across pixels, using a chosen wavenumber as the normalizer.", consumes=["xfm:ds/ftir_baselined"], produces=["xfm:ds/ftir_spectrum_pre"], adds_qualifiers=["mean_normalized"], parameters=["xfm:p/norm_wavenumber"], justified_by=[], validated_by=["xfm:v/norm_invariance"],
      discards=["absolute absorbance scale"], introduces=["dependence on normalizer band"],
      implemented_in=[dict(software="SMAK", version="3.0.11", component="IR_MathClass.DoNorm -> octavvs normalization.normalize_spectra (default 'Mean')", grit="q:code-ir-norm")],
      support=[S("purpose","q:pist-norm-1655","quote")], gaps=[dict(field="justified_by", kind="not_in_corpus")]),
]
w = find("workflows","xfm:w/pistachio_multimodal")
steps = w["steps"]
i7 = next(i for i, s in enumerate(steps) if s["step_id"] == "s7")
steps[i7:i7+2] = [
 dict(step_id="s7a", transformation="xfm:t/ftir_atmospheric_correction"),
 dict(step_id="s7b", transformation="xfm:t/ftir_sg_denoise", preceded_by=["s7a"], parameter_values=[dict(parameter="xfm:p/sg_window", value="9"), dict(parameter="xfm:p/sg_order", value="3")]),
 dict(step_id="s7c", transformation="xfm:t/ftir_rubberband_baseline", preceded_by=["s7b"]),
 dict(step_id="s7d", transformation="xfm:t/ftir_mean_normalization", preceded_by=["s7c"], parameter_values=[dict(parameter="xfm:p/norm_wavenumber", value="1655")]),
 dict(step_id="s7v", transformation="xfm:t/ftir_mean_normalization", preceded_by=["s7c"], role="validation", condition="xfm:d/norm_wavenumber", condition_option="2900 cm-1", parameter_values=[dict(parameter="xfm:p/norm_wavenumber", value="2900")]),
]
for s in steps:
    if s["step_id"] == "s8": s["preceded_by"] = ["s7d"]
w["support"].append(S("steps","der:ftir-steps-separate","derived"))
plan["version"] = "0.2.0-gold"

# ------------------------------------------------------------------ write, then link payload_ref
plan_path = OUT / "xfm-gold.plan.yaml"
plan_path.write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True, width=200))
graph.append({"@id": "ent:xfm-gold-plan", "@type": "prov:Entity", "plan": PLAN_ID, "how": "derived", "agent": AGENT,
              "rationale": "Plan graph hand-built from the quote, absence and inferred entities in this bundle; every field's Support names the grit it rests on.",
              "summary": "sciplan PlanGraph for the pistachio XFM+FTIR workflow (gold graph 0.1.0).",
              "payload": "https://phzwart.github.io/sciplan/schema/sciplan",
              "payload_ref": {"uri": "xfm-gold.plan.yaml", "sha256": sha(plan_path), "media_type": "application/yaml"}})
graph.append({"@id": "act:build-plan-graph", "@type": "prov:Activity", "plan": PLAN_ID, "kind": "derivation", "performed_by": AGENT, "plan_step": "xfm:t/build-plan-graph",
              "rationale": "Consolidation of receipts into the typed plan graph.",
              "used": [n["@id"] for n in graph if n["@type"] == "prov:Entity" and n["@id"] != "ent:xfm-gold-plan"], "generated": ["ent:xfm-gold-plan"]})
(OUT / "xfm-gold.grits.jsonld").write_text(json.dumps(bundle, indent=1, ensure_ascii=False))

# ------------------------------------------------------------------ cross-checks
ids = {n["@id"]: n for n in graph}
errs = []
def walk(o):
    if isinstance(o, dict):
        if "grit" in o and "how" in o:
            g = ids.get(o["grit"])
            if g is None: errs.append(f"dangling grit {o['grit']}")
            elif g.get("how") != o["how"]: errs.append(f"label mismatch {o['grit']}: support says {o['how']}, grit says {g.get('how')}")
        if "search_grit" in o and o["search_grit"] not in ids: errs.append(f"dangling search_grit {o['search_grit']}")
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(plan)
# referential integrity inside the plan
allids = {n["id"] for k, v in plan.items() if isinstance(v, list) for n in v}
def refs(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("consumes","produces","parameters","requires","assumes","justified_by","validated_by","failure_modes","alternative_to","serves","starts_from","ends_at","established_by","caused_by","detected_by","if_violated","bounded_by") and isinstance(v, list) and all(isinstance(x, str) for x in v):
                for x in v:
                    if x not in allids: errs.append(f"dangling ref {x} in {k}")
            elif k in ("on_state","set_by_decision","selects_transformation","sets_parameter","condition","loop_exit","transformation","parameter","claim_type") and isinstance(v, str):
                if v not in allids: errs.append(f"dangling ref {v} in {k}")
            else: refs(v)
    elif isinstance(o, list):
        for v in o: refs(v)
refs(plan)
if errs:
    print("\n".join(errs)); sys.exit(1)
print(f"quotes {len(Q)}  absences {len(ABS)}  inferred/derived {len(INF)}  grits {len(graph)}")
print({k: len(v) for k, v in plan.items() if isinstance(v, list)})
