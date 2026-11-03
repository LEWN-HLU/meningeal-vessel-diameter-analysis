# Processing and statistical methods

## Vessel-trace processing and quality control

1. Load exported 1-Hz vessel-diameter traces and retain rows marked `accepted`.
2. Validate required columns and consistency of vessel timestamps within each session.
3. Calculate each vessel's baseline mean and SD. The default baseline comprises the first 890 samples; SD uses `ddof=0`.
4. Exclude an entire vessel if any frame reaches the configured extreme absolute z-score threshold (default: |Z| ≥ 20). Points are not clipped to the threshold. Excluded vessels are listed in the QC output.
5. Detect response recordings from time gaps greater than 1.5 seconds. Each response starts at its detected boundary and uses up to 590 samples by default. Durations and thresholds are configurable and recorded with results.
6. Omit isolated missing values from means and SD calculations. Zero or invalid baseline SD does not provide an interpretable normalized response. Unavailable model results must not be interpreted as nonsignificant effects.

Z-score at each time is `(diameter − baseline mean) / baseline SD`. Response-window z-scores are averaged over time within vessel. Absolute change is response-window mean diameter minus baseline mean diameter. Percent change is `100 × absolute change / baseline mean diameter`, calculated separately for each vessel before aggregation.

Small vessels have baseline diameter <20 µm; large vessels have diameter >50 µm. Intermediate vessels (20–50 µm inclusive) are not included in the small-versus-large comparison. Compartment labels are dura, pia or unknown. These labels are annotation metadata, not inferred from response dynamics.

## Statistical implementations

| Manuscript analysis | Method | Implementation |
| --- | --- | --- |
| Fig. 1 treatment response | Independent Mann–Whitney tests for levcromakalim/PACAP; paired Wilcoxon for NTG; two-sided, without across-study correction | `publication.manuscript_statistics` |
| Fig. 2 response classes | Descriptive vessel counts at mean Z ≤ −1.96 and ≥ +1.96 | `publication.manuscript_statistics` |
| Fig. 2 diameter–response association | Response ~ baseline diameter + mouse random intercept; ML and two-sided Wald slope test; Holm across agent slopes | `statistical_methods.linear_relationship_stats`, `holm_adjust` |
| Fig. 2 small/large treatment comparison | Treatment × size fixed effects and mouse random intercept; Wald interaction and two within-size contrasts; Holm within study | `statistical_methods.mixed_model_size_interaction` |
| Fig. 3 baseline diameter by compartment | Compartment and study/source fixed effects; mouse and session random intercepts | `statistical_methods.fit_baseline_compartment` |
| Fig. 3h dura/pia response | Compartment fixed effect and mouse random intercept; partially matched mice retained | `statistical_methods.fit_pacap_compartment` |
| Response relative to baseline | Two-sided one-sample t-tests on mouse means against zero; Holm across dura and pia | `publication.baseline_tests` |
| Diameter–response within compartments | Separate diameter slopes with mouse random intercept; Holm across the two compartment slopes | `statistical_methods.fit_paper_compartment_slope`, `holm_adjust` |
| Treatment effect by compartment | Treatment × compartment fixed effects and mouse random intercept; Wald interaction and two Holm-adjusted within-compartment contrasts | `statistical_methods.treatment_by_compartment` |
| Fig. 4 antibody comparison | Treatment fixed effect and mouse random intercept; two-sided Wald test | `statistical_methods.mixed_model_two_group_mouse_intercept` |
| Fig. 4/S3 category-specific antibody effects | Treatment × compartment or size mixed models; Holm across each two-category contrast family | `statistical_methods.mixed_model_two_group_category_interaction` |
| S4a sequential dosing | Descriptive within-mouse averages for each explicitly timed dose period; no inferential test | `core.dose_helpers`, `publication.inscopix_doses` |
| S4b acute dosing | Kruskal–Wallis and tie-corrected, two-sided Dunn comparisons; Bonferroni across three pairwise comparisons | `publication.dosing_comparison` |
| Supplementary response summaries | Mean, SEM and n across mouse-level Z-score or percent-change values | `publication.descriptive`, `manuscript_statistics` |

Model functions accept the observations for the relevant comparison. `session_values` averages vessel responses within mouse/session and category; `mouse_compartment_values` additionally averages repeated sessions within mouse and compartment. Mouse identity specifies dependence in mixed models, including partially paired conditions. Vessels are not treated as independent animals in treatment-response tests.

The baseline-diameter model is vessel-level and accounts for clustering by mouse and session. Diameter–response slope models are vessel-level with a mouse random intercept. OLS trend summaries and Spearman coefficients returned by diameter helpers are descriptive, not substitutes for the clustered slope test.

Mixed models use maximum likelihood (`reml=False`). Outputs expose estimates, standard errors, Wald tests and diagnostics. Convergence, variance estimates and finite uncertainty estimates should be examined together when evaluating a fit. Holm adjustments apply to the stated comparison families, not indiscriminately to all manuscript tests.

## Time courses

Vessel traces are averaged within each mouse/session and selected category. The default display uses 30-second bins. Group means and SEM are then calculated across session-level values. Separate traces are drawn for each recording block, leaving blank 5-minute display intervals between blocks; no interpolation or inferential testing is applied across those gaps or at individual time bins. Display intervals are a plotting convention, not reconstructed observations.

Size and compartment selections alter the vessels included in the average, not the normalization formula. The statistical unit and input aggregation are specified by each model function rather than by the number of plotted vessels.
