# Data dictionary: OWAS public release

27 October 2026

DATA UNITS
frame_key = dataset:frame_id. A recording-specific synchronized pair is the
unit of observation. Same-number identifiers across recordings are distinct.
Recording labels: coverall, everyday, pooled. Pooled reuses both sets; it is
not a third recording. 69 + 112 = 181 frame pairs from one worker.
Original rows: 181 x 2 versions x 8 conditions = 2896 evaluations.
Expert references: 181 x 3 presentations = 543 consensus codes.
illuminance_lux is numeric 100, 250 or 400. Counts are 52, 58 and 71 pooled.
Literal NA marks unavailable component/code/statistic, never a posture class.
Import frame_id as text to preserve leading zeroes.

PRIMARY OUTCOMES
posture_match = all back/arms/legs digits equal paired-image reference.
complete_match = all posture digits issued. Load is constant 1.
action_match = derived action equals paired-image consensus action.
back_match / arms_match / legs_match retain NA as nonmatching.
Original and derived policy streams are kept in separate files.

TABLES
coding_matrix.csv: original responses, expert targets and binary outcomes.
expert_consensus.csv: one row per recording-specific pair, all three references.
performance.csv: 144 dataset/version/condition/reference rows; pooled included.
illuminance_performance.csv: the same cells at each numerical lux level.
component_performance.csv: all-frame and conditional agreement, macro recall,
macro F1 over reference-present categories, and available-pair nominal kappa.
category_performance.csv: precision/recall/F1 with prediction and reference
counts. Unpredicted category precision and absent-category recall are NA.
confusion_matrices.csv: all legal component categories and an NA output column.
reference_composition.csv and reference_agreement.csv: consensus composition
and comparisons across presentations, not initial expert-expert reliability.

TESTS AND CONTRASTS
version_tests.csv: B = M61, A = M56. Gains/losses are paired binary changes.
information_tests.csv: B is condition_b and A is condition_a.
component_tests.csv: B is component_b and A is component_a.
shared_complete_tests.csv: complete-posture outcomes on common coded subset.
omnibus_tests.csv: eight-condition Cochran Q sensitivity calculations.
illuminance_tests.csv: probability-ordered exact three-by-two tables.
All tests carry raw p, Holm p, family, family_n and assumptions.
multiplicity_families.csv: all nine exact correction families, totalling 870
exploratory calculations (564 paired, 288 lux, 18 omnibus).
information_frame_transitions.csv: exact / mismatch / unresolved transitions.
previous_code_transitions.csv: prior error correction, retention or omission.

DEPENDENCE AND COMPOSITION SENSITIVITY
ordered_bootstrap_intervals.csv: complete frame-vector rate intervals.
ordered_bootstrap_contrasts.csv: paired version or information differences.
recording_contrast_differences.csv: everyday minus coverall information changes.
Intervals are pointwise percentile sensitivity results, not population CIs.
resampling_settings.json specifies 10000 replicates, block lengths 1/3/5,
seeds, strata and limitations. The block order is selected frame-ID order.
recording_weight_sensitivity.csv: frame weights versus equal recording weights.
illuminance_standardized_components.csv: reference-category standardized
matching proportions on categories present at all three lux levels in a set.
illuminance_standardization_audit.csv: weights, eligible counts and matches.
These procedures do not identify causal illuminance or clothing effects.

CANDIDATES
candidate_audit.csv: all 1058 original NA components, numerical candidates,
source, preserved numerical candidate excerpt and reference-membership score. Selection does not
consult the reference. Descriptive-only notes are not translated to numbers.
candidate_summary.csv: single accuracy versus multi-candidate coverage.
candidate_set_sizes.csv and candidate_sources.csv: width and provenance.
candidate_derived_matrix.csv: visual_single and all_single policies; all
available original digits remain fixed. Original values and updated outcomes
are retained alongside the derived code; there are 5792 derived evaluations.
candidate_policy_performance.csv: changes in complete/exact/mismatch counts.
candidate_frame_compatibility.csv: full-reference compatibility where every
missing component has explicit candidates; not an observed model choice.
No independent component-visibility variable was inferred.

PROVENANCE
The regenerated input_audit.csv and input_sha256.json identify public CSV inputs, not private source archives. Original image paths, worksheet names and row coordinates are excluded.
prior_provenance_audit.csv: all 9050 prior-field comparisons agree.
action_lookup_load1.csv: all 84 fixed-load code-to-action entries.
*_QA.json documents completed checks and their scope.

PUBLIC INPUT SCHEMA
ai_coding_input.csv retains these source fields as strings: frame_id; back; arms; legs; load; OWAS_code; status; short_note; back_evidence; arms_evidence; legs_evidence; back_prior_influence; arms_prior_influence; legs_prior_influence; prior_source; prior_back; prior_arms; prior_legs; prior_load; prior_OWAS_code; prior_status. dataset, model and combination identify the recording, model label and information condition. Original source_files paths are removed.
expert_consensus_input.csv has dataset, frame_id, illuminance_lux, ref_front_code, ref_side_code and ref_both_code. Ref-derived action and component fields are recomputed by the analysis.

INFORMATION CONDITIONS
1: side image. 2: front image. 3: both images.
4: both images + this model's Condition 1 prior codes.
5: both images + this model's Condition 2 prior codes.
6: both images + this model's Condition 3 prior codes.
7: side image + this model's Condition 2 prior codes (no front image).
8: front image + this model's Condition 1 prior codes (no side image).
Previous codes concern the same frame and model; they are not expert targets.

REPRODUCTION OUTPUTS
Frozen results are under this package's results/. Generated audit/result files are written under ../owas-code/generated/results/. resampling_settings.json is included here under audit/ and regenerated with results. All CSVs use UTF-8 with a BOM and literal NA. Counts, codes and frame IDs should not be silently retyped by a spreadsheet importer.
