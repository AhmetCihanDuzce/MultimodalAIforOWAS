# Supplement S2. Data and code for expert-supervised OWAS coding

Two development recordings comprise 181 paired observations. The held-out recording contains 113 paired observations from the same CNC operator. The primary reference is the experts’ consensus based on the two synchronized photographs. The fourth OWAS digit (load) is fixed at 1. Each observation is identified by the combination of recording and frame ID. An unresolved full code is retained in the denominator.

## Reproduce the reported results

Extract this ZIP to a writable directory. From that directory run:

```bash
python reproduce.py
python draw_figures.py
```

The numerical replay uses pandas (tested with 2.2.3). Drawing Figures 2–4 additionally uses matplotlib (tested with 3.10.8). Outputs are written to `reproduced/`. The commands run offline on the distributed data. The workflow diagram is generated from its recorded design; the other two figures use recounts from predictions. Figure 1 appears in the article. The photographic inputs to that figure and the full image collection are not distributed here.

- `raw_AI_summary.csv` reproduces the exact and complete direct-LLM counts by condition used in Table 3.
- `primary_image_model.csv` and `integrated_model_summary.csv` reproduce the exact-match counts in Table 4 and Supplementary Table S3.
- `paired_test_transitions.csv` reproduces Table 5 and Figure 3. Each method in this table is compared with the direct paired-image answer from the same LLM.
- `test_outcomes.csv` recounts complete codes, exact codes, exact action categories, and high/low priority errors across all 66 method configurations. It includes both sides of the eight-code M61 SVM comparison in Figure 4 and Section 3.4.
- `class_support.csv` reproduces Supplementary Table S1. The script also checks the development paired-image counts by illuminance and the worked example for test observation 00129.

## Replay model inference from saved features

```bash
python predict_test.py
```

This optional replay uses the released image features, archived LLM codes and fitted model files. It checks model hashes and compares the resulting 7458 predictions to the archive by recording, frame, workflow and method. It reads no expert codes and performs no fitting. The compatible inference environment includes NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, joblib 1.5.3 and threadpoolctl. The pipeline does not need the source photographs or a live LLM to replay these predictions. The original feature-extraction environment is documented in `development/requirements-ml.txt`.

## File guide

| Path | Material |
| --- | --- |
| development/audit/paired_manifest.csv | Expert codes for front, side and paired presentations, observation identifiers and development group fields |
| development/inputs | Development LLM codes, expert inputs and OWAS action-category lookup |
| development/held_out_predictions.csv.gz | Observation-level development predictions for 60-second grouping, the 30-second sensitivity analysis and transfer between the two development recordings |
| development/features and development model subdirectories | Derived image features, feature extraction and archived development classifier implementations |
| test/references.csv and test/ai_codes.csv | Three expert presentations and eight LLM input conditions per workflow on the test recording |
| test/features and test/frozen_models | Test feature arrays, fitted classifiers and their configurations |
| test/frozen_predictions.csv | Test predictions from all 66 configurations, including shared controls |
| test/*metrics.csv, test/confusion_counts.csv and test/paired_contrasts.csv | Archived summaries, component metrics, sensitivity analyses and paired contrasts |
| prompts | The eight English prompts archived for development |
| published_tables and verification | Published source counts and archived checks of data and model outputs |
| SHA256SUMS.txt | File hashes for the unmodified release, excluding generated replay outputs |

Component prediction zero denotes an unresolved component. The full-code field may contain the literal value NA. In the numerical replay, an unresolved predicted action category is designated 0. Expert action categories are 1–4. “High” denotes categories 3–4 and “low” categories 1–2; retaining a high category does not require an exact match between categories 3 and 4. M56 and M61 identify the archived workflows. Historical labels in code and file names, including job3 and third_job, refer to the third recording of the same operator.

The stand-alone replay commands above are the entry points for this release. Development scripts and protocols document the original fitting procedures; some archived training and image-acquisition paths follow the original research environment. Access to source photographs is subject to participant consent and institutional permission through the corresponding author.
