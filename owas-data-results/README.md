# OWAS: data, results and supplementary methods

Companion to **Expert-Referenced Evaluation of Multimodal AI for OWAS Coding Across Workplace Visual Conditions** (manuscript snapshot: 27 October 2026).

## Two-part repository layout

Extract both release ZIPs into the same parent directory, preserving these sibling names:

```text
repository/
  owas-code/                 # Python scripts, pinned dependencies and statistical seeds
  owas-data-results/         # This folder: public numerical inputs and frozen outputs
```

From `owas-code`, run `python run_reproduction.py` after installing its requirements. The run writes only to `owas-code/generated/`; the frozen results here are never overwritten. See the code README for virtual-environment commands. No GitHub repository has been created or uploaded by this package.

## Contents

- `inputs/ai_coding_input.csv`: 2,896 archived coding rows with original issued digits, prior fields, evidence labels and English notes. Original image paths are removed.
- `inputs/expert_consensus_input.csv`: 181 recording-specific frames and 543 consensus codes (front, side, paired). No expert names or original workbook metadata.
- `results/`: every numerical result table from the source analysis, with only unneeded source-path/workbook metadata removed. The old source-file audit is replaced by the public-input audit produced during reproduction.
- `figures/`: four numerical plots; no participant images.
- `supplement/`: English companion DOCX with the current manuscript title and corrected release instructions.
- `docs/DATA_DICTIONARY.md`: table meanings, denominators, units and exclusions.
- `docs/MANUSCRIPT_REFERENCES.md`: the 21 references copied from the latest manuscript, preserving its numbering; not a new bibliographic verification.
- `audit/`: statistical resampling settings and release verification evidence.
- `MANIFEST.json` and `SHA256SUMS.txt`: relative paths, byte sizes and SHA-256 checksums. These two manifest files exclude themselves from their inventories.

## Design and interpretation

One worker contributes 69 coverall and 112 everyday-workwear frame pairs. The two AI labels, M56 and M61, are archival experiment labels, not independently authenticated provider model IDs. Eight information conditions yield 181 × 2 × 8 = 2,896 outputs. Each frame has three expert consensus codes, not three independent participants. Load is fixed at category 1, not visually estimated.

At 100/250/400 lux, counts are 15/22/32 (coverall), 37/36/39 (everyday) and 52/58/71 pooled. The recordings are not matched clothing interventions; lighting groups contain different task moments. Same-number frame IDs across recordings are distinct: always use `dataset:frame_id`. Preserve leading zeros and literal `NA`.

The primary results use original issued codes. Candidate substitution is explicitly secondary; preserved English `short_note` fields supply the reference-blind candidate parser. Candidate sets are not new AI predictions and their coverage is not selected-code accuracy. No images are recoded and no AI service is called.

See `../owas-code/docs/PROVENANCE_AND_LIMITATIONS.md` for inference limitations, the documented prior-preview deviation and the scope of public reproducibility. These privacy-minimized archives do not establish legal/ethical release authorization or grant an open-source/data licence; see `LICENSE_STATUS.md`.
