# OWAS: data, results and supplementary methods

Companion to **Expert-Referenced Evaluation of Multimodal AI for OWAS Coding Across Workplace Visual Conditions** (revised manuscript: 28 October 2026; release v4).

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
- `docs/MANUSCRIPT_REFERENCES.md`: the 22 references aligned to the revised manuscript, preserving its numbering; not a new bibliographic verification.
- `audit/`: statistical resampling settings and release verification evidence.
- `MANIFEST.json` and `SHA256SUMS.txt`: relative paths, byte sizes and SHA-256 checksums. These two manifest files exclude themselves from their inventories.

## Design and interpretation

One worker contributes 69 coverall and 112 everyday-workwear frame pairs. The two AI labels, M56 and M61, are archival experiment labels, not independently authenticated provider model IDs. Eight information conditions yield 181 × 2 × 8 = 2,896 outputs. Each frame has three expert consensus codes, not three independent participants. Load is fixed at category 1, not visually estimated.

At 100/250/400 lux, counts are 15/22/32 (coverall), 37/36/39 (everyday) and 52/58/71 pooled. The recordings are not matched clothing interventions; lighting groups contain different task moments. Same-number frame IDs across recordings are distinct: always use `dataset:frame_id`. Preserve leading zeros and literal `NA`.

The primary results use original issued codes. Candidate substitution is explicitly secondary; preserved English `short_note` fields supply the reference-blind candidate parser. Candidate sets are not new AI predictions and their coverage is not selected-code accuracy. No images are recoded and no AI service is called.

See `../owas-code/docs/PROVENANCE_AND_LIMITATIONS.md` for inference limitations, the unresolved inherited prior-preview entry and the scope of public reproducibility. These privacy-minimized archives do not establish legal/ethical release authorization or grant an open-source/data licence; see `LICENSE_STATUS.md`.

## Added in v3

- `prompts/`: all eight complete supplied prompt files, unchanged in content, with version and SHA-256 metadata.
- `reference/`: the independently frozen OWAS action table and source provenance. Fixed-load codes 3141 and 3151 now map to actions 3 and 4; existing study results are unaffected.
- `audit/protocol_metadata_flags.csv`, `protocol_metadata_normalization.csv`, `protocol_metadata_QA.json`: separate metadata audit and deterministic normalizations. The original input CSVs and posture digits are unchanged.
- `docs/PROTOCOL_AND_PROMPTS.md`, `ARCHIVED_METADATA_DEVIATIONS.md`: protocol details and audit scope.
- `CHANGELOG.md`: manuscript and package changes, including retained investigator-confirmed session isolation.

Only `results/action_lookup_load1.csv` changes among the 33 scientific result CSVs. All other scientific result tables are preserved from v2. The original v2 release verification is explicitly archived separately; the current verification describes v4. `generated/`, bytecode and local font caches are omitted from the code ZIP and are recreated by the runner.

## Packaging update in v4

The investigator information checklist is supplied separately as `OWAS_Yazarlardan_Beklenen_Bilgiler.docx`; it is not included in either ZIP. Links and file inventories are updated. All numerical inputs, all 33 scientific CSV tables, the three protocol audit files, four figures and the supplementary DOCX are byte-identical to v3.
