# Provenance, exclusions and reproducibility limits

## Manuscript alignment

This release supports **Expert-Referenced Evaluation of Multimodal AI for OWAS Coding Across Workplace Visual Conditions**, manuscript snapshot 27 October 2026. The source manuscript SHA-256 is `e611c92e13106055f43d880c65ba4e514daeedd166f4d2f2ef3190a45e8421ad`. Its current title, 181 frame pairs (69 + 112), 2 models × 8 conditions, 2,896 outputs, 543 expert codes and 21-reference bibliography were used for release alignment. The manuscript file itself is intentionally excluded. No scientific outcome was edited to improve agreement.

## Frozen numerical source and adaptation

The supplied complete-analysis and reproducibility archives contain byte-identical result CSVs. Every original AI coding CSV row was compared field-for-field with the archived coding matrix (2,896 rows) before public sanitization. The original independent check was rerun against source workbook XML: all 543 consensus codes, 144 performance rows, 2,016 category rows, 564 paired tests, 288 illuminance tests, 9 Holm families, 1,058 candidate memberships and 5,792 derived outputs passed.

The public release starts from extracted CSV inputs, not the private original workbooks or image archives. Source-image paths and source worksheet names/row coordinates were removed; original source filenames and unused workbook sheets were not carried forward. Original issued codes, expert targets, prior fields, numerical candidate clauses, frame order and scientific calculations are preserved. Public input/source/output checks and a fresh full numerical reproduction are documented in `../owas-data-results/audit/RELEASE_VERIFICATION.json`.

The public independent check compares 543 consensus values against the released CSV, not against a workbook that is no longer present. It therefore does not independently re-establish source expert judgements, AI inference or image acquisition. Changing the input adapter does not imply that private source files can be recovered from these archives.

## Statistical seeds are not AI-generation seeds

The archived analysis uses NumPy `default_rng`, 10,000 replicates, circular moving blocks of selected-frame-ID order, recording × illuminance strata, and whole frame vectors. All 16 model/condition evaluations for a frame remain together. Seeds are 20261028 for block length 1, 20261030 for length 3 and 20261032 for length 5. These fixed statistical seeds reproduce resampling only. They do not reproduce an AI run or establish provider model identity, temperature, generation seed or session state. No AI-generation seed is invented or supplied.

## Inference/session transparency

The original outputs are archived experiment records, with one recorded generation per model/condition/frame cell. Full provider model identifiers, generation settings and strict cross-run context isolation were not independently verified. The public files do not contain a complete prompt/session transcript and cannot reconstruct the original AI experiment.

A source QA log for **M56, everyday-workwear recording, Condition 4** explicitly documents that a short raw prior-CSV preview exposed excluded full-result columns before a six-column restriction was applied. The log states that excluded values were not used for coding, but this is a run-record assertion, not independently verified proof of code-only exposure. Consequently strict code-only exposure cannot be claimed for that run. This statement was verified in the source QA log during release preparation; it cannot be inferred from the numerical CSV alone. The private QA log and its original filenames are not redistributed. Prior-field equality checks (9,050 values) establish matching embedded codes, not isolation or uncontaminated prompting.

## Statistical and design limitations

- One worker and two recording sets do not establish new-worker performance or population reliability. There is no randomized clothing or lighting intervention.
- Ordinary p values/Wilson intervals assume frame independence and are exploratory sensitivity calculations. Nine separate Holm families contain 870 calculations; this is not experiment-wide multiplicity control over all exploration.
- Ordered-block percentile limits are conditional, pointwise sensitivity intervals. Temporal spacing, stationarity and coverage for a new worker/session are not established.
- NA is an unresolved output, not an independently annotated visibility label. Action categories are derived from the fixed-load lookup. Candidate-set coverage is not an observed model choice or clinical/safety validation.

## Excluded from both ZIPs

Original participant photographs/video, image archives, the main manuscript and its embedded photographs, signed consent, personal names in original source paths/workbooks, unused workbook sheets, raw chats and full session logs, private editorial notes, obsolete manuscript builders, historical manuscript/reference-numbering QA, and previous reference lists. The supplement contains only a numerical plot. Participant identity is confidential and is not linked to any author group here.

No source photos were replaced by synthetic examples. No new AI coding was performed. No external repository was created, no GitHub upload was made, and no third party was contacted. Before any public upload, investigators should confirm release authority and choose appropriate code/data licences.
