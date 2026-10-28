# Provenance exclusions and reproducibility limits

## Manuscript alignment

This v4 release supports **Expert-Referenced Evaluation of Multimodal AI for OWAS Coding Across Workplace Visual Conditions**, revised 28 October 2026. The source manuscript SHA-256 is `002a53799038eae4f8bd4553ad33432eebf7986927e7ff9e1ef23a2803ab070e`. The aligned manuscript contains 181 frame pairs (69 + 112), 2 versions × 8 conditions, 2896 outputs, 543 consensus codes and 22 references. The manuscript itself is excluded from both numerical ZIPs. Changed and added manuscript text is purple in the separately provided DOCX.

## Preserved observations and revised constants

Both public input CSVs are preserved byte-for-byte from v2. Every scientific result CSV except `action_lookup_load1.csv` is preserved from v2. The two changed load-1 entries, 3141 and 3151, are absent from every released expert, original AI, embedded prior and derived complete code, so the study's numerical performance results do not change. Three otherwise unused load-2/3 constants were also corrected against the same source. See `../owas-data-results/reference/REFERENCE_PROVENANCE.md`.

Historical source-stage assertions, including private-workbook XML checks, are retained in `../owas-data-results/audit/RELEASE_VERIFICATION_v2_ARCHIVE.json`. The public checks start from released CSV inputs and do not re-establish private expert judgements or image acquisition. The retained v3 computational validation and current packaging status are recorded in `audit/RELEASE_VERIFICATION.json` in that package.

The 326 metadata schema flags and 118 deterministic metadata normalizations are separate derived audit records. They do not overwrite original issued digits, prior values, evidence strings or notes. No image was recoded and no AI call was made.

## Inference and session transparency

The original outputs are archived experiment records, with one recorded generation per model/condition/frame cell. Each evaluation was conducted as a separate, isolated session by the investigators. Except in Conditions 4–8, where the specified previous AI codes were deliberately supplied as inputs, each session received only the current prompt and the images defined for that condition; no information from other evaluation sessions was intentionally provided.

Eight complete prompt files are now supplied, dated 26 October 2026: version 1.0 for Conditions 1–3 and version 1.1 for Conditions 4–8. Full session transcripts, provider model identifiers and all generation settings are not independently established by the result exports. See `../owas-data-results/docs/PROTOCOL_AND_PROMPTS.md` for the prompt files and source-definition reference.

An inherited audit entry names a prior-preview deviation in M56 everyday-workwear Condition 4, but the underlying log and scope were not supplied. This entry does not identify which fields were exposed or establish an effect on coding. It remains an outstanding investigator-record item in the separately supplied `OWAS_Yazarlardan_Beklenen_Bilgiler.docx` (outside both ZIPs); the 9050 matching prior fields do not resolve it. The investigator-confirmed between-session isolation statement is retained unchanged.

## Statistical seeds and scope

The archived analysis uses NumPy `default_rng`, 10000 replicates, circular moving blocks of selected-frame-ID order, recording × illuminance strata, and whole frame vectors. All 16 model/condition evaluations for a frame remain together. Seeds are 20261028 for block length 1, 20261030 for length 3 and 20261032 for length 5. These are statistical seeds, not AI-generation seeds.

One worker and two recording sets do not establish new-worker performance or a causal clothing/lighting effect. Ordinary p values and Wilson intervals assume frame independence. Nine separate Holm families contain 870 calculations and do not control all exploration as one family. Ordered-block limits are conditional pointwise sensitivity intervals with unverified temporal spacing and population coverage. NA is not an independently annotated visibility variable. Candidate-set coverage is not selected-code accuracy.

## Exclusions and release status

Original participant photographs/video, signed consent, private workbooks and paths, raw chats and full session logs, and the main manuscript are excluded. The supplement contains no participant images. The prompt source-reference PDF is linked rather than bundled. No new repository upload or third-party message is performed by preparing these ZIPs. Code and data licence choices remain with the investigators; `LICENSE_STATUS.md` is unchanged.
