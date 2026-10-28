# Archived metadata deviations

The raw CSV inputs remain byte-identical to v2. `audit_protocol.py` compares their fields with the supplied prompt schema and regenerates three files under `owas-code/generated/audit/`, compared with frozen copies in this package's `audit/`.

- `protocol_metadata_flags.csv`: 326 flagged fields across 187 original rows. Raw values are retained.
- `protocol_metadata_normalization.csv`: 118 deterministic metadata-only normalizations, retained as a separate derived table rather than written into original inputs.
- `protocol_metadata_QA.json`: counts, input hash and audit scope.

The flags comprise 112 prior statuses (`ok` or `partial_na`) used as source-code availability rather than prior-row validity; 207 free-text evidence fields across 69 coverall M56 Condition 8 rows; five influence flags with an NA prior component; one NA primary with a resolved-evidence label; and one resolved primary with an insufficient-evidence label.

For the 112 prior statuses, unique matching and equality of all five frozen code fields establish `matched_valid` even if a component is NA. Five component prior-influence flags become `not_applicable` because their matched prior component is NA. For coverall/M56/Condition 5/frame 00065, the derived label follows the prompt's rule that a primary NA has evidence `insufficient`.

No supporting camera view is inferred for free-text evidence. For everyday/M56/Condition 6/frame 00028, `legs=4` and `legs_evidence=insufficient` conflict; that case remains flagged for source review. These 208 unresolved fields are not repaired by guessing. Original posture digits, explanatory notes, prior values and performance calculations are preserved.

`PASS_WITH_ARCHIVED_SCHEMA_DEVIATIONS` means the audit ran successfully and documented deviations. It does not claim that all original model outputs followed the requested schema or that visual support has been independently verified.
