# OWAS analysis code and statistical seeds

Companion code for **Expert-Referenced Evaluation of Multimodal AI for OWAS Coding Across Workplace Visual Conditions** (28 October 2026 revised manuscript; v4).

## Extract both ZIPs together

The public release is split into two sibling directories. Do not place one inside the other:

```text
release/
├── owas-code/
│   ├── run_reproduction.py
│   ├── requirements.txt
│   └── ...
└── owas-data-results/
    ├── inputs/
    ├── results/
    ├── figures/
    ├── audit/
    └── supplement/
```

The data archive contains the deidentified numerical inputs, full frozen scientific result tables, numerical figures and supplementary explanations. The code archive contains executable analysis, independent checks, dependency pins and provenance documentation. See `docs/PROVENANCE_AND_LIMITATIONS.md` before interpreting the results.

## Reproduce

Use **Python 3.12**. The release was verified using Python 3.12.14 and the exact direct-package versions in `requirements.txt`.

From `owas-code` on Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_reproduction.py
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_reproduction.py
```

If PowerShell script execution is restricted, use `.venv\Scripts\python.exe` directly instead of activation. After dependencies are installed, reproduction needs no internet access, API key, model access, photographs, optimization solver or document renderer. Package installation may need internet access. No participant images, original AI calls or manuscript-generation scripts are included.

The runner resolves all paths relative to its own location, checks the required direct-package versions, and sets numerical-library thread counts to one for its child processes. It first verifies both payload manifests with `verify_checksums.py`, then runs:

1. `analyse_study.py` — recompute original-output and derived-policy analyses from the public CSV inputs.
2. `independent_checks.py` — independently check all 543 released consensus codes, outcome counts, exact tests, Holm adjustments and candidate-policy invariants.
3. `audit_protocol.py` — regenerate metadata flags and separate deterministic metadata normalizations without changing original coding inputs.
4. `make_figures.py` — regenerate four numerical PNG figures.
5. `verify_release.py` — compare all 33 scientific CSV tables, three protocol audit files and the resampling settings with the frozen release.

Fresh outputs go to `owas-code/generated/results/`, `owas-code/generated/audit/` and `owas-code/generated/figures/`. Frozen files in `owas-data-results` are never overwritten. The main final validation record is `generated/results/reproduction_QA.json`; independent checks and runtime versions are recorded alongside it. Text and integer values must agree exactly; non-integer numeric values may differ by at most `1e-9` in absolute value. The comparison does not claim byte-identical PNG rendering across operating systems.

`requirements.txt` preserves the seven direct-package pins from the archived analysis environment. `openpyxl` and `python-docx` remain in that environment record although the public CSV-only runner does not need to read private workbooks or generate Word documents. It is not a complete transitive dependency lockfile.

## Statistical seeds — not AI generation seeds

| Circular block length | NumPy `default_rng` seed | Replicates |
| --- | ---: | ---: |
| 1 | 20261028 | 10000 |
| 3 | 20261030 | 10000 |
| 5 | 20261032 | 10000 |

The exact implementation is in `ordered_bootstrap()` in `analyse_study.py`; the machine-readable settings are in `statistical_resampling.json` (with a frozen audit copy in `../owas-data-results/audit/resampling_settings.json`). These seeds control **statistical resampling only**. They are not model-generation seeds, cannot recreate the original AI responses, and do not establish repeated-run stability.

Resampling is stratified by recording and illuminance (100, 250, 400 lux), uses circular moving blocks in selected-frame-ID order, and retains all 16 model/condition evaluations for each sampled frame together. Intervals are pointwise conditional sensitivity summaries, not verified new-worker confidence intervals. Ordinary descriptive and exact-test calculations are deterministic.

## Numerical inputs and scope

The two public CSV inputs preserve all original issued digits, missing-value markers, embedded prior values, evidence fields and English explanatory notes needed by the analysis. Photographic source paths and private workbook coordinates are removed. Frame IDs must remain strings, including leading zeroes; `dataset:frame_id` is the unique observation key. Identical frame numbers in the two recordings do not denote a matched pair across recordings.

The dataset contains one worker, two recording sets, 181 paired frames (69 coverall and 112 everyday workwear), 2,896 archived model/condition outputs and 543 expert-consensus codes. Load is fixed to code 1, not estimated. The analyses neither recode images nor create new AI responses. The public independent reference check compares the released consensus CSV, not private workbook XML; the latter was checked separately during source audit.

This release supplies an auditable computational analysis, not complete inference-run replication, a claim of verified provider model identity, a causal clothing/illuminance experiment, or an independent-worker validation. The original evaluations were conducted in separate isolated sessions; the provenance document records the session protocol; the inherited prior-preview note remains an explicitly unresolved investigator-record item. Review `LICENSE_STATUS.md` before redistribution under a new license.

## Changes in v3

All eight supplied prompts are in `../owas-data-results/prompts/`, with versions and hashes. The cited coding-definition source is Lins and Hein (2022), distinct from the Lins et al. (2021) reliability study. See `../owas-data-results/docs/PROTOCOL_AND_PROMPTS.md`.

The load-1 lookup entries 3141 and 3151 are corrected to action categories 3 and 4. The full constants are checked against a frozen 252-cell external reference. Neither corrected load-1 code occurs in the observed or derived complete codes, so study performance is unchanged.

The raw input CSVs remain byte-identical. Metadata deviations are documented in separate reproducible files; `PASS_WITH_ARCHIVED_SCHEMA_DEVIATIONS` is an audit outcome, not a claim of complete original schema compliance. Publication facts requiring investigator records and unselected licences are listed in the separately supplied `OWAS_Yazarlardan_Beklenen_Bilgiler.docx`, outside both ZIPs.

## Packaging update in v4

The investigator information checklist is supplied as a separate Word document. References and inventories are updated; every Python script and the numerical analysis remain unchanged from v3.
