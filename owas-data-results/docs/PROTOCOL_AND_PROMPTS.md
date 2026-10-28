# Coding protocol and supplied prompts

Eight investigator-supplied prompt files dated 26 October 2026 are preserved byte-for-byte in `../prompts/`. Download-counter suffixes were removed from filenames only. `PROMPT_MANIFEST.json` records source and release names, versions, conditions and SHA-256 values.

| Condition | File | Version | Current images | Frozen no-prior baseline |
| --- | --- | --- | --- | --- |
| 1 | `01_Side_NoPrior_EN.txt` | 1.0 | side | None |
| 2 | `02_Front_NoPrior_EN.txt` | 1.0 | front | None |
| 3 | `03_Both_NoPrior_EN.txt` | 1.0 | both | None |
| 4 | `04_Both_With_SideCodes_EN_v1_1.txt` | 1.1 | both | 1 |
| 5 | `05_Both_With_FrontCodes_EN_v1_1.txt` | 1.1 | both | 2 |
| 6 | `06_Both_With_BothCodes_EN_v1_1.txt` | 1.1 | both | 3 |
| 7 | `07_Side_With_FrontCodes_EN_v1_1.txt` | 1.1 | side | 2 |
| 8 | `08_Front_With_SideCodes_EN_v1_1.txt` | 1.1 | front | 1 |

The posture-definition source is Lins, C.; Hein, A. *Classification of body postures using smart workwear*. BMC Musculoskeletal Disorders 2022, 23, 921. https://doi.org/10.1186/s12891-022-05821-9. The prompts refer to Tables 2–4, specify load=1 by study protocol, and prohibit AI-generated action/risk classifications. Actions are derived later with the separately documented OWAS lookup.

The prompts require each selected frame to be assessed independently using permitted current images. They prohibit borrowing posture from adjacent frames. Unresolved primary components remain NA, including when a prior supplies a definite value. Explicit numerical candidates belong to `short_note` and are not primary measurements.

Conditions 4–8 accept one unchanged same-model, same-effort no-prior baseline result, either as a full output CSV or an existing six-column extract. Before coding, only `frame_id,back,arms,legs,load,OWAS_code` may be exposed from that prior. Notes, evidence and other fields are excluded. This describes the supplied input-use protocol; it does not resolve the inherited prior-preview audit entry described in the separately supplied `OWAS_Yazarlardan_Beklenen_Bilgiler.docx` (outside both ZIPs).

The investigator has confirmed that the evaluations used separate isolated sessions. The manuscript's session-isolation statement is retained unchanged. The full prompt files are now available, while complete run transcripts and all provider-level generation settings are not included.

Optional sequential Conditions 09 and 10 mentioned in five templates are outside this study. The unchanged prompt files are archival materials, not invitations to add those conditions retrospectively. New executions would be separate observations; this release makes no new AI calls.

The cited reference PDF and participant image archives are not included in these public numerical packages. The PDF can be obtained from its publisher; access to the private study images is a separate matter. The numerical reproduction does not require either.
