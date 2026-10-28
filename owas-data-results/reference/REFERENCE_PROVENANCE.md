# OWAS action reference

The numerical code-to-action table was transcribed programmatically from Table 6 of Diego-Mas, J.A., *Método OWAS para la evaluación de la carga postural*, Ergonautas, Universitat Politècnica de València (2025), accessed 28 October 2026:
https://www.ergonautas.upv.es/metodos/owas/owas-ayuda.php

`owas_action_reference.csv` freezes all 252 entries (4 back × 3 arm × 7 leg × 3 load categories). The source HTML snapshot used for extraction had SHA-256 `a9e1ed169ee91a48de9177e9fbae70c2bcec99dcc21ba09e90aac0c3bbbde333`.

Only the 84 entries ending in load digit 1 are used in this study. They are compared with the generated `results/action_lookup_load1.csv`. The independent check also parses the full constant array from `owas_core.py` and compares it against all 252 frozen reference entries, without importing the lookup under test.

The v2 array differed at 3141 (2 instead of 3), 3151 (3 instead of 4), and three unused load-2/3 entries: 3133 (1 instead of 2), 3232 (2 instead of 1), 3233 (4 instead of 2). All five constants are corrected. Neither of the affected load-1 codes occurs in any released expert code, original complete AI code, embedded prior complete code or derived complete code, so the study's performance results are unchanged.
