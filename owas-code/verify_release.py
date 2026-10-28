"""Compare every released scientific CSV against freshly generated results.

Text and integers must match exactly. Non-integer numeric values may differ by
at most 1e-9 in absolute value, consistent with the original release validation.
No photographic source paths or private workbook coordinates are compared.
"""
from pathlib import Path
import csv
import json
import math
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'owas-data-results'
GENERATED = ROOT / 'generated' / 'results'
FLOAT = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+[eE][-+]?\d+|\d+\.\d*[eE][-+]?\d+)')
REMOVED_PRIVATE_FIELDS = {'source_files', 'excel_sheet', 'excel_row'}


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = [field for field in reader.fieldnames if field not in REMOVED_PRIVATE_FIELDS]
        rows = [{field: row[field] for field in fields} for row in reader]
    if path.name == 'prior_provenance_audit.csv':
        rows.sort(key=lambda row: (row['dataset'], row['model'], int(row['combination'])))
    return fields, rows


def main():
    expected = sorted((DATA / 'results').glob('*.csv'))
    assert expected, 'No frozen result CSVs found; extract the data ZIP next to owas-code.'
    checked = []
    cells = 0
    tolerated_float_differences = 0
    max_abs_difference = 0.0
    for source in expected:
        target = GENERATED / source.name
        assert target.is_file(), f'Missing generated result: {source.name}'
        source_fields, source_rows = read(source)
        target_fields, target_rows = read(target)
        assert set(source_fields) == set(target_fields), f'Column mismatch: {source.name}'
        assert len(source_rows) == len(target_rows), f'Row count mismatch: {source.name}'
        for number, (original, reproduced) in enumerate(zip(source_rows, target_rows), start=2):
            for field in source_fields:
                a, b = original[field], reproduced[field]
                cells += 1
                if a == b:
                    continue
                # Numeric IDs, counts and codes are not silently converted to floats.
                if FLOAT.fullmatch(a) or FLOAT.fullmatch(b):
                    try:
                        x, y = float(a), float(b)
                    except ValueError:
                        pass
                    else:
                        difference = abs(x - y)
                        if math.isfinite(x) and math.isfinite(y) and difference <= 1e-9:
                            tolerated_float_differences += 1
                            max_abs_difference = max(max_abs_difference, difference)
                            continue
                raise AssertionError(f'Mismatch in {source.name}, CSV row {number}, {field}: {a!r} != {b!r}')
        checked.append({'file': source.name, 'rows': len(source_rows), 'columns': len(source_fields)})
    settings = json.loads((GENERATED / 'resampling_settings.json').read_text())
    frozen_settings = json.loads((DATA / 'audit' / 'resampling_settings.json').read_text())
    assert settings == frozen_settings, 'Statistical resampling settings differ.'
    protocol_audit_files = []
    for name in ['protocol_metadata_flags.csv', 'protocol_metadata_normalization.csv']:
        source_fields, source_rows = read(DATA / 'audit' / name)
        target_fields, target_rows = read(ROOT / 'generated' / 'audit' / name)
        assert source_fields == target_fields and source_rows == target_rows, f'Protocol metadata audit mismatch: {name}'
        protocol_audit_files.append(name)
    name = 'protocol_metadata_QA.json'
    assert json.loads((DATA / 'audit' / name).read_text()) == json.loads((ROOT / 'generated' / 'audit' / name).read_text())
    protocol_audit_files.append(name)
    result = {
        'status': 'PASS',
        'scientific_csv_tables_compared': len(checked),
        'protocol_audit_files_compared': len(protocol_audit_files),
        'cells_compared': cells,
        'tolerated_float_differences': tolerated_float_differences,
        'maximum_absolute_float_difference': max_abs_difference,
        'tolerance': 'Exact text and integer equality; floating values within absolute 1e-9.',
        'scope': 'Fresh calculations from released deidentified numerical inputs versus all released scientific CSV results. Source-file paths, workbook coordinates and private raw archives are not part of this public reproduction.',
        'resampling_settings_equal': True,
        'tables': checked,
    }
    (GENERATED / 'reproduction_QA.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'tables'}, indent=2))


if __name__ == '__main__':
    main()
