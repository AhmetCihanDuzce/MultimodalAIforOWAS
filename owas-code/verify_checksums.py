"""Verify both release payload manifests using the Python standard library."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent

def main():
    checked=0
    for base in [ROOT, ROOT.parent/'owas-data-results']:
        manifest=base/'MANIFEST.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'Missing {manifest}. Extract both complete release ZIPs into the same parent directory.')
        payload=json.loads(manifest.read_text(encoding='utf-8'))
        assert payload['manifest_version']==1
        for row in payload['files']:
            path=Path(row['path'])
            assert not path.is_absolute() and '..' not in path.parts, path
            f=base/path
            if not f.is_file():raise FileNotFoundError(f)
            raw=f.read_bytes()
            if len(raw)!=row['size_bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                raise AssertionError(f'Checksum mismatch: {f}')
            checked+=1
        print(f'PASS: {base.name}: {len(payload["files"])} payload files')
    print(f'PASS: {checked} payload file checksums. Manifests exclude themselves and SHA256SUMS.txt; these are integrity checks, not digital signatures.')

if __name__=='__main__':main()
