"""Download unmodified official MediaPipe models and record exact hashes."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'pose_landmarker_full.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'mobilenet_v3_small.tflite': 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite',
}

if __name__ == '__main__':
    directory = ROOT / 'models'
    directory.mkdir(exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        path = directory / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(100 * 1024 * 1024)
            if not data or len(data) >= 100 * 1024 * 1024:
                raise ValueError('Unexpected model size')
            path.write_bytes(data)
        data = path.read_bytes()
        record = {'filename': name, 'url': url, 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        records.append(record)
        print(json.dumps(record), flush=True)
    (directory / 'model_assets.json').write_text(json.dumps(records, indent=2)+'\n')
