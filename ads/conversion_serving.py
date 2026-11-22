"""Versioned NumPy artifacts and the shared conversion feature encoder."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction import FeatureHasher

FIELDS = ('campaign', *[f'cat{i}' for i in range(1, 10)])
MODELS = ('naive_recent', 'mature_only', 'delay_likelihood')
BUCKETS = 2**16


def transform(records):
    hasher = FeatureHasher(n_features=BUCKETS, input_type='string', alternate_sign=False)
    return hasher.transform((f'{field}={record[field]}' for field in FIELDS) for record in records)


def export(directory, parameters, provenance, records, reference):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    array = np.stack([parameters[name] for name in MODELS])
    if array.shape != (len(MODELS), BUCKETS + 1) or not np.isfinite(array).all():
        raise ValueError('Invalid coefficient matrix')
    path = directory / 'weights.npy'
    np.save(path, array, allow_pickle=False)
    manifest = {'format': 1, 'fields': list(FIELDS), 'models': list(MODELS),
                'buckets': BUCKETS, 'alternate_sign': False,
                'weights_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **provenance}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    predictor = Predictor(directory)
    matrix = transform(records)
    parity = {}
    for name in MODELS:
        actual = predictor.predict(matrix, name)
        np.testing.assert_allclose(actual, reference[name], rtol=1e-10, atol=1e-12)
        parity[name] = float(np.max(np.abs(actual-reference[name])))
    (directory / 'parity.json').write_text(json.dumps({'rows':len(records),
                                                      'max_absolute_error':parity},indent=2)+'\n')


class Predictor:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / 'manifest.json').read_text())
        path = directory / 'weights.npy'
        if (self.manifest['format'] != 1 or self.manifest['fields'] != list(FIELDS)
                or self.manifest['models'] != list(MODELS) or self.manifest['buckets'] != BUCKETS
                or self.manifest['alternate_sign'] is not False
                or hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest['weights_sha256']):
            raise ValueError('Feature schema or artifact checksum mismatch')
        self.weights = np.load(path, allow_pickle=False)
        if self.weights.shape != (len(MODELS), BUCKETS+1) or not np.isfinite(self.weights).all():
            raise ValueError('Invalid model weights')
        self.version = self.manifest['weights_sha256'][:16]

    def predict(self, matrix, name):
        weights = self.weights[MODELS.index(name)]
        return expit(matrix @ weights[:-1] + weights[-1])
