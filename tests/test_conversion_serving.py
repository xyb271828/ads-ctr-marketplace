"""Artifact integrity, feature parity and HTTP schema regression checks."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from ads.conversion_service import create_app
from ads.conversion_serving import BUCKETS, FIELDS, MODELS, Predictor, export, transform


class ServingTest(unittest.TestCase):
    def test_export_http_and_integrity(self):
        record = {key: index+1 for index, key in enumerate(FIELDS)}
        matrix = transform([record])
        np.testing.assert_array_equal(matrix.toarray(), transform([dict(reversed(list(record.items())))]).toarray())
        parameters = {model: np.zeros(BUCKETS+1) for model in MODELS}
        with tempfile.TemporaryDirectory() as directory:
            export(directory, parameters, {'target': 'test'}, [record], {model: [.5] for model in MODELS})
            old = os.environ.get('CONVERSION_ARTIFACT_DIR')
            os.environ['CONVERSION_ARTIFACT_DIR'] = directory
            try:
                with TestClient(create_app()) as client:
                    for model in MODELS:
                        result = client.post('/score', json={'model':model, 'records':[record]})
                        self.assertEqual(result.status_code, 200)
                        self.assertEqual(result.json()['probabilities'], [.5])
                    for payload in ({'records':[]}, {'records':[record]*201},
                                    {'records':[{**record,'campaign':-1}]},
                                    {'records':[{**record,'campaign':'1'}]},
                                    {'records':[record],'model':'unknown'}):
                        self.assertEqual(client.post('/score', json=payload).status_code, 422)
            finally:
                if old is None:
                    os.environ.pop('CONVERSION_ARTIFACT_DIR', None)
                else:
                    os.environ['CONVERSION_ARTIFACT_DIR'] = old
            path = Path(directory)/'weights.npy'
            path.write_bytes(path.read_bytes()+b'corruption')
            with self.assertRaises(ValueError):
                Predictor(directory)


if __name__ == '__main__':
    unittest.main()
