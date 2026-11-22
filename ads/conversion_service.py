"""CPU HTTP inference for frozen seven-day post-impression conversion models."""
import os
from time import perf_counter
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, create_model

from ads.conversion_serving import FIELDS, Predictor, transform

Features = create_model('Features', __config__=ConfigDict(extra='forbid'),
                        **{key: (int, Field(ge=0, strict=True)) for key in FIELDS})


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: Literal['naive_recent', 'mature_only', 'delay_likelihood'] = 'delay_likelihood'
    records: list[Features] = Field(min_length=1, max_length=200)


def create_app():
    predictor = Predictor(os.environ.get('CONVERSION_ARTIFACT_DIR', 'models/conversion'))
    app = FastAPI(title='Seven-day conversion scoring')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'version': predictor.version, 'target': predictor.manifest['target']}

    @app.post('/score')
    def score(request: Request):
        start = perf_counter()
        matrix = transform([record.model_dump() for record in request.records])
        encoded = perf_counter()
        probabilities = predictor.predict(matrix, request.model)
        inferred = perf_counter()
        return {'model': request.model, 'version': predictor.version,
                'probabilities': probabilities.tolist(),
                'stage_ms': {'features': (encoded-start)*1000,
                             'inference': (inferred-encoded)*1000,
                             'total': (inferred-start)*1000}}

    return app
