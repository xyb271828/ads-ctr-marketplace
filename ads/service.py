"""Contextual auction demo using a model fitted on synthetic sponsor impressions."""
from time import perf_counter_ns

from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ads.auction import Bid, rank


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str = Field(min_length=1, max_length=80)
    grade: int = Field(ge=1, le=12)
    min_relevance: float = Field(default=.5, ge=0, le=1)


def create_app():
    rng = np.random.default_rng(42)
    x = np.column_stack([rng.integers(0, 2, 12000), rng.integers(0, 6, 12000)])
    labels = rng.random(len(x)) < (.04+.08*x[:, 0]-.005*x[:, 1])
    lr = LogisticRegression().fit(x[:9000], labels[:9000])
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        lr.decision_function(x[9000:]), labels[9000:]
    )
    catalog = [("a1", "fractions", 5, 1.3), ("a2", "fractions", 4, 1.5),
               ("a3", "geometry", 5, 1.2), ("a4", "reading", 5, 1.1)]
    app = FastAPI(title="Contextual ads")

    @app.get("/health")
    def health():
        return {"ready": True}

    @app.post("/auction")
    def auction(context: Context):
        start = perf_counter_ns()
        topic = context.topic.strip().casefold()
        relevant = [(advertiser, bid, abs(grade-context.grade)) for advertiser, t, grade, bid in catalog
                    if t == topic and 1/(1+abs(grade-context.grade)) >= context.min_relevance]
        sponsored = None
        if relevant:
            probs = calibrator.predict(lr.decision_function([[1, gap] for _, _, gap in relevant]))
            ranked = rank([Bid(i, b, float(p), b) for (i, b, _), p in zip(relevant, probs, strict=True)],
                          reserve=.25)
            if ranked:
                sponsored = {"id": ranked[0].advertiser, "cpc": ranked[0].cpc, "label": "Sponsored"}
        return {"sponsored": sponsored, "model_source": "synthetic_context",
                "stage_ms": {"pctr_auction": (perf_counter_ns()-start)/1e6}}

    return app
