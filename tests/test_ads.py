import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from ads.auction import Bid, rank
from ads.calibration import correct_negative_sampling, pointwise_loss
from ads.ctr import benchmark
from ads.data import fixture, read_rows, prepare
from ads.marketplace import replay
from ads.service import create_app


class AdsTests(unittest.TestCase):
    def test_second_price_and_uniform_scale(self):
        bids = [Bid("a", 2, .1, 3), Bid("b", 1, .15, 3), Bid("empty", 9, .9, .05)]
        winners = rank(bids, .25, 2)
        self.assertEqual([w.advertiser for w in winners], ["a", "b"])
        self.assertAlmostEqual(winners[0].cpc, 1.5)
        self.assertEqual(winners[1].cpc, .25)
        shifted = rank([replace(b, pctr=1.1*b.pctr) for b in bids], .25)
        self.assertAlmostEqual(winners[0].cpc, shifted[0].cpc)

    def test_remaining_budget_is_binding(self):
        winner = rank([Bid("a", 10, .4, .3), Bid("b", .35, .3, 5)], .25)[0]
        self.assertEqual(winner.advertiser, "a")
        self.assertLessEqual(winner.cpc, .3)

    def test_replay_uniform_scale(self):
        for paced in (True, False):
            control = replay(42, "calibrated_oracle", paced, 3000)
            shifted = replay(42, "global_10", paced, 3000)
            self.assertEqual(control["exposure"], shifted["exposure"])
            self.assertEqual(control["clicks"], shifted["clicks"])
            self.assertAlmostEqual(control["revenue"], shifted["revenue"])

    def test_negative_sampling_prior(self):
        np.testing.assert_allclose(correct_negative_sampling(np.array([0., .5, 1.]), .1),
                                   [0, 1/11, 1])

    def test_float32_probability_endpoints(self):
        values = pointwise_loss(np.array([0, 1]), np.array([1., 0.], np.float32))
        self.assertTrue(np.isfinite(values).all())

    def test_ordered_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "source.tsv"
            source.write_text("".join("\t".join([str(i % 2), str(i)] + ["0"]*38)+"\n" for i in range(600)))
            out = path / "split"
            result = prepare(source, out, 400)
            for split, window in result["windows"].items():
                ids = [int(row[1]) for row in read_rows(out / f"{split}.tsv")]
                self.assertEqual(sorted(set(ids)), ids)
                self.assertEqual(len(ids), window["sample_rows"])
                self.assertTrue(all(window["start_row_zero_based"] <= i < window["end_row_exclusive"]
                                    for i in ids))

    def test_end_to_end_ctr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path / "input.tsv", 2400)
            result = benchmark(path / "input.tsv", path / "output", rows=2400)
            self.assertAlmostEqual(result["models"]["background"]["ne"], 1.)
            self.assertIsNotNone(result["models"]["hashed_lr"]["auc"])
            self.assertIsNotNone(result["models"]["platt"]["paired_vs_lr"])

    def test_auction_endpoint_context_only(self):
        client = TestClient(create_app())
        self.assertEqual(client.post("/auction", json={"topic":"fractions","grade":5})
                         .json()["sponsored"]["label"], "Sponsored")
        self.assertIsNone(client.post("/auction", json={"topic":"mismatch","grade":5})
                          .json()["sponsored"])
        self.assertEqual(client.post("/auction", json={"topic":"fractions","grade":5,"user_id":1})
                         .status_code, 422)
