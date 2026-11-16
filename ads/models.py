"""Tree leaf encoding plus logistic regression, and a CPU DeepFM."""
from time import perf_counter

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression


def train_gbdt_lr(x, y, test, seed, cap=200_000, cut=None):
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    if cut is None or not 0 < cut < len(y):
        cut = int(.7 * len(y))
    a = np.sort(rng.choice(cut, min(cut, cap), replace=False))
    b = cut + np.sort(rng.choice(len(y)-cut, min(len(y)-cut, cap), replace=False))
    tree = lgb.LGBMClassifier(n_estimators=48, num_leaves=31, n_jobs=2,
                              random_state=seed, verbosity=-1)
    hasher = FeatureHasher(n_features=2**16, input_type="string")

    def leaves(values):
        ids = tree.booster_.predict(values, pred_leaf=True, num_threads=2)
        return hasher.transform((f"{i}:{leaf}" for i, leaf in enumerate(row)) for row in ids)

    start = perf_counter()
    tree.fit(x[a], y[a], categorical_feature=list(range(13, 39)))
    lr = LogisticRegression(max_iter=150).fit(leaves(x[b]), y[b])
    elapsed = perf_counter() - start
    start = perf_counter()
    scores = np.concatenate([lr.predict_proba(leaves(test[i:i+8192]))[:, 1]
                             for i in range(0, len(test), 8192)])
    return scores, {"fit_rows_tree": len(a), "fit_rows_leaf_lr": len(b),
                    "fit_rows_per_second": (len(a)+len(b))/elapsed,
                    "test_scoring_ms": (perf_counter()-start)*1000,
                    "model_bytes": len(tree.booster_.model_to_string().encode())
                    + lr.coef_.nbytes + lr.intercept_.nbytes}


def train_deepfm(train, valid, test, seed, epochs):
    import torch
    from torch import nn

    class DeepFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(2**20, 8, sparse=True)
            self.linear = nn.Embedding(2**20, 1, sparse=True)
            self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(39*8, 64), nn.ReLU(),
                                     nn.Linear(64, 1))
            nn.init.normal_(self.emb.weight, std=.01)
            nn.init.zeros_(self.linear.weight)

        def forward(self, ids, weights):
            emb = self.emb(ids) * weights.unsqueeze(-1)
            fm = .5 * (emb.sum(1).square() - emb.square().sum(1)).sum(1)
            return (self.linear(ids).squeeze(-1)*weights).sum(1) + fm + self.mlp(emb).squeeze(-1)

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    model = DeepFM()
    sparse_opt = torch.optim.SGD([model.emb.weight, model.linear.weight], lr=.1)
    dense_opt = torch.optim.AdamW(model.mlp.parameters(), lr=.002)
    objective = nn.BCEWithLogitsLoss()

    def forward(data, rows):
        return model(torch.from_numpy(data[0][rows]), torch.from_numpy(data[1][rows]))

    def logits(data):
        model.eval()
        with torch.no_grad():
            return np.concatenate([forward(data, slice(i, i+4096)).numpy()
                                   for i in range(0, len(data[2]), 4096)])

    started = perf_counter()
    best_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        indices = rng.permutation(len(train[2]))
        for i in range(0, len(indices), 1024):
            rows = indices[i:i+1024]
            sparse_opt.zero_grad()
            dense_opt.zero_grad()
            loss = objective(forward(train, rows), torch.from_numpy(train[2][rows]).float())
            loss.backward()
            sparse_opt.step()
            dense_opt.step()
        scores = logits(valid)
        loss = float(np.mean(np.logaddexp(0, scores) - valid[2]*scores))
        if loss < best_loss:
            best_loss = loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
    elapsed = perf_counter() - started
    model.load_state_dict(best_state)
    start = perf_counter()
    scores = expit(logits(test))
    return scores, {"fit_rows_per_second": epochs*len(train[2])/elapsed,
                    "test_scoring_ms": (perf_counter()-start)*1000, "epochs": epochs,
                    "model_bytes": sum(x.numel()*x.element_size() for x in best_state.values())}
