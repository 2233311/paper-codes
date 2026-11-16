"""Regressors used in the comparison experiments.

All learners expose ``fit(X, y)`` and ``predict(X)`` with X shaped ``(n, 25)``
or ``(n, 26)``. The sequence learners require exactly the 25 window features and
reshape them to five depth steps of five channels internally; a 26-column input
raises an error rather than dropping the extra column silently.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from . import config as cfg


class ScaledRegressor:
    """Wrap a regressor that requires a standardized target."""

    def __init__(self, base):
        self.base = base
        self.scaler = StandardScaler()

    def fit(self, x, y):
        self.base.fit(x, self.scaler.fit_transform(np.asarray(y).reshape(-1, 1)).ravel())
        return self

    def predict(self, x):
        return self.scaler.inverse_transform(
            self.base.predict(x).reshape(-1, 1)
        ).ravel()


def _to_sequence(x: np.ndarray) -> np.ndarray:
    if x.shape[1] != 25:
        raise ValueError(
            "sequence learners require exactly 25 window features, "
            f"got {x.shape[1]}"
        )
    n = x.shape[0]
    return np.transpose(x.reshape(n, 5, 5), (0, 2, 1)).astype(np.float32)


class TorchSequenceRegressor:
    """LSTM / RNN / GRU regressor over the five-sample depth window."""

    def __init__(self, kind: str, hidden: int = 32, epochs: int = 80,
                 patience: int = 10, batch_size: int = 256, seed: int = cfg.SEED):
        import torch
        import torch.nn as nn

        # Seed before building the network, otherwise the initial weights come
        # from the process-wide RNG state and differ between runs.
        torch.manual_seed(seed)
        self.torch = torch
        self.kind = kind.lower()
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.y_mean = 0.0
        self.y_std = 1.0

        recurr = {"lstm": nn.LSTM, "rnn": nn.RNN, "gru": nn.GRU}[self.kind]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.recurrent = recurr(5, hidden, num_layers=1, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, seq):
                out, _ = self.recurrent(seq)
                return self.head(out[:, -1]).squeeze(1)

        self.net = Net()

    def fit(self, x, y):
        torch = self.torch
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)
        seq = torch.from_numpy(_to_sequence(x))
        y = np.asarray(y, dtype=float)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-8)
        target = torch.from_numpy(((y - self.y_mean) / self.y_std).astype(np.float32))

        idx = rng.permutation(len(y))
        n_val = max(1, int(0.1 * len(y)))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        best_loss, best_state, stale = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            order = rng.permutation(train_idx)
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.net(seq[batch]), target[batch])
                loss.backward()
                optimizer.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net(seq[val_idx]), target[val_idx]))
            if val_loss < best_loss - 1e-7:
                best_loss, best_state, stale = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, x):
        torch = self.torch
        with torch.no_grad():
            out = self.net(torch.from_numpy(_to_sequence(x))).numpy()
        return out * self.y_std + self.y_mean


def make_learner(name: str, **overrides):
    """Build a regressor by short name. ``overrides`` replace defaults."""
    name = name.lower()
    if name == "xgb":
        params = {**cfg.XGB_PARAMS, **overrides}
        return XGBRegressor(**params)
    if name == "rf":
        params = dict(n_estimators=300, max_depth=18, min_samples_leaf=2,
                      random_state=cfg.SEED, n_jobs=4)
        params.update(overrides)
        return RandomForestRegressor(**params)
    if name == "mlp":
        params = dict(hidden_layer_sizes=(64, 32, 16), activation="relu",
                      solver="adam", alpha=1e-4, batch_size=128,
                      learning_rate_init=1e-3, max_iter=500,
                      early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=30, random_state=cfg.SEED)
        params.update(overrides)
        return MLPRegressor(**params)
    if name == "ann":
        params = dict(hidden_layer_sizes=(128, 64, 32), activation="relu",
                      solver="adam", alpha=1e-4, batch_size=128,
                      learning_rate_init=1e-3, max_iter=500,
                      early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=30, random_state=cfg.SEED)
        params.update(overrides)
        return MLPRegressor(**params)
    if name == "linear":
        return LinearRegression(**overrides)
    if name == "knn":
        params = dict(n_neighbors=10, weights="distance")
        params.update(overrides)
        return KNeighborsRegressor(**params)
    if name == "svr":
        params = dict(C=10.0, epsilon=0.1, gamma="scale")
        params.update(overrides)
        return ScaledRegressor(SVR(**params))
    if name in {"lstm", "rnn", "gru"}:
        return TorchSequenceRegressor(name, **overrides)
    raise ValueError(f"unknown learner: {name}")


LEARNER_NAMES = ["xgb", "rf", "mlp", "ann", "lstm", "rnn", "gru",
                 "linear", "knn", "svr"]
