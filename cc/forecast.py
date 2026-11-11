"""Forecasters: seasonal naive, ridge (ARIMAX-type linear), and an LSTM
encoder with known-future covariates (the deep learning layer)."""
import numpy as np
import torch, torch.nn as nn
from sklearn.linear_model import Ridge

H, WIN = 7, 28

def dem_series(w):
    n, C, P = w['dem'].shape
    y = w['dem'].reshape(n, C * P)
    promo = w['promo'].reshape(n, C * P)
    return y, promo

def _calendar(n):
    d = np.arange(n); return np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)

def build_xy(y, promo, t_amb, t_fc, origins, scale):
    n, M = y.shape; ds, dc = _calendar(n)
    ys = y / scale
    past, fut, tgt, sid = [], [], [], []
    for d in origins:
        for i in range(M):
            sl = slice(d - WIN, d)
            past.append(np.stack([ys[sl, i], ds[sl], dc[sl], promo[sl, i], t_amb[sl] / 20], 1))
            fs = slice(d, d + H)
            fut.append(np.concatenate([promo[fs, i], ds[fs], dc[fs], t_fc[d] / 20]))
            tgt.append(ys[fs, i] if d + H <= n else np.zeros(H)); sid.append(i)
    oh = np.eye(M)[sid]
    return (np.array(past, np.float32), np.array(fut, np.float32), oh.astype(np.float32),
            np.array(tgt, np.float32))

class Net(nn.Module):
    def __init__(self, M, nfeat=5, nfut=4 * H, hid=32):
        super().__init__()
        self.lstm = nn.LSTM(nfeat, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + M + nfut, 64), nn.ReLU(), nn.Linear(64, H))
    def forward(self, past, fut, oh):
        _, (h, _) = self.lstm(past)
        return self.head(torch.cat([h[-1], oh, fut], 1))

class LSTMForecaster:
    def __init__(self, epochs=40, seed=0):
        self.epochs, self.seed = epochs, seed
    def fit(self, y, promo, t_amb, t_fc, train_end):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.scale = y[:train_end].mean(0) + 1e-9
        orig = np.arange(WIN, train_end - H + 1)
        P_, F_, O_, T_ = build_xy(y, promo, t_amb, t_fc, orig, self.scale)
        self.net = Net(y.shape[1]); opt = torch.optim.Adam(self.net.parameters(), 3e-3)
        P_, F_, O_, T_ = map(torch.tensor, (P_, F_, O_, T_))
        n = len(T_)
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for b in range(0, n, 256):
                ix = perm[b:b + 256]
                loss = ((self.net(P_[ix], F_[ix], O_[ix]) - T_[ix]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self
    def predict(self, y, promo, t_amb, t_fc, origins):
        P_, F_, O_, _ = build_xy(y, promo, t_amb, t_fc, origins, self.scale)
        with torch.no_grad():
            out = self.net(*map(torch.tensor, (P_, F_, O_))).numpy()
        M = y.shape[1]
        return np.maximum(out.reshape(len(origins), M, H).transpose(0, 2, 1) * self.scale, 0)  # [orig, h, M]

class RidgeForecaster:
    def fit(self, y, promo, t_amb, t_fc, train_end):
        self.scale = y[:train_end].mean(0) + 1e-9
        X, T = self._feat(y, promo, t_fc, np.arange(WIN, train_end - H + 1), True)
        self.m = [Ridge(1.0).fit(X[h], T[:, h]) for h in range(H)]
        return self
    def _feat(self, y, promo, t_fc, origins, with_t=False):
        ys = y / self.scale; n, M = y.shape
        Xh = [[] for _ in range(H)]; T = []
        for d in origins:
            for i in range(M):
                lags = ys[d - 7:d, i]; same = [ys[d - 14 + h % 7, i] for h in range(H)]
                for h in range(H):
                    dow = np.eye(7)[(d + h) % 7]
                    Xh[h].append(np.concatenate([lags, [same[h], ys[d + h - 7, i], promo[d + h, i], t_fc[d, h] / 20],
                                                 dow, np.eye(M)[i]]))
                if with_t: T.append(ys[d:d + H, i])
        return [np.array(x) for x in Xh], np.array(T)
    def predict(self, y, promo, t_amb, t_fc, origins):
        X, _ = self._feat(y, promo, t_fc, origins); M = y.shape[1]
        out = np.stack([self.m[h].predict(X[h]) for h in range(H)], 1)   # [orig*M, H]
        return np.maximum(out.reshape(len(origins), M, H).transpose(0, 2, 1) * self.scale, 0)

def snaive(y, origins):
    return np.stack([np.stack([y[d + h - 7] for h in range(H)]) for d in origins])   # [orig, h, M]
