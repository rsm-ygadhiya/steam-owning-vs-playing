"""Task 3 (stage A, PyTorch): baselines, bias model, latent factors, factorization machine.

Predict y = log(1 + playtime_forever) for a (user, game) pair in a library.

Model ladder
  3.0 global / user / item mean
  3.1 bias model          y ~ a + b_u + b_i               (closed-form alternating updates)
  3.2 latent factor model y ~ a + b_u + b_i + p_u.q_i     (PyTorch, Adam)
  3.3 factorization machine on user, item and game metadata (PyTorch)
  3.4 LightGBM stacked on leak-free features + 3.1-3.3 predictions
  3.5 hurdle: P(played) * E[y | played]   (both stages LightGBM)
  3.6 ensemble: linear blend fit on out-of-fold validation predictions

Stage A fits 3.0-3.3 on TRAIN and caches their predictions on valid/test/cold.
Stage B (regression_stack.py, LightGBM) fits 3.4-3.6 on VALID.
They run in separate processes: on macOS, torch and LightGBM ship different
OpenMP runtimes that crash when loaded together.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from .config import ART, SEED
from . import features, metrics, splits

torch.set_num_threads(8)
RES = ART / "results"
RES.mkdir(exist_ok=True)


# ===================================================================== 3.0 / 3.1
class BiasModel:
    """y ~ alpha + b_u + b_i with L2 penalty lam, fit by exact coordinate updates."""

    def __init__(self, lam=5.0, n_iter=15):
        self.lam, self.n_iter = lam, n_iter

    def fit(self, u, i, y, nu, ni):
        self.bu, self.bi = np.zeros(nu), np.zeros(ni)
        for _ in range(self.n_iter):
            self.alpha = np.mean(y - self.bu[u] - self.bi[i])
            r = y - self.alpha - self.bi[i]
            self.bu = np.bincount(u, r, nu) / (self.lam + np.bincount(u, minlength=nu))
            r = y - self.alpha - self.bu[u]
            self.bi = np.bincount(i, r, ni) / (self.lam + np.bincount(i, minlength=ni))
        return self

    def predict(self, u, i):
        return self.alpha + self.bu[u] + self.bi[i]


# ========================================================================= 3.2
class MF(torch.nn.Module):
    def __init__(self, nu, ni, k):
        super().__init__()
        self.P = torch.nn.Embedding(nu, k)
        self.Q = torch.nn.Embedding(ni, k)
        self.bu = torch.nn.Embedding(nu, 1)
        self.bi = torch.nn.Embedding(ni, 1)
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        for e in (self.P, self.Q):
            torch.nn.init.normal_(e.weight, std=0.01)
        for e in (self.bu, self.bi):
            torch.nn.init.zeros_(e.weight)

    def forward(self, u, i):
        p, q = self.P(u), self.Q(i)
        pred = self.alpha + self.bu(u).squeeze(1) + self.bi(i).squeeze(1) + (p * q).sum(1)
        reg = (p ** 2).sum(1) + (q ** 2).sum(1) + self.bu(u).squeeze(1) ** 2 + self.bi(i).squeeze(1) ** 2
        return pred, reg

    def dot(self, u, i):
        return (self.P(u) * self.Q(i)).sum(1)


# ========================================================================= 3.3
class FM(torch.nn.Module):
    """Second-order factorization machine (Rendle 2010) over sparse feature lists.

    Each example is a padded list of (feature index, value). Fields: user id,
    user library-size bucket, item id, item genres, top tags, price / year /
    metascore buckets, free-to-play, early access.
    """

    def __init__(self, n_feat, k):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.zeros(1))
        self.w = torch.nn.Embedding(n_feat, 1)
        self.V = torch.nn.Embedding(n_feat, k)
        torch.nn.init.zeros_(self.w.weight)
        torch.nn.init.normal_(self.V.weight, std=0.01)

    def forward(self, idx, val):
        v = self.V(idx) * val.unsqueeze(-1)               # (B, L, k)
        linear = (self.w(idx).squeeze(-1) * val).sum(1)
        inter = 0.5 * ((v.sum(1) ** 2) - (v ** 2).sum(1)).sum(1)
        reg = (v ** 2).sum((1, 2))
        return self.w0 + linear + inter, reg


class FMEncoder:
    """Builds per-user and per-item feature lists once; batches gather them."""

    def __init__(self, nu, ni, train_items):
        m = features.item_meta()
        genres, tags = features.vocab()
        lib = features.train_stats()[0].library_size.values
        off = 0
        self.user_id = np.arange(nu) + off; off += nu
        lib_b = np.digitize(np.log1p(lib), np.linspace(0, 9, 12))
        self.user_lib = lib_b + off; off += 13
        self.item_id = np.arange(ni) + off; off += ni
        g_off = off; off += len(genres)
        t_off = off; off += len(tags)
        price_b = np.digitize(m.price.fillna(-1).values, [-0.5, 0.01, 5, 10, 15, 20, 30, 60])
        p_off = off; off += 10
        year_b = np.digitize(m.release_year.fillna(0).values, [1, 2000, 2005, 2008, 2010, 2012, 2014, 2016, 2017])
        y_off = off; off += 11
        meta_b = np.digitize(m.metascore.fillna(0).values, [1, 60, 70, 80, 90])
        s_off = off; off += 7
        free_f, ea_f = off, off + 1; off += 2
        self.n_feat = off

        gi = {g: k for k, g in enumerate(genres)}
        ti = {t: k for k, t in enumerate(tags)}
        L = 3 + 6 + 8 + 5           # id + price/year/meta/free/ea + genres + tags
        idx = np.zeros((ni, L), np.int64)
        val = np.zeros((ni, L), np.float32)
        seen = np.zeros(ni, bool)
        seen[train_items] = True
        for i in range(ni):
            row = [(self.item_id[i], 1.0 if seen[i] else 0.0)]   # unseen (cold) id -> masked out
            if m.has_meta.values[i]:
                row += [(p_off + price_b[i], 1.0), (y_off + year_b[i], 1.0), (s_off + meta_b[i], 1.0)]
                if m.free.values[i] == 1:
                    row.append((free_f, 1.0))
                if m.early_access.values[i] == 1:
                    row.append((ea_f, 1.0))
                gs = [g_off + gi[g] for g in m.genres.values[i]][:8]
                row += [(g, 1.0 / np.sqrt(len(gs))) for g in gs]
                ts = [t_off + ti[t] for t in m.tags.values[i] if t in ti][:5]
                row += [(t, 1.0 / np.sqrt(len(ts))) for t in ts]
            row = row[:L]
            idx[i, :len(row)] = [r[0] for r in row]
            val[i, :len(row)] = [r[1] for r in row]
        self.item_idx, self.item_val = torch.from_numpy(idx), torch.from_numpy(val)
        self.u_idx = torch.from_numpy(np.stack([self.user_id, self.user_lib], 1).astype(np.int64))

    def batch(self, u, i):
        idx = torch.cat([self.u_idx[u], self.item_idx[i]], 1)
        val = torch.cat([torch.ones(len(u), 2), self.item_val[i]], 1)
        return idx, val


# ============================================================ torch training
def _train_torch(model, fwd, tr, va, lam, epochs, lr=0.01, bs=32768, log=None):
    """Generic minibatch Adam loop with L2 on the rows touched by each batch.
    Keeps the epoch with the best validation MSE (early stopping)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ut, it = torch.from_numpy(tr.u.values.astype(np.int64)), torch.from_numpy(tr.i.values.astype(np.int64))
    yt = torch.from_numpy(tr.y.values.astype(np.float32))
    g = torch.Generator().manual_seed(SEED)
    best, best_state, hist = np.inf, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), generator=g)
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            pred, reg = fwd(ut[b], it[b])
            loss = ((pred - yt[b]) ** 2).mean() + lam * reg.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += ((pred.detach() - yt[b]) ** 2).sum().item()
        v = metrics.mse(va.y.values, _predict_torch(fwd, va))
        hist.append({"epoch": ep + 1, "train_mse": tot / len(yt), "valid_mse": v})
        if log:
            log(f"  epoch {ep + 1}: train {tot / len(yt):.4f}  valid {v:.4f}")
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def _predict_torch(fwd, df, bs=262144):
    u = torch.from_numpy(df.u.values.astype(np.int64))
    i = torch.from_numpy(df.i.values.astype(np.int64))
    out = [fwd(u[s:s + bs], i[s:s + bs])[0] for s in range(0, len(u), bs)]
    return torch.cat(out).numpy()


def fit_mf(tr, va, nu, ni, k, lam, epochs=12, log=None):
    torch.manual_seed(SEED)
    m = MF(nu, ni, k)
    m.alpha.data.fill_(float(tr.y.mean()))
    hist = _train_torch(m, m.forward, tr, va, lam, epochs, log=log)
    return m, hist


def fit_fm(tr, va, enc, k, lam, epochs=10, log=None):
    torch.manual_seed(SEED)
    m = FM(enc.n_feat, k)
    m.w0.data.fill_(float(tr.y.mean()))
    fwd = lambda u, i: m(*enc.batch(u, i))
    hist = _train_torch(m, fwd, tr, va, lam, epochs, lr=0.005, log=log)
    return m, fwd, hist


# ======================================================================= main
def run(log=print, quick=False):
    t0 = time.time()
    st = splits.stats()
    nu, ni = st["n_users"], st["n_items"]
    tr, va, te, co = (splits.load(n) for n in ["train", "valid", "test", "cold"])
    if quick:
        tr = tr.sample(400_000, random_state=SEED)
    sets = {"valid": va, "test": te, "cold": co}
    preds = {name: {} for name in sets}
    tuning = {}

    # ---- 3.0 baselines
    ustats, istats, _, _ = features.train_stats()
    gmean = float(tr.y.mean())
    for name, d in sets.items():
        preds[name]["Global mean"] = np.full(len(d), gmean)
        preds[name]["User mean"] = ustats.mean_y.fillna(gmean).values[d.u.values]
        preds[name]["Item mean"] = istats.mean_y.fillna(gmean).values[d.i.values]
    log(f"[3.0] baselines done ({time.time() - t0:.0f}s)")

    # ---- 3.1 bias model, tune lambda on valid
    best = (np.inf, None)
    tuning["bias"] = []
    for lam in [0.5, 1, 2, 5, 10, 20, 50]:
        bm = BiasModel(lam).fit(tr.u.values, tr.i.values, tr.y.values, nu, ni)
        v = metrics.mse(va.y.values, bm.predict(va.u.values, va.i.values))
        tuning["bias"].append({"lambda": lam, "valid_mse": v})
        if v < best[0]:
            best = (v, bm)
    bm = best[1]
    for name, d in sets.items():
        preds[name]["Bias model"] = bm.predict(d.u.values, d.i.values)
    log(f"[3.1] bias model lambda={bm.lam} valid MSE={best[0]:.4f} ({time.time() - t0:.0f}s)")

    # ---- 3.2 latent factor model, grid over K x lambda
    tuning["mf"] = []
    # The loss is MSE (~4) + lam * mean squared-norm of the touched rows, so lam <= 1e-3 is
    # ~1/4000 of the loss and does nothing (valid MSE barely moved, MF overfit after epoch 2).
    # Probing showed the useful range is ~0.03-0.2; lam >= 0.3 underfits.
    grid = [(8, 0.1), (32, 0.1)] if quick else [(k, l) for k in [32, 64, 128] for l in [1e-3, 0.03, 0.1, 0.15, 0.2]]
    best = (np.inf, None, None)
    for k, lam in grid:
        m, hist = fit_mf(tr, va, nu, ni, k, lam, epochs=3 if quick else 12)
        v = min(h["valid_mse"] for h in hist)
        tuning["mf"].append({"K": k, "lambda": lam, "valid_mse": v})
        log(f"[3.2] MF K={k} lambda={lam:g}: valid MSE {v:.4f} ({time.time() - t0:.0f}s)")
        if v < best[0]:
            best = (v, m, hist)
    mf, tuning["mf_curve"] = best[1], best[2]
    mf.eval()
    for name, d in sets.items():
        preds[name]["Latent factor (MF)"] = _predict_torch(mf.forward, d)
        with torch.no_grad():
            preds[name]["_mf_dot"] = np.concatenate([
                mf.dot(torch.from_numpy(d.u.values[s:s + 262144].astype(np.int64)),
                       torch.from_numpy(d.i.values[s:s + 262144].astype(np.int64))).numpy()
                for s in range(0, len(d), 262144)])
    # cold games have untrained item factors: MF falls back to its user bias + alpha
    with torch.no_grad():
        cu = torch.from_numpy(co.u.values.astype(np.int64))
        preds["cold"]["Latent factor (MF)"] = (mf.alpha + mf.bu(cu).squeeze(1)).numpy()
        preds["cold"]["_mf_dot"] = np.full(len(co), np.nan)
    np.save(ART / "mf_item_factors.npy", mf.Q.weight.detach().numpy())

    # ---- 3.3 factorization machine
    enc = FMEncoder(nu, ni, np.unique(tr.i.values))
    tuning["fm"] = []
    best = (np.inf, None, None, None)
    # Same scale problem as MF: lam=1e-4 is negligible next to an MSE of ~4. The FM penalty sums
    # over ~20 active feature vectors per row (vs 2 for MF), so its useful lam range is ~10x smaller.
    fm_grid = [(16, 0.01)] if quick else [(32, 1e-4), (32, 0.01), (64, 0.03), (64, 0.1), (64, 0.2), (128, 0.1)]
    for k, lam in fm_grid:
        m, fwd, hist = fit_fm(tr, va, enc, k, lam, epochs=3 if quick else 12)
        v = min(h["valid_mse"] for h in hist)
        tuning["fm"].append({"K": k, "lambda": lam, "valid_mse": v})
        log(f"[3.3] FM K={k} lambda={lam:g}: valid MSE {v:.4f} ({time.time() - t0:.0f}s)")
        if v < best[0]:
            best = (v, m, fwd, hist)
    fm, fm_fwd, tuning["fm_curve"] = best[1], best[2], best[3]
    fm.eval()
    for name, d in sets.items():
        preds[name]["Factorization machine"] = _predict_torch(fm_fwd, d)

    for name in sets:
        pd.DataFrame(preds[name]).astype(np.float32).to_parquet(ART / f"reg_base_{name}.parquet")
    with open(RES / "regression_cf.json", "w") as f:
        json.dump(tuning, f, indent=1)
    log(f"[3A] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys
    run(quick="--quick" in sys.argv)
