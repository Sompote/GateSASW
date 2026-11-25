"""Apply the trained estimator to the real foundations.
usage: python apply.py model_dir real_inputs.json out.json"""
import sys, json, time
import numpy as np, torch
sys.argv_backup = sys.argv
mdir, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
ck = torch.load(f"{mdir}/model.pt", map_location="cpu", weights_only=False)
G, K, L = ck["G"], ck["K"], ck["nout"]; VMU, VSD = ck["vmu"], ck["vsd"]

import torch.nn as nn
class Block(nn.Module):
    def __init__(s, w):
        super().__init__(); s.f = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, w))
    def forward(s, x): return x + s.f(x)
class Net(nn.Module):
    def __init__(s, nin, nout, w=768, depth=8):
        super().__init__()
        s.inp = nn.Linear(nin, w); s.blocks = nn.Sequential(*[Block(w) for _ in range(depth)])
        s.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 2 * nout)); s.nout = nout
    def forward(s, x):
        o = s.out(s.blocks(s.inp(x)))
        return o[:, :s.nout] * VSD + VMU, o[:, s.nout:].clamp(-6, 3)
net = Net(ck["nin"], L); net.load_state_dict(ck["state"]); net.eval()

R = json.load(open(inp)); sites = sorted(R)
def feats(r, use_kpt):
    lv = np.array(r["lnv"], float); mk = np.array(r["mask"], bool); sg = np.array(r["sig"], float)
    xd = np.where(mk, (lv - VMU) / VSD, 0.0); xs = np.where(mk, np.clip(sg, 0.05, 0.4) * 5.0, 0.0)
    ky = np.array(r["kpt"], float); km = np.array(r["kpt_mask"], bool) & use_kpt
    xk = np.where(km, (ky - VMU) / VSD, 0.0)
    return np.concatenate([xd, mk.astype(float), xs, xk, km.astype(float)])
res = {}
for use in (True, False):
    X = torch.tensor(np.array([feats(R[s], use) for s in sites]), dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad(): mu, ls = net(X)
    dt = (time.perf_counter() - t0) / len(sites)
    for i, s in enumerate(sites):
        res.setdefault(s, {})["with_kpt" if use else "sasw_only"] = dict(m=mu[i].tolist(), sd=ls[i].exp().tolist())
    print(("with KPT" if use else "SASW only"), f"inference {dt*1000:.2f} ms per foundation on CPU")
json.dump(dict(thick=ck["thick"].tolist(), sites=res, ms_per_site=dt * 1000), open(out, "w"))
print("saved", out, len(sites), "sites")
