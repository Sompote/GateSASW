"""Neural fusion estimator: trains a network that maps (dispersion curve,
KPT sounding) to the posterior mean and standard deviation of ln Vs per layer.

The observation model is applied on the GPU at every batch:
  dispersion : wavelength coverage sampled from the real sites, multiplicative
               noise with the relative uncertainty of real composite curves
  KPT        : y_k = ln Vs(z_k) + b + e_k, b ~ N(0, tau^2), e ~ N(0, sigma_w^2),
               admissible only for 147 <= Vs_KPT <= 340 m/s, stops at refusal,
               absent in a share of the examples (so the same network also
               serves foundations with no sounding)
usage: python train.py data.npz real_stats.json out_dir [steps] [batch]"""
import sys, json, time, math
import numpy as np
import torch, torch.nn as nn

data, stats_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

D = np.load(data); ST = json.load(open(stats_path))
M = torch.tensor(D["m"]); V = torch.tensor(D["lnv"]); LAM = torch.tensor(D["lam"], dtype=torch.float32)
THICK = D["thick"]; ZTOP = np.concatenate([[0], np.cumsum(THICK)])
N, L = M.shape; G = V.shape[1]
KZ = np.arange(0.1, 5.6, 0.2); K = len(KZ)                       # 28 KPT increments
KL = torch.tensor(np.minimum(np.searchsorted(ZTOP, KZ, side="right") - 1, L - 1))
TAU, SW = ST["tau"], ST["sigma_w"]
LO, HI = math.log(147.0), math.log(340.0)
real_lmin = torch.tensor(ST["lam_min"], dtype=torch.float32); real_lmax = torch.tensor(ST["lam_max"], dtype=torch.float32)
real_sig = torch.tensor(ST["sig_rel"], dtype=torch.float32)
nval = min(20000, N // 10); perm = torch.randperm(N)
vi, ti = perm[:nval], perm[nval:]
M, V, LAM, KL = M.to(dev), V.to(dev), LAM.to(dev), KL.to(dev)
real_lmin, real_lmax, real_sig = real_lmin.to(dev), real_lmax.to(dev), real_sig.to(dev)
VMU, VSD = 5.4, 0.6


def observe(idx, gen, p_none=0.35, kpt="random"):
    """Build network inputs from clean simulations."""
    m, v = M[idx], V[idx]; B = len(idx)
    j = torch.randint(0, len(real_lmin), (B,), device=dev, generator=gen)
    lmin = real_lmin[j] * torch.exp(0.15 * torch.randn(B, device=dev, generator=gen))
    lmax = real_lmax[j] * torch.exp(0.15 * torch.randn(B, device=dev, generator=gen))
    sig = real_sig[torch.randint(0, len(real_sig), (B,), device=dev, generator=gen)].clamp(0.05, 0.4)
    mask = (LAM[None] >= lmin[:, None]) & (LAM[None] <= lmax[:, None]) & torch.isfinite(v)
    noisy = torch.nan_to_num(v) + sig[:, None] * torch.randn(B, G, device=dev, generator=gen)
    xd = torch.where(mask, (noisy - VMU) / VSD, torch.zeros_like(noisy))
    xs = torch.where(mask, sig[:, None].expand(B, G) * 5.0, torch.zeros_like(noisy))
    # KPT
    y = m[:, KL] + TAU * torch.randn(B, 1, device=dev, generator=gen) + SW * torch.randn(B, K, device=dev, generator=gen)
    refused = torch.cumsum((y > HI).float(), 1) > 0                     # driving stops at refusal
    zmax = 1.0 + 4.6 * torch.rand(B, 1, device=dev, generator=gen)
    kz = torch.tensor(KZ, dtype=torch.float32, device=dev)[None]
    km = (y >= LO) & (~refused) & (kz <= zmax)
    if kpt == "random":
        km = km & (torch.rand(B, 1, device=dev, generator=gen) > p_none)
    elif kpt == "none":
        km = torch.zeros_like(km)
    xk = torch.where(km, (y - VMU) / VSD, torch.zeros_like(y))
    x = torch.cat([xd, mask.float(), xs, xk, km.float()], 1)
    return x, m, mask, km


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
        return o[:, :s.nout] * VSD + VMU, o[:, s.nout:].clamp(-6, 3)     # mean, log sd of ln Vs


NIN = 3 * G + 2 * K
net = Net(NIN, L).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.05)
gen = torch.Generator(device=dev); gen.manual_seed(1)
print(f"device {dev}; {N} profiles, {L} layers, {G} wavelengths, {K} KPT increments; params {sum(p.numel() for p in net.parameters())/1e6:.1f} M; tau {TAU:.3f} sigma_w {SW:.3f}", flush=True)


def nll(mu, ls, m): return (0.5 * ((m - mu) / ls.exp()) ** 2 + ls).mean()


def evaluate(kpt):
    net.eval(); g = torch.Generator(device=dev); g.manual_seed(123)
    with torch.no_grad():
        x, m, mask, km = observe(vi.to(dev), g, kpt=kpt)
        mu, ls = net(x); e = (mu - m); sd = ls.exp()
        res = dict(nll=float(nll(mu, ls, m)), rmse=torch.sqrt((e ** 2).mean(0)).cpu().tolist(), sd=sd.mean(0).cpu().tolist(),
                   cov68=(e.abs() <= sd).float().mean(0).cpu().tolist(), cov95=(e.abs() <= 1.96 * sd).float().mean(0).cpu().tolist(),
                   kpt_share=float(km.any(1).float().mean()))
    net.train(); return res


t0 = time.time(); hist = []
for step in range(1, STEPS + 1):
    idx = ti[torch.randint(0, len(ti), (BATCH,))].to(dev)
    x, m, _, _ = observe(idx, gen)
    mu, ls = net(x); loss = nll(mu, ls, m)
    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    if step % 2000 == 0 or step == STEPS:
        a, b = evaluate("all"), evaluate("none")
        hist.append(dict(step=step, train=float(loss), val_kpt=a["nll"], val_none=b["nll"], t=time.time() - t0))
        print(f"step {step:6d} train {float(loss):+.3f} | val NLL with KPT {a['nll']:+.3f} without {b['nll']:+.3f} | rmse top-5 layers with {np.mean(a['rmse'][:5]):.3f} without {np.mean(b['rmse'][:5]):.3f} | {time.time()-t0:.0f} s", flush=True)

import os
os.makedirs(out, exist_ok=True)
final = dict(with_kpt=evaluate("all"), without_kpt=evaluate("none"), hist=hist, seconds=time.time() - t0, device=dev,
             n_profiles=int(N), steps=STEPS, batch=BATCH, params=int(sum(p.numel() for p in net.parameters())),
             zmid=D["zmid"].tolist(), thick=THICK.tolist(), tau=TAU, sigma_w=SW)
json.dump(final, open(f"{out}/metrics.json", "w"))
torch.save(dict(state=net.state_dict(), nin=NIN, nout=L, G=G, K=K, lam=D["lam"], thick=THICK, kz=KZ, vmu=VMU, vsd=VSD), f"{out}/model.pt")
print("saved", out, f"{time.time()-t0:.0f} s")
