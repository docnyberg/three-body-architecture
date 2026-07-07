#!/usr/bin/env python3
"""
script8_desi_template_filter.py — Tier-3 D3 TEMPLATE FILTER against real
DESI DR1 data (ratified: "template-level DESI first as the filter;
desilike-grade only if the template shows signal").

DATA (fetched 2026-07-06, public https, no auth):
  desi_dr1/likelihood_spectrum-poles-rotated_syst-rotation-hod-photo_
           LRG_GCcomb_z0.8-1.1_thetacut0.05.h5
  = P0,P2 data vector (72), full syst covariance (72x72), window matrix
    (72x1047) and the release's own fiducial theory multipoles (l=0,2,4
    on a 349-pt theory k-grid). Source: data.desi.lbl.gov DR1 VAC
    "Full Shape and BAO clustering products" v1.0/data/likelihood.

METHOD (filter-grade, stated limitations):
  The bundle's theory-side `value` arrays are ZERO placeholders (verified) —
  the theory group supplies only the k-grid. So the fiducial is built from
  OUR CLASS ν-less baseline: P_m(k_th, z=0.95) interpolated from t3base.
  Two-block linear model (per-multipole amplitudes absorb b², b·f, f²
  Kaiser combinations, data-calibrated; theory ℓ=4 block left zero —
  its window-mixed contribution is small at filter grade):
      m(S) = A0 · W@[S·P_m | 0 | 0]  +  A2 · W@[0 | S·P_m | 0]
  GLS over (A0, A2):  θ* = (MᵀC⁻¹M)⁻¹ MᵀC⁻¹ d ;
      chi2_min = dᵀC⁻¹d − dᵀC⁻¹M θ*
  Delta_chi2(case) = chi2_min(S_case) − chi2_min(S=1)
  S(k) from guarded (v2) CLASS runs at z_pk = 0.95 ~ LRG3 bin effective z.
  NOT included: EFT counterterms/FoG, AP distortion, shot-noise freedom —
  hence FILTER verdicts only:  <4 survives | 4-25 pressured | >25 excluded
  (chi-square intuition; formal ruling = desilike-grade, only if something
  here shows signal). Baseline GOF printed — if poor at high k (expected
  without EFT terms), Delta_chi2 remains valid comparatively since baseline
  and S-models share identical nuisance freedom.

Outputs: fig8a_desi_p0_overlay.(png|pdf), fig8b_dchi2_scan.(png|pdf),
         script8_results.json, console verdict table.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
H5 = (HERE / "desi_dr1" /
      "likelihood_spectrum-poles-rotated_syst-rotation-hod-photo_"
      "LRG_GCcomb_z0.8-1.1_thetacut0.05.h5")
T2 = HERE / "class_tier2"

CASES = [  # (root, gamma0, eps)
    ("t3E", 0.003, 1e-6),
    ("t3D", 0.01,  1e-6),
    ("t3A", 0.03,  1e-6),
    ("t3F", 0.01,  1e-5),
    ("t3B", 0.1,   1e-5),
]
ZTARGET = 0.95


def pk_at_z(root: str, ztarget: float) -> np.ndarray:
    """Load the pk file whose header redshift matches ztarget; return (k, P)."""
    for p in sorted(T2.glob(f"{root}_00_z*_pk.dat")):
        head = p.read_text(encoding="utf-8", errors="ignore")[:400]
        m = re.search(r"redshift\s*z\s*=\s*([0-9.eE+-]+)", head)
        if m and abs(float(m.group(1)) - ztarget) < 0.01:
            return np.loadtxt(p)
    raise FileNotFoundError(f"{root}: no pk file at z={ztarget}")


def phiphi_delta(root: str, base: str, ell0: int = 40) -> float:
    def load(r):
        p = sorted(T2.glob(f"{r}_00_cl.dat"))[-1]
        names = {}
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#"):
                    toks = re.findall(r"(\d+):([A-Za-z]+)", line)
                    if toks:
                        names = {n: int(i) - 1 for i, n in toks}
                else:
                    break
        return names, np.loadtxt(p)
    nb, db = load(base)
    nc, dc = load(root)
    i = nb["phiphi"]
    ell = db[:, 0]
    j = int(np.argmin(np.abs(ell - ell0)))
    return float((dc[j, i] / db[j, i] - 1.0) * 100)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 78)
    print("script8 — Tier-3 D3 template filter vs DESI DR1 LRG3 (z0.8-1.1, GCcomb)")
    print("=" * 78)

    # ---------- load the DESI bundle ----------
    with h5py.File(H5) as f:
        d = np.concatenate([f["observable/spectrum/0/value"][:],
                            f["observable/spectrum/2/value"][:]])
        k_obs = f["observable/spectrum/0/k"][:]
        C = f["covariance/value"][:]
        W = f["window/value"][:]
        k_th = f["window/theory/spectrum/0/k"][:]
        t_zero = np.concatenate([f["window/theory/spectrum/0/value"][:],
                                 f["window/theory/spectrum/2/value"][:],
                                 f["window/theory/spectrum/4/value"][:]])
    n_obs, n_th = W.shape
    nk = len(k_th)
    print(f"data vector: {d.shape[0]}  covariance: {C.shape}  window: {W.shape}")
    print(f"theory grid: {nk} pts, k in [{k_th.min():.4g}, {k_th.max():.4g}] h/Mpc")
    assert n_obs == d.shape[0] == C.shape[0] and n_th == 3 * nk
    if np.all(t_zero == 0):
        print("confirmed: bundle theory-side fiducial is a ZERO placeholder — "
              "building fiducial from CLASS t3base (see METHOD).")

    Cinv = np.linalg.inv(C)

    # ---------- CLASS baseline P_m on the theory grid ----------
    base_dat = pk_at_z("t3base", ZTARGET)
    kb, Pb = base_dat[:, 0], base_dat[:, 1]
    Pm_th = np.interp(k_th, kb, Pb)          # k_th within CLASS range (checked above)

    def blocks(S: np.ndarray) -> np.ndarray:
        """Design matrix (n_obs x 2): windowed S*P_m in the l=0 block, and in
        the l=2 block; theory l=4 left zero (filter-grade)."""
        z = np.zeros(nk)
        col0 = W @ np.concatenate([S * Pm_th, z, z])
        col2 = W @ np.concatenate([z, S * Pm_th, z])
        return np.column_stack([col0, col2])

    def chi2_min(M: np.ndarray) -> tuple[float, np.ndarray]:
        A = M.T @ Cinv @ M
        b = M.T @ Cinv @ d
        theta = np.linalg.solve(A, b)
        return float(d @ Cinv @ d - b @ theta), theta

    S_ONE = np.ones(nk)
    M0 = blocks(S_ONE)
    chi2_0, th0 = chi2_min(M0)
    dof = n_obs - 2
    print(f"BASELINE (CLASS fiducial, A0/A2 marginalized): "
          f"chi2 = {chi2_0:.1f} / dof ~ {dof}  (A0={th0[0]:.3f}, A2={th0[1]:.3f})")
    if chi2_0 > 2.5 * dof:
        print("WARNING: baseline fits poorly (expected without EFT terms at high k) "
              "— Delta_chi2 remains valid comparatively.")

    results = {"baseline_chi2": chi2_0, "dof": dof,
               "A0": float(th0[0]), "A2": float(th0[1]), "cases": []}
    print()
    hdr = (f"{'case':<8}{'Gamma0':>8}{'eps':>9}{'S(k=0.1)':>10}"
           f"{'Dchi2':>10}{'dC_pp@40':>10}   verdict")
    print(hdr); print("-" * len(hdr))

    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    ib = slice(0, len(k_obs))
    sig0 = np.sqrt(np.diag(C))[ib]
    ax_a.errorbar(k_obs, k_obs * d[ib], yerr=k_obs * sig0, fmt="k.", ms=4,
                  lw=0.8, label="DESI DR1 LRG3 P0 (GCcomb)")
    fit0 = M0 @ th0
    ax_a.plot(k_obs, k_obs * fit0[ib], "b-", lw=1.3,
              label=f"windowed CLASS fiducial, A0/A2 fit (χ²={chi2_0:.0f}/{dof})")

    bars = []
    for root, g0, eps in CASES:
        case_dat = pk_at_z(root, ZTARGET)
        kc, Pc = case_dat[:, 0], case_dat[:, 1]
        assert np.allclose(kc, kb)
        ratio = Pc / Pb
        S = np.interp(k_th, kb, ratio, left=ratio[0], right=ratio[-1])
        Mc = blocks(S)
        chi2_c, thc = chi2_min(Mc)
        dchi2 = chi2_c - chi2_0
        s01 = float(np.interp(0.1, kb, ratio))
        dpp = phiphi_delta(root, "t3base")
        verdict = ("SURVIVES filter" if dchi2 < 4 else
                   "PRESSURED" if dchi2 < 25 else "EXCLUDED (filter-grade)")
        if abs(dpp) > 5 and dchi2 < 25:
            verdict += " but phi-phi kills it"
        print(f"{root:<8}{g0:>8g}{eps:>9.0e}{(s01-1)*100:>+9.1f}%"
              f"{dchi2:>10.1f}{dpp:>+9.1f}%   {verdict}")
        results["cases"].append({
            "root": root, "gamma0": g0, "eps": eps,
            "S_k0.1_minus1_pct": (s01 - 1) * 100, "delta_chi2": dchi2,
            "A0": float(thc[0]), "A2": float(thc[1]),
            "dClpp_l40_pct": dpp, "verdict": verdict})
        bars.append((f"Γ₀={g0:g}\nε={eps:.0e}", dchi2, dpp))
        fitc = Mc @ thc
        ax_a.plot(k_obs, k_obs * fitc[ib], "--", lw=1.1,
                  label=f"Γ₀={g0:g}, ε={eps:.0e} (Δχ²={dchi2:+.1f})")

    ax_a.set_xlabel("k  [h/Mpc]"); ax_a.set_ylabel("k·P₀(k)")
    ax_a.set_title("Tier-3 filter: DESI DR1 LRG3 monopole vs windowed suppression models")
    ax_a.legend(fontsize=7)
    fig_a.tight_layout()
    for ext in ("png", "pdf"):
        fig_a.savefig(HERE / f"fig8a_desi_p0_overlay.{ext}", dpi=150)
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(bars))
    ax_b.bar(xs, [b[1] for b in bars], color="teal", alpha=0.8,
             label="Δχ² vs ΛCDM (DESI LRG3 filter)")
    ax_b.axhline(4, color="orange", ls="--", lw=1, label="Δχ²=4 (survive/pressure)")
    ax_b.axhline(25, color="crimson", ls="--", lw=1, label="Δχ²=25 (filter exclusion)")
    ax2 = ax_b.twinx()
    ax2.plot(xs, [b[2] for b in bars], "ko-", ms=5, lw=1.2,
             label="ΔC^φφ @ℓ=40 [%] (right)")
    ax2.axhline(-5, color="gray", ls=":", lw=1)
    ax2.set_ylabel("ΔC^φφ/C^φφ @ℓ=40  [%]")
    ax_b.set_xticks(xs, [b[0] for b in bars], fontsize=8)
    ax_b.set_ylabel("Δχ²")
    ax_b.set_title("Tier-3 filter verdicts: DESI P(k) Δχ² + Planck-facing φφ")
    ax_b.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")
    fig_b.tight_layout()
    for ext in ("png", "pdf"):
        fig_b.savefig(HERE / f"fig8b_dchi2_scan.{ext}", dpi=150)
    plt.close(fig_b)

    (HERE / "script8_results.json").write_text(json.dumps(results, indent=1),
                                               encoding="utf-8")
    print("\nfigures: fig8a_desi_p0_overlay, fig8b_dchi2_scan (+ script8_results.json)")
    print("NOTE: filter-grade only — no EFT nuisances/AP/shot-noise freedom; "
          "desilike-grade earns entry ONLY on signal (ratified D3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
