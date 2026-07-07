#!/usr/bin/env python3
"""
script6_class_tier2_analysis.py — Tier-2 analysis of the patched-CLASS runs.

Reads class_tier2/*.dat (produced by run_tier2.sh in WSL ~/class_public):
  base   — vanilla env (GEO unset)          [latest index used]
  zero   — GEO_GAMMA0=0 (patch off-state)   [REAL integrity check vs base]
  caseA  — Gamma0=0.03, eps=1e-6  (Tier-1 surviving corner, -9.2% @k=0.1)
  caseB  — Gamma0=0.1,  eps=1e-5  (Tier-1 surviving corner, -6.3% @k=0.1)
  caseC  — Gamma0=6,    eps=1     (pushed-cliff ISW probe)

INTEGRITY NOTE (honest record): run_tier2.sh's in-shell integrity gate
compared filenames WITHOUT the _NN_ index CLASS inserts (base_pk.dat vs
actual base_00_pk.dat), so both greps read nothing and cmp compared
empty-with-empty — a VACUOUS pass. The REAL off-state identity check is
performed HERE, numerically, and reported. (Physics gate discipline: a
gate that cannot fail is not a gate.)

Outputs: fig6a_dPP_class.(png|pdf), fig6b_TT_lowl.(png|pdf),
         fig6c_clpp.(png|pdf), script6_results.json, console summary.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = Path(__file__).parent / "class_tier2"
OUT = Path(__file__).parent

CASES = {
    "caseA": {"gamma0": 0.03, "eps": 1e-6, "tier1_dpp_k01": -9.2},
    "caseB": {"gamma0": 0.1,  "eps": 1e-5, "tier1_dpp_k01": -6.3},
    "caseC": {"gamma0": 6.0,  "eps": 1.0,  "tier1_dpp_k01": None},
}


def latest(root: str, kind: str) -> Path:
    """Pick the highest _NN_ index file for a run root ('base', 'caseA'...)."""
    cands = sorted(DIR.glob(f"{root}_*_{kind}.dat"))
    if not cands:
        raise FileNotFoundError(f"no {root}_*_{kind}.dat in {DIR}")
    return cands[-1]


def read_cols(path: Path) -> tuple[dict, np.ndarray]:
    """Parse a CLASS .dat: return (name->index map from the last header line,
    data array)."""
    names = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                toks = re.findall(r"(\d+):([A-Za-z_()^+\-\[\]/0-9.]+)", line)
                if toks:
                    names = {name: int(i) - 1 for i, name in toks}
            else:
                break
    data = np.loadtxt(path)
    return names, data


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 74)
    print("script6 — Tier-2 CLASS analysis (patched class_public @ e858083)")
    print("=" * 74)

    # ---------- REAL integrity check: zero vs base ----------
    _, pk_base = read_cols(latest("base", "pk"))
    _, pk_zero = read_cols(latest("zero", "pk"))
    if pk_base.shape != pk_zero.shape:
        print("INTEGRITY: shape mismatch base vs zero — FAIL")
        return 2
    same_k = np.allclose(pk_base[:, 0], pk_zero[:, 0], rtol=0, atol=0)
    max_rel = float(np.max(np.abs(pk_zero[:, 1] / pk_base[:, 1] - 1.0)))
    integ = same_k and max_rel == 0.0
    print(f"REAL off-state integrity (zero vs base): identical k-grid={same_k}, "
          f"max |P_zero/P_base - 1| = {max_rel:.3e} -> "
          f"{'PASS (bitwise)' if integ else ('PASS (numeric<1e-12)' if max_rel < 1e-12 else 'FAIL')}")
    if not same_k or max_rel > 1e-12:
        print("ABORT: patched off-state is not vanilla — patch physics suspect.")
        return 2

    results = {"integrity_max_rel": max_rel, "cases": {}}

    # ---------- Delta P / P ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, meta in CASES.items():
        _, pk = read_cols(latest(name, "pk"))
        assert np.allclose(pk[:, 0], pk_base[:, 0])
        dpp = pk[:, 1] / pk_base[:, 1] - 1.0
        k = pk[:, 0]
        ax.semilogx(k, dpp * 100, lw=1.5,
                    label=f"{name}: Γ₀={meta['gamma0']}, ε={meta['eps']:g}")
        i01 = int(np.argmin(np.abs(k - 0.1)))
        meta_out = {"dpp_k0.1_class_pct": float(dpp[i01] * 100),
                    "tier1_dpp_k0.1_pct": meta["tier1_dpp_k01"]}
        results["cases"][name] = meta_out
        t1 = meta["tier1_dpp_k01"]
        print(f"{name}: CLASS ΔP/P(k=0.1,z=0) = {dpp[i01]*100:+.2f}%"
              + (f"   [Tier-1 growth-ODE: {t1:+.1f}%]" if t1 is not None else "   [ISW probe]"))
        if t1 is not None:
            ax.plot(0.1, t1, "x", ms=9, mew=2, color=ax.lines[-1].get_color())
    ax.set_xlabel("k  [h/Mpc]"); ax.set_ylabel("ΔP/P at z=0  [%]")
    ax.set_title("Tier 2 (full Boltzmann): ΔP/P — ×=Tier-1 cross-check")
    ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig6a_dPP_class.{ext}", dpi=150)
    plt.close(fig)

    # ---------- lensed TT low-l ----------
    nb, cl_b = read_cols(latest("base", "cl_lensed"))
    iTT = nb.get("TT", 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, meta in CASES.items():
        nc, cl = read_cols(latest(name, "cl_lensed"))
        m = min(len(cl), len(cl_b))
        ell = cl_b[:m, 0]
        rel = cl[:m, iTT] / cl_b[:m, iTT] - 1.0
        sel = (ell >= 2) & (ell <= 100)
        ax.semilogx(ell[sel], rel[sel] * 100, lw=1.4,
                    label=f"{name}: Γ₀={meta['gamma0']}, ε={meta['eps']:g}")
        low = (ell >= 2) & (ell <= 30)
        results["cases"][name]["max_abs_dTT_l2_30_pct"] = float(
            np.max(np.abs(rel[low])) * 100)
        print(f"{name}: max |ΔC_l^TT|/C_l (lensed, 2<=l<=30) = "
              f"{np.max(np.abs(rel[low]))*100:.3f}%")
    ax.set_xlabel("ℓ"); ax.set_ylabel("ΔC_ℓ^TT / C_ℓ^TT  [%]  (lensed)")
    ax.set_title("Tier 2: late-ISW imprint — what Tier 1 could not see")
    ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig6b_TT_lowl.{ext}", dpi=150)
    plt.close(fig)

    # ---------- lensing potential phiphi (from unlensed cl.dat) ----------
    nbu, clu_b = read_cols(latest("base", "cl"))
    ipp = nbu.get("phiphi")
    if ipp is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for name, meta in CASES.items():
            _, clu = read_cols(latest(name, "cl"))
            m = min(len(clu), len(clu_b))
            ell = clu_b[:m, 0]
            rel = clu[:m, ipp] / clu_b[:m, ipp] - 1.0
            sel = (ell >= 2) & (ell <= 400)
            ax.semilogx(ell[sel], rel[sel] * 100, lw=1.4,
                        label=f"{name}: Γ₀={meta['gamma0']}, ε={meta['eps']:g}")
            results["cases"][name]["dClpp_l40_pct"] = float(
                rel[int(np.argmin(np.abs(ell - 40)))] * 100)
            print(f"{name}: ΔC_l^φφ/C_l^φφ at l=40 = "
                  f"{rel[int(np.argmin(np.abs(ell-40)))]*100:+.3f}%")
        ax.set_xlabel("ℓ"); ax.set_ylabel("ΔC_ℓ^φφ / C_ℓ^φφ  [%]")
        ax.set_title("Tier 2: lensing-potential suppression")
        ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"fig6c_clpp.{ext}", dpi=150)
        plt.close(fig)
    else:
        print("phiphi column not found in cl.dat — skipping fig6c")

    (OUT / "script6_results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    print("\nfigures: fig6a_dPP_class, fig6b_TT_lowl, fig6c_clpp "
          "(+ script6_results.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
