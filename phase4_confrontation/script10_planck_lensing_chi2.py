#!/usr/bin/env python3
"""
script10_planck_lensing_chi2.py — referee-hardening part C: formal Planck
lensing bandpower chi-square, replacing the qualitative "phi-phi @ l=40" check.

DATA (fetched 2026-07-06, raw.githubusercontent.com/carronj/planck_PR4_lensing):
  planck_pr4/pp_consext8_..._PS1_bandpowers.dat  — 9 bins, L 8-400:
      columns [bin, L_min, L_max, L_av, PP, Error, Ahat]
      (Ahat = amplitude relative to the FFP10 fiducial per bin)
  planck_pr4/pp_consext8_..._PS1_cov.dat         — 9x9 covariance of PP

METHOD (filter-grade, stated):
  Amplitude space: per-bin fiducial PP_fid_b = PP_b / Ahat_b;
  Cov_A[b,b'] = Cov_PP[b,b'] / (PP_fid_b * PP_fid_b').
  Model amplitude per bin: r_b(case) = flat band-average over [L_min, L_max]
  of C_L^pp(case)/C_L^pp(base) from our guarded CLASS runs (ratios vary
  slowly within DL~45 bins — stated). chi2(case) = (Ahat - r)^T Cov_A^-1
  (Ahat - r); Delta_chi2 = chi2(case) - chi2(r=1).
  NOT included: the PR4 linear (fiducial-correction) term and the exact bin
  window functions — both subdominant to tens-of-percent phi-phi shifts.
  Run for BOTH the nu-less (t4) and nu-restored (t4n) CLASS sets.

Outputs: fig10_planck_lensing.(png|pdf), script10_results.json, console table.
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

HERE = Path(__file__).parent
T2 = HERE / "class_tier2"
PR4 = HERE / "planck_pr4"
BP = PR4 / "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_bandpowers.dat"
COV = PR4 / "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_cov.dat"

CASES = [("E", 0.003, 1e-6), ("D", 0.01, 1e-6), ("A", 0.03, 1e-6),
         ("F", 0.01, 1e-5), ("B", 0.1, 1e-5)]
SETS = {"nu-less": "t4", "nu-restored": "t4n"}


def load_clpp(root: str) -> tuple[np.ndarray, np.ndarray]:
    p = sorted(T2.glob(f"{root}_00_cl.dat"))[-1]
    names = {}
    with open(p, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                toks = re.findall(r"(\d+):([A-Za-z]+)", line)
                if toks:
                    names = {n: int(i) - 1 for i, n in toks}
            else:
                break
    dat = np.loadtxt(p)
    return dat[:, 0], dat[:, names["phiphi"]]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 80)
    print("script10 — formal Planck PR4 lensing bandpower chi2 (amplitude method)")
    print("=" * 80)

    bp = np.loadtxt(BP)
    Lmin, Lmax, PP, Ahat = bp[:, 1], bp[:, 2], bp[:, 4], bp[:, 6]
    cov = np.loadtxt(COV)
    nb = len(Ahat)
    PP_fid = PP / Ahat
    covA = cov / np.outer(PP_fid, PP_fid)
    covA_inv = np.linalg.inv(covA)
    ones = np.ones(nb)
    chi2_null = float((Ahat - ones) @ covA_inv @ (Ahat - ones))
    print(f"{nb} bandpower bins, L in [{Lmin.min():.0f}, {Lmax.max():.0f}]")
    print(f"NULL (r=1, i.e. no suppression): chi2 = {chi2_null:.2f} / {nb} bins "
          f"— PR4 data vs its own fiducial (sanity: O(nb) expected)")

    results = {"chi2_null": chi2_null, "nbins": nb, "sets": {}}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    Lc = bp[:, 3]
    ax.errorbar(Lc, Ahat, yerr=np.sqrt(np.diag(covA)), fmt="k.", ms=5,
                label="PR4 Ahat ± σ (amplitude vs fiducial)")
    ax.axhline(1.0, color="k", lw=0.7)

    for set_name, prefix in SETS.items():
        ellb, clb = load_clpp(f"{prefix}base")
        print(f"\n--- SET: {set_name} ---")
        print(f"  {'case':<6}{'G0':>7}{'eps':>9}{'<r>':>8}{'chi2':>10}{'Dchi2':>10}   verdict")
        out = []
        for tag, g0, eps in CASES:
            ellc, clc = load_clpp(f"{prefix}{tag}")
            m = min(len(clb), len(clc))
            ratio = clc[:m] / clb[:m]
            ell = ellb[:m]
            r = np.array([float(np.mean(ratio[(ell >= lo) & (ell <= hi)]))
                          for lo, hi in zip(Lmin, Lmax)])
            chi2 = float((Ahat - r) @ covA_inv @ (Ahat - r))
            dchi2 = chi2 - chi2_null
            verdict = ("survives" if dchi2 < 4 else
                       "PRESSURED" if dchi2 < 25 else "EXCLUDED")
            print(f"  {tag:<6}{g0:>7g}{eps:>9.0e}{np.mean(r):>8.3f}"
                  f"{chi2:>10.1f}{dchi2:>+10.1f}   {verdict}")
            out.append({"case": tag, "gamma0": g0, "eps": eps,
                        "mean_r": float(np.mean(r)), "chi2": chi2,
                        "delta_chi2": dchi2, "verdict": verdict})
            if set_name == "nu-restored":
                ax.plot(Lc, r, "o--", ms=3, lw=1,
                        label=f"Γ₀={g0:g}, ε={eps:.0e} (Δχ²={dchi2:+.0f})")
        results["sets"][set_name] = out

    ax.set_xlabel("L"); ax.set_ylabel("lensing amplitude vs fiducial")
    ax.set_title("Planck PR4 lensing bandpowers vs guarded-CLASS suppression (ν-restored)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig10_planck_lensing.{ext}", dpi=150)
    plt.close(fig)

    (HERE / "script10_results.json").write_text(json.dumps(results, indent=1),
                                                encoding="utf-8")
    print("\nfigure: fig10_planck_lensing (+ script10_results.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
