#!/usr/bin/env python3
"""
script9_hardening_analysis.py — referee-hardening part A+B (ratified 2026-07-06).

A) 3-BIN JOINT DESI FILTER: repeats the script8 GLS filter on ALL THREE LRG
   bundles (z0.4-0.6 @ z_eff~0.5, z0.6-0.8 @ ~0.7, z0.8-1.1 @ ~0.95; mid-bin
   z_eff = stated filter-grade approximation), joint Delta_chi2 = sum over bins
   (disjoint z-ranges treated as independent — stated).
B) NU-ROBUSTNESS: the same joint filter computed from the nu-RESTORED CLASS set
   (N_ncdm=1, m=0.06 eV) — Delta_chi2 stability check vs the nu-less pair.

S(k) per bin from run_tier4.sh outputs (t4* nu-less, t4n* nu-restored), all
sub-horizon-guarded. Outputs: fig9_joint_filter.(png|pdf),
script9_results.json, console table.
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
T2 = HERE / "class_tier2"
DESI = HERE / "desi_dr1"

BINS = [  # (zrange tag, z_eff approx)
    ("z0.4-0.6", 0.5),
    ("z0.6-0.8", 0.7),
    ("z0.8-1.1", 0.95),
]
CASES = [("E", 0.003, 1e-6), ("D", 0.01, 1e-6), ("A", 0.03, 1e-6),
         ("F", 0.01, 1e-5), ("B", 0.1, 1e-5)]
SETS = {"nu-less": "t4", "nu-restored": "t4n"}


def pk_at_z(root: str, ztarget: float) -> np.ndarray:
    for p in sorted(T2.glob(f"{root}_00_z*_pk.dat")):
        head = p.read_text(encoding="utf-8", errors="ignore")[:400]
        m = re.search(r"redshift\s*z\s*=\s*([0-9.eE+-]+)", head)
        if m and abs(float(m.group(1)) - ztarget) < 0.02:
            return np.loadtxt(p)
    raise FileNotFoundError(f"{root} @ z={ztarget}")


def load_bundle(zr: str):
    p = (DESI / f"likelihood_spectrum-poles-rotated_syst-rotation-hod-photo_"
                f"LRG_GCcomb_{zr}_thetacut0.05.h5")
    with h5py.File(p) as f:
        d = np.concatenate([f["observable/spectrum/0/value"][:],
                            f["observable/spectrum/2/value"][:]])
        C = f["covariance/value"][:]
        W = f["window/value"][:]
        k_th = f["window/theory/spectrum/0/k"][:]
    return d, np.linalg.inv(C), W, k_th, C.shape[0]


def bin_machinery(d, Cinv, W, k_th, Pm_th):
    nk = len(k_th)
    z = np.zeros(nk)

    def chi2_min(S):
        col0 = W @ np.concatenate([S * Pm_th, z, z])
        col2 = W @ np.concatenate([z, S * Pm_th, z])
        M = np.column_stack([col0, col2])
        A = M.T @ Cinv @ M
        b = M.T @ Cinv @ d
        theta = np.linalg.solve(A, b)
        return float(d @ Cinv @ d - b @ theta)
    return chi2_min


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 84)
    print("script9 — referee hardening: 3-bin joint DESI filter + nu-robustness")
    print("=" * 84)

    results = {"bins": [b[0] for b in BINS], "sets": {}}
    joint_rows = {}

    for set_name, prefix in SETS.items():
        print(f"\n--- SET: {set_name} ({prefix}*) ---")
        per_bin_machinery = []
        base_chi2 = []
        for zr, zeff in BINS:
            d, Cinv, W, k_th, nobs = load_bundle(zr)
            bdat = pk_at_z(f"{prefix}base", zeff)
            kb, Pb = bdat[:, 0], bdat[:, 1]
            Pm_th = np.interp(k_th, kb, Pb)
            f = bin_machinery(d, Cinv, W, k_th, Pm_th)
            chi0 = f(np.ones(len(k_th)))
            per_bin_machinery.append((zr, zeff, f, kb, Pb, k_th))
            base_chi2.append(chi0)
            print(f"  bin {zr} (z_eff~{zeff}): baseline chi2 = {chi0:.1f} / ~{nobs-2}")
        results["sets"][set_name] = {"baseline_chi2_per_bin": base_chi2, "cases": []}

        hdr = (f"  {'case':<6}{'G0':>7}{'eps':>9}"
               + "".join(f"{('d.chi2 ' + b[0]):>16}" for b in BINS) + f"{'JOINT':>10}")
        print(hdr)
        for tag, g0, eps in CASES:
            row = []
            for zr, zeff, f, kb, Pb, k_th in per_bin_machinery:
                cdat = pk_at_z(f"{prefix}{tag}", zeff)
                ratio = cdat[:, 1] / Pb
                S = np.interp(k_th, kb, ratio, left=ratio[0], right=ratio[-1])
                i = [b[0] for b in BINS].index(zr)
                row.append(f(S) - base_chi2[i])
            joint = sum(row)
            print(f"  {tag:<6}{g0:>7g}{eps:>9.0e}"
                  + "".join(f"{r:>16.1f}" for r in row) + f"{joint:>10.1f}")
            results["sets"][set_name]["cases"].append(
                {"case": tag, "gamma0": g0, "eps": eps,
                 "dchi2_per_bin": row, "dchi2_joint": joint})
            joint_rows.setdefault(tag, {})[set_name] = joint

    # --- stability table ---
    print("\nNU-ROBUSTNESS (joint Delta_chi2, nu-less vs nu-restored):")
    for tag, g0, eps in CASES:
        a = joint_rows[tag]["nu-less"]
        b = joint_rows[tag]["nu-restored"]
        shift = abs(b - a) / max(abs(a), 1.0) * 100
        print(f"  {tag} (G0={g0:g}, eps={eps:.0e}):  {a:+9.1f}  vs  {b:+9.1f}   "
              f"(shift {shift:.0f}%)")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(CASES))
    w = 0.38
    for off, (set_name, _) in zip((-w / 2, w / 2), SETS.items()):
        vals = [joint_rows[t]["" + set_name] for t, _, _ in CASES]
        ax.bar(xs + off, vals, width=w, label=f"joint Δχ² ({set_name})", alpha=0.85)
    ax.axhline(4, color="orange", ls="--", lw=1); ax.axhline(25, color="crimson", ls="--", lw=1)
    ax.set_xticks(xs, [f"Γ₀={g:g}\nε={e:.0e}" for _, g, e in CASES], fontsize=8)
    ax.set_ylabel("joint Δχ² (3 LRG bins)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("Referee hardening: 3-bin joint DESI filter, ν-less vs ν-restored")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig9_joint_filter.{ext}", dpi=150)
    plt.close(fig)

    (HERE / "script9_results.json").write_text(json.dumps(results, indent=1),
                                               encoding="utf-8")
    print("\nfigure: fig9_joint_filter (+ script9_results.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
