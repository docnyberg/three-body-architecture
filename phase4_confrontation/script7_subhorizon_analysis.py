#!/usr/bin/env python3
"""
script7_subhorizon_analysis.py — v1 (no guard) vs v2 (sub-horizon guard k>aH)
across the three observables: Delta P/P(k=0.1, z=0), lensed low-l TT, C_l^phiphi.

Answers the quarantine question from scope §7: how much of the v1 low-l TT
effect (incl. the 1800% caseC number) was super-horizon-drag artifact, and does
the lensing kill on the surviving corner survive the causal guard?

Reads class_tier2/ (v1: caseA/B/C_00_*; v2 guarded: caseA_sh/B_sh/C_sh_00_*;
regression twin caseBv2_00_* checked in run_tier2b.sh).
Outputs: fig7a_dPP_guarded, fig7b_TT_guarded, fig7c_clpp_guarded,
script7_results.json, console table.
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

PAIRS = [  # (v1 root, v2 root, label)
    ("caseA", "caseA_sh", "Γ₀=0.03, ε=1e-6"),
    ("caseB", "caseB_sh", "Γ₀=0.1, ε=1e-5"),
    ("caseC", "caseC_sh", "Γ₀=6, ε=1"),
]


def latest(root: str, kind: str) -> Path:
    cands = sorted(DIR.glob(f"{root}_*_{kind}.dat"))
    # guard against caseA matching caseA_sh: require next char after root is digit-ish '_NN_'
    cands = [c for c in cands if re.fullmatch(rf"{re.escape(root)}_\d+_{kind}\.dat", c.name)]
    if not cands:
        raise FileNotFoundError(f"no {root}_NN_{kind}.dat in {DIR}")
    return cands[-1]


def read_cols(path: Path):
    names = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                toks = re.findall(r"(\d+):([A-Za-z_()^+\-\[\]/0-9.]+)", line)
                if toks:
                    names = {name: int(i) - 1 for i, name in toks}
            else:
                break
    return names, np.loadtxt(path)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 78)
    print("script7 — sub-horizon guard (k > aH): v1 vs v2 across the observables")
    print("=" * 78)

    _, pk_base = read_cols(latest("base", "pk"))
    nb, cl_base = read_cols(latest("base", "cl_lensed"))
    iTT = nb.get("TT", 1)
    nbu, clu_base = read_cols(latest("base", "cl"))
    ipp = nbu.get("phiphi")

    results = {}
    figA, axA = plt.subplots(figsize=(7.5, 4.6))
    figB, axB = plt.subplots(figsize=(7.5, 4.6))
    figC, axC = plt.subplots(figsize=(7.5, 4.6))

    hdr = (f"{'case':<22}{'ΔP/P k=0.1':>22}{'max|ΔTT| ℓ2-30':>22}{'ΔC^φφ ℓ=40':>18}")
    print(hdr); print("-" * len(hdr))

    for v1, v2, label in PAIRS:
        row = {}
        colors = None
        for tag, root in (("v1", v1), ("v2", v2)):
            _, pk = read_cols(latest(root, "pk"))
            assert np.allclose(pk[:, 0], pk_base[:, 0])
            dpp = pk[:, 1] / pk_base[:, 1] - 1.0
            k = pk[:, 0]
            i01 = int(np.argmin(np.abs(k - 0.1)))

            _, cl = read_cols(latest(root, "cl_lensed"))
            m = min(len(cl), len(cl_base))
            ell = cl_base[:m, 0]
            relTT = cl[:m, iTT] / cl_base[:m, iTT] - 1.0
            low = (ell >= 2) & (ell <= 30)
            ttmax = float(np.max(np.abs(relTT[low])) * 100)

            dpp40 = None
            if ipp is not None:
                _, clu = read_cols(latest(root, "cl"))
                mu = min(len(clu), len(clu_base))
                ellu = clu_base[:mu, 0]
                relpp = clu[:mu, ipp] / clu_base[:mu, ipp] - 1.0
                dpp40 = float(relpp[int(np.argmin(np.abs(ellu - 40)))] * 100)

            row[tag] = {"dpp_k0.1_pct": float(dpp[i01] * 100),
                        "maxTT_l2_30_pct": ttmax, "dclpp_l40_pct": dpp40}

            ls = "--" if tag == "v1" else "-"
            lw = 1.0 if tag == "v1" else 1.7
            lbl = f"{label} [{tag}{'+guard' if tag == 'v2' else ''}]"
            ln, = axA.semilogx(k, dpp * 100, ls, lw=lw, label=lbl,
                               color=colors)
            colors = ln.get_color()
            sel = (ell >= 2) & (ell <= 100)
            axB.semilogx(ell[sel], relTT[sel] * 100, ls, lw=lw, label=lbl,
                         color=colors)
            if ipp is not None:
                selu = (ellu >= 2) & (ellu <= 400)
                axC.semilogx(ellu[selu], relpp[selu] * 100, ls, lw=lw,
                             label=lbl, color=colors)

        results[v1] = row
        print(f"{label:<22}"
              f"{row['v1']['dpp_k0.1_pct']:+9.2f}% → {row['v2']['dpp_k0.1_pct']:+8.2f}%"
              f"{row['v1']['maxTT_l2_30_pct']:>11.1f}% → {row['v2']['maxTT_l2_30_pct']:>7.1f}%"
              f"{row['v1']['dclpp_l40_pct']:>+10.1f}% → {row['v2']['dclpp_l40_pct']:+7.1f}%")

    for ax, ylab, title, fn in (
        (axA, "ΔP/P at z=0  [%]", "P(k): guard barely matters at interior k", "fig7a_dPP_guarded"),
        (axB, "ΔC_ℓ^TT/C_ℓ^TT  [%]", "low-ℓ TT: how much was super-horizon artifact", "fig7b_TT_guarded"),
        (axC, "ΔC_ℓ^φφ/C_ℓ^φφ  [%]", "lensing: does the kill survive the causal guard", "fig7c_clpp_guarded"),
    ):
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("k [h/Mpc]" if ax is axA else "ℓ")
        ax.set_ylabel(ylab); ax.set_title(title); ax.legend(fontsize=7)
        fig = ax.figure; fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{fn}.{ext}", dpi=150)
        plt.close(fig)

    (OUT / "script7_results.json").write_text(json.dumps(results, indent=1),
                                              encoding="utf-8")
    print("\nfigures: fig7a_dPP_guarded, fig7b_TT_guarded, fig7c_clpp_guarded "
          "(+ script7_results.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
