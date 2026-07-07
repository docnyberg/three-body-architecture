#!/usr/bin/env python3
"""
script5_kdep_growth.py — Tier 1 of PHASE3_BOLTZMANN_DESI_SCOPE_2026-07-06.md

K-RESOLVED MODIFIED-GROWTH PIPELINE (CAMB post-processing).
Implements the piece of physics-as-code the archaeology found left behind:
the mode-dependent trigger of Syllabus Task 3.2, on the CORRECTED growth
baseline (Deep Think eq.-22 fix: EdS friction 3/(2a), not 3/a).

Growth equation (sub-horizon, Newtonian gauge, matter clustering only):
    delta'' + [ 3/a + dlnH/da + Gamma_geo(k,a)/a ] delta'
            - (3/(2 a^2)) Omega_m(a) delta = 0        (' = d/da)
EdS limit, Gamma=0:  delta'' + (3/(2a)) delta' - (3/(2a^2)) delta = 0
                     -> delta ~ a  (the Deep Think check passes by construction)

Mode-dependent trigger (Syllabus Task 3.2):  r^3 |R(a)| > eps
CONVENTION (resolved ambiguity, flagged): r is the PROPER mode scale
    r_prop(a,k) = a * (2*pi/k)      [comoving 2pi/k, in units of c/H0]
    R(a)        = 3 (Om_m a^-3 + 4 Om_L)   [FRW Ricci in H0^2 units,
                                            matches syllabus/toy solver]
With comoving r the trigger fires EARLY (R ~ a^-3 diverges backward), which
contradicts the toy's late (z~24) onset; proper r gives late, large-scale-first
onset:  r^3 R = 3 (2pi/k)^3 (Om_m + 4 Om_L a^3)  -> monotone INCREASING in a.
Onset therefore exists iff eps > 3 (2pi/k)^3 Om_m, and larger scales (small k)
cross first. This is what makes the suppression k-DEPENDENT.

Friction form matches the toy/syllabus: Gamma_geo = Gamma0/a while triggered
(entering the delta' coefficient as Gamma0/a * delta', i.e. +Gamma0/a^2 inside
the bracket's 1/a... explicitly: coefficient 3/a + dlnH/da + Gamma0/a^2? NO —
see note). NOTE ON NORMALIZATION: the toy wrote  delta'' = -(3/a + Gamma0/a)
delta' + ...  i.e. Gamma0/a sits BESIDE 3/a. We keep exactly that:
    total delta' coefficient = 3/a + dlnH/da + (Gamma0/a if triggered else 0)
so Gamma0 is directly comparable to the syllabus's Gamma0 = 6.

BASELINE GATE (Step-0 discipline): with Gamma0 = 0 the ODE growth ratio
D(z)/D(0) must match CAMB's sqrt(P(k_gate,z)/P(k_gate,0)) to < 0.5% for
z in [0, 30], else NOTHING downstream is reported.

Planck 2018 (no retuning): H0=67.36, ombh2=0.02237, omch2=0.1200,
ns=0.9649, As tuned once so sigma8(0)=0.811, mnu=0.06 eV in CAMB.
(The ODE is nu-less; the gate measures whether that approximation holds
at k_gate over z<30 — if the gate fails, the fallback is mnu=0, reported.)

Outputs (all in this directory):
    fig5a_baseline_gate.png/.pdf      gate residual vs z
    fig5b_dP_over_P.png/.pdf          Delta P/P (k) at z=0, family over eps
    fig5c_sigma8_scan.png/.pdf        heatmap: sigma8 suppression over (Gamma0, eps)
    fig5d_fsigma8.png/.pdf            fsigma8(z) at k_ref=0.1 h/Mpc (theory only;
                                      data overlay is Tier 3)
    script5_scan_results.json         full scan numbers
    script5_output.txt (via tee/redirect when run by the harness)

Reproducibility: deterministic; re-run must match byte-for-byte (scripts 1-4
standard).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import camb

OUTDIR = Path(__file__).parent

# ----------------------------- cosmology ---------------------------------
H0 = 67.36
h = H0 / 100.0
ombh2, omch2 = 0.02237, 0.1200
ns = 0.9649
# FALLBACK TAKEN (first run, 2026-07-06): with MNU=0.06 the baseline gate FAILED
# at 0.579% vs the 0.5% tolerance — the nu-less ODE vs massive-nu CAMB is a
# like-for-unlike comparison at k_gate. Tier 1 therefore runs the CONSISTENT
# nu-less pair (MNU=0 in CAMB too, sigma8 re-tuned to 0.811); massive-nu
# treatment is deferred to Tier 2 (CLASS), where it belongs. Gate discipline
# preserved: the first run reported NOTHING downstream (see script5_output
# history in git/console).
MNU = 0.0
SIGMA8_TARGET = 0.811

Om_m = (ombh2 + omch2) / h**2            # baryon+CDM (nu-less ODE)
Om_r = 4.15e-5 / h**2                     # photons + massless nu (standard)
Om_L = 1.0 - Om_m - Om_r                  # flat

A_INIT = 1e-2                             # z = 99 start, EdS growing-mode ICs
Z_GATE_MAX = 30.0
K_GATE = 0.05                             # h/Mpc — gate scale
GATE_TOL = 0.005                          # < 0.5%

C_over_H0_Mpc_h = 2997.92458              # c/H0 in Mpc/h units (k in h/Mpc)


def E2(a):
    return Om_r * a**-4 + Om_m * a**-3 + Om_L


def dlnH_da(a: float) -> float:
    """d ln H / da for H^2 = H0^2 E2(a)."""
    dE2 = -4 * Om_r * a**-5 - 3 * Om_m * a**-4
    return 0.5 * dE2 / E2(a)


def Omega_m_a(a: float) -> float:
    return Om_m * a**-3 / E2(a)


def ricci_H0sq(a: float) -> float:
    """FRW Ricci scalar in H0^2 units — syllabus/toy convention."""
    return 3.0 * (Om_m * a**-3 + 4.0 * Om_L)


def a_onset(k: float, eps: float) -> float:
    """Trigger r_prop^3 R(a) > eps with r_prop = a*(2pi/k)/(c/H0):
    3 * rc^3 * (Om_m + 4 Om_L a^3) > eps,  rc = (2pi/k)/(c/H0)  (dimensionless).
    Returns onset a in (0, inf); np.inf if never, 0.0 if always."""
    rc = (2.0 * np.pi / k) / C_over_H0_Mpc_h
    base = 3.0 * rc**3
    if base * Om_m >= eps:            # already on at a -> 0
        return 0.0
    a3 = (eps / base - Om_m) / (4.0 * Om_L)
    return a3 ** (1.0 / 3.0)


# ----------------------------- growth ODE --------------------------------
def grow(k: float, gamma0: float, eps: float, a_eval: np.ndarray) -> np.ndarray:
    """Integrate delta(a) from A_INIT with EdS growing-mode ICs; returns
    delta at a_eval (normalized delta(A_INIT)=A_INIT)."""
    a_on = a_onset(k, eps) if gamma0 > 0 else np.inf

    def rhs(a, y):
        d, dp = y
        fric = 3.0 / a + dlnH_da(a)
        if a >= a_on:
            fric += gamma0 / a
        return [dp, -fric * dp + 1.5 * Omega_m_a(a) / a**2 * d]

    sol = solve_ivp(rhs, (A_INIT, 1.0), [A_INIT, 1.0],
                    t_eval=a_eval, method="LSODA", rtol=1e-9, atol=1e-14)
    if not sol.success:
        raise RuntimeError(f"ODE failed k={k} g={gamma0} eps={eps}: {sol.message}")
    return sol.y[0]


# ----------------------------- CAMB baseline ------------------------------
def camb_baseline():
    """Return (results, As_used) with sigma8(0) tuned to SIGMA8_TARGET."""
    As = 2.10e-9
    res = None
    for _ in range(2):  # one rescale pass is enough (linear in As)
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=MNU, omk=0)
        pars.InitPower.set_params(As=As, ns=ns)
        pars.set_matter_power(redshifts=[0.0], kmax=5.0)
        res = camb.get_results(pars)
        s8 = res.get_sigma8_0()
        if abs(s8 - SIGMA8_TARGET) / SIGMA8_TARGET < 1e-4:
            break
        As *= (SIGMA8_TARGET / s8) ** 2
    return res, As


def camb_growth_ratio(As: float, zs: np.ndarray) -> np.ndarray:
    """CAMB growth proxy at K_GATE: sqrt(P(k_gate,z)/P(k_gate,0))."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=MNU, omk=0)
    pars.InitPower.set_params(As=As, ns=ns)
    zlist = sorted(set(zs.tolist()) | {0.0})
    pars.set_matter_power(redshifts=zlist, kmax=1.0)
    res = camb.get_results(pars)
    kh, z_out, pk = res.get_matter_power_spectrum(minkh=K_GATE * 0.9,
                                                  maxkh=K_GATE * 1.1, npoints=3)
    ik = np.argmin(np.abs(kh - K_GATE))
    z_out = np.asarray(z_out)
    p_gate = pk[:, ik]
    p0 = p_gate[np.argmin(np.abs(z_out - 0.0))]
    ratio = np.sqrt(p_gate / p0)
    # map requested zs onto CAMB's output ordering
    out = np.array([ratio[np.argmin(np.abs(z_out - z))] for z in zs])
    return out


# ------------------------------- main -------------------------------------
def main() -> int:
    # Windows console is cp1252 by default — force UTF-8 so the Greek in the
    # summary can't kill the run after the science is done (first-run lesson).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 74)
    print("script5 — Tier 1 k-dependent growth (mode-dependent trigger, corrected"
          " baseline)")
    print("=" * 74)
    print(f"Planck18: H0={H0} Om_m={Om_m:.5f} Om_L={Om_L:.5f} Om_r={Om_r:.3e} "
          f"ns={ns} mnu(CAMB)={MNU}")

    # ---- CAMB baseline & sigma8 tune
    res0, As = camb_baseline()
    print(f"CAMB As tuned: {As:.4e}  -> sigma8(0) = {res0.get_sigma8_0():.4f} "
          f"(target {SIGMA8_TARGET})")

    # ---- BASELINE GATE
    z_gate = np.linspace(0.0, Z_GATE_MAX, 16)
    a_gate = 1.0 / (1.0 + z_gate)
    a_eval = np.sort(a_gate)
    d_lcdm_gate = grow(K_GATE, 0.0, 1.0, a_eval)
    # order back to z_gate
    D_ode = np.array([d_lcdm_gate[np.argmin(np.abs(a_eval - a))] for a in a_gate])
    D_ode_ratio = D_ode / D_ode[0]                      # D(z)/D(0)
    D_camb_ratio = camb_growth_ratio(As, z_gate)
    resid = D_ode_ratio / D_camb_ratio - 1.0
    gate_max = float(np.max(np.abs(resid)))
    gate_pass = gate_max < GATE_TOL
    print(f"BASELINE GATE @k={K_GATE} h/Mpc, z in [0,{Z_GATE_MAX:.0f}]: "
          f"max |ODE/CAMB - 1| = {gate_max:.3e}  -> "
          f"{'PASS' if gate_pass else 'FAIL'} (tol {GATE_TOL})")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(z_gate, resid * 100, "o-", color="teal")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.5, color="crimson", ls="--", lw=0.8)
    ax.axhline(-0.5, color="crimson", ls="--", lw=0.8)
    ax.set_xlabel("z"); ax.set_ylabel("ODE/CAMB growth ratio − 1  [%]")
    ax.set_title(f"Baseline gate (Γ₀=0): max dev {gate_max*100:.3f}% — "
                 f"{'PASS' if gate_pass else 'FAIL'}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig5a_baseline_gate.{ext}", dpi=150)
    plt.close(fig)

    if not gate_pass:
        print("GATE FAILED — no friction results reported (Step-0 discipline). "
              "Fallback: rerun with MNU=0 or refine ICs.")
        return 2

    # ---- scan grids
    k_grid = np.logspace(-3, 0, 120)                    # h/Mpc
    gamma_grid = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0])
    # eps chosen to sweep onset epochs for k ~ 0.02-0.2:
    eps_grid = np.logspace(-6, 2, 9)
    a_out = np.linspace(A_INIT, 1.0, 220)
    z_f = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    K_REF = 0.1

    # LCDM reference growth (k-independent in this nu-less ODE)
    d_ref = grow(K_REF, 0.0, 1.0, a_out)
    D0_ref = d_ref[-1]

    scan = {"meta": {
        "H0": H0, "Om_m": Om_m, "Om_L": Om_L, "Om_r": Om_r, "ns": ns,
        "As": As, "sigma8_0": SIGMA8_TARGET, "mnu_camb": MNU,
        "gate_max_dev": gate_max, "gate_tol": GATE_TOL,
        "trigger": "proper r = a*(2pi/k); on where 3*rc^3*(Om_m+4*Om_L*a^3) > eps",
        "friction": "Gamma0/a added to delta' coefficient while triggered",
        "baseline": "delta''+[3/a+dlnH/da]delta'-(3/2a^2)Om_m(a)delta=0 "
                    "(EdS -> 3/(2a): Deep Think eq-22 corrected)"},
        "runs": []}

    # ---- Delta P / P (k) at z=0, for Gamma0=6 (syllabus) across eps family,
    #      plus the sigma8-suppression (Gamma0, eps) heatmap at k_ref window.
    print("\nscanning...")
    heat = np.zeros((len(gamma_grid), len(eps_grid)))
    dpp_family = {}
    for j, eps in enumerate(eps_grid):
        # k-resolved for the family figure at syllabus Gamma0=6
        ratio_k = np.array([grow(k, 6.0, eps, a_out)[-1] / D0_ref
                            for k in k_grid])
        dpp_family[eps] = ratio_k**2 - 1.0
        for i, g0 in enumerate(gamma_grid):
            Dk = grow(K_REF, g0, eps, a_out)[-1] / D0_ref
            heat[i, j] = Dk**2 - 1.0        # ~ Delta P/P at k_ref (z=0)
            zon = (1.0 / a_onset(K_REF, eps) - 1.0) if a_onset(K_REF, eps) > 0 else np.inf
            scan["runs"].append({"gamma0": g0, "eps": eps, "k_ref": K_REF,
                                 "z_onset_kref": None if not np.isfinite(zon) else zon,
                                 "dP_over_P_kref_z0": heat[i, j]})
        aon = a_onset(K_REF, eps)
        if aon == 0.0:
            lbl = "always-on"
        elif aon >= 1.0:
            lbl = f"onset after today at k=0.1 (a_on={aon:.2f}); larger scales active"
        else:
            lbl = f"z_onset = {1/aon - 1:.2f}"
        print(f"  eps={eps:.1e} done ({lbl})")

    # fig 5b — Delta P/P (k)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for eps in eps_grid:
        aon = a_onset(K_REF, eps)
        lbl_z = f"z_on(k=0.1)={1/aon-1:.1f}" if 0 < aon < 1 else \
                ("always-on" if aon == 0 else "never@k=0.1")
        ax.semilogx(k_grid, dpp_family[eps] * 100, lw=1.4,
                    label=f"ε={eps:.0e} ({lbl_z})")
    ax.set_xlabel("k  [h/Mpc]"); ax.set_ylabel("ΔP/P at z=0  [%]")
    ax.set_title("Tier 1: k-dependent suppression, Γ₀=6 (syllabus), z=0")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig5b_dP_over_P.{ext}", dpi=150)
    plt.close(fig)

    # fig 5c — heatmap
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(heat * 100, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-100, vmax=100,
                   extent=(np.log10(eps_grid[0]), np.log10(eps_grid[-1]),
                           np.log10(gamma_grid[0]), np.log10(gamma_grid[-1])))
    ax.set_xlabel("log10 ε"); ax.set_ylabel("log10 Γ₀")
    ax.set_title(f"ΔP/P at k={K_REF} h/Mpc, z=0  [%]")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig5c_sigma8_scan.{ext}", dpi=150)
    plt.close(fig)

    # fig 5d — fsigma8(z) at k_ref for representative params (theory only)
    reps = [(0.0, 1.0, "ΛCDM"), (6.0, 1e-2, "Γ₀=6, ε=1e-2"),
            (1.0, 1e-2, "Γ₀=1, ε=1e-2"), (0.1, 1e-2, "Γ₀=0.1, ε=1e-2"),
            (6.0, 1.0, "Γ₀=6, ε=1")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for g0, eps, lbl in reps:
        d = grow(K_REF, g0, eps, a_out)
        lnD = np.log(d); lna = np.log(a_out)
        f = np.gradient(lnD, lna)
        s8_z = SIGMA8_TARGET * d / D0_ref
        zz = 1 / a_out - 1
        m = zz <= 3.0
        ax.plot(zz[m], (f * s8_z)[m], lw=1.5, label=lbl)
    ax.set_xlabel("z"); ax.set_ylabel("fσ₈(z)  [k=0.1 h/Mpc proxy]")
    ax.set_title("Tier 1 fσ₈ (theory; data overlay = Tier 3)")
    ax.invert_xaxis(); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig5d_fsigma8.{ext}", dpi=150)
    plt.close(fig)

    (OUTDIR / "script5_scan_results.json").write_text(
        json.dumps(scan, indent=1), encoding="utf-8")

    # ---- console verdict summary
    print("\n" + "=" * 74)
    print("TIER-1 SUMMARY")
    print("=" * 74)
    print(f"gate: PASS ({gate_max*100:.3f}% max dev)")
    sub = heat[gamma_grid >= 1.0][:, :]  # strong-friction rows
    print(f"syllabus corner (Γ₀=6): ΔP/P(k=0.1,z=0) spans "
          f"[{dpp_family[eps_grid[0]][np.argmin(np.abs(k_grid-0.1))]*100:+.1f}%, "
          f"{dpp_family[eps_grid[-1]][np.argmin(np.abs(k_grid-0.1))]*100:+.1f}%] over ε grid")
    # survival heuristic: |ΔP/P| at k_ref within ~[1%, 15%] — big enough to
    # matter for DESI full-shape, small enough not to wreck sigma8.
    window = (np.abs(heat) * 100 >= 1.0) & (np.abs(heat) * 100 <= 15.0)
    n_win = int(window.sum())
    print(f"parameter points in the 'interesting window' (1%–15% |ΔP/P| @k=0.1): "
          f"{n_win} of {window.size}")
    for i, g0 in enumerate(gamma_grid):
        for j, eps in enumerate(eps_grid):
            if window[i, j]:
                print(f"   Γ₀={g0:<5g} ε={eps:.0e}  ΔP/P={heat[i,j]*100:+.1f}%")
    print("\nfigures: fig5a_baseline_gate, fig5b_dP_over_P, fig5c_sigma8_scan, "
          "fig5d_fsigma8  (+ script5_scan_results.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
