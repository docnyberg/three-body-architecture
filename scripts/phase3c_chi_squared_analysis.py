"""
Phase 3C: Chi-Squared DESI BAO Confrontation
=============================================
Statistical comparison of cosmological models against DESI DR1 BAO data.

Models tested:
  1. LCDM (Planck 2018 best-fit)
  2. w0-wa CPL dark energy (grid scan to find best-fit)
  3. Three-Body Architecture (geometric friction -> effective H(z) modification)

Data: 12 DESI DR1 BAO measurements with full 12x12 covariance matrix
  - DV/rs at z=0.295 (BGS)
  - DM/rs, DH/rs at z=0.510, 0.706, 0.930, 1.317 (LRG, ELG)
  - DV/rs at z=1.491 (QSO)
  - DM/rs, DH/rs at z=2.330 (Lya)

Physics of the Three-Body modification:
  The geometric friction Gamma_geo = beta/a activates when the Ricci scalar
  R < R_threshold (z < 24). This modifies both perturbation growth AND the
  effective expansion rate. At the background level, the additional friction
  maps to a modified Friedmann equation:

    H_eff(z) = H_LCDM(z) * sqrt(1 + alpha * g(z))

  where alpha is the dimensionless coupling strength and g(z) is the
  activation profile. We treat alpha as a free parameter and fit to DESI.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize
import camb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
DESI_DIR = os.path.join(os.path.dirname(RESULTS_DIR), 'desi_data', 'bao_data')

c_km_s = 299792.458  # speed of light [km/s]

# Planck 2018 best-fit parameters
H0_PLANCK = 67.36     # km/s/Mpc
OM_PLANCK = 0.3153    # Omega_matter
OL_PLANCK = 0.6847    # Omega_Lambda

# Three-Body onset redshift (from Phase 2B CAMB computation)
Z_ONSET = 24.125


# ============================================================
# DATA LOADING
# ============================================================

def load_desi_dr1():
    """Load DESI DR1 BAO measurements and full covariance matrix."""
    data_file = os.path.join(DESI_DIR, 'desi_2024_gaussian_bao_ALL_GCcomb_mean.txt')
    cov_file = os.path.join(DESI_DIR, 'desi_2024_gaussian_bao_ALL_GCcomb_cov.txt')

    z_vals, values, quantities = [], [], []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            z_vals.append(float(parts[0]))
            values.append(float(parts[1]))
            quantities.append(parts[2])

    cov = np.loadtxt(cov_file)
    cov_inv = np.linalg.inv(cov)

    data_vec = np.array(values)
    n = len(data_vec)

    print(f"  Loaded {n} DESI DR1 BAO measurements")
    print(f"  Covariance matrix: {cov.shape}")
    print(f"  Condition number: {np.linalg.cond(cov):.1f}")

    return z_vals, data_vec, quantities, cov, cov_inv


def get_sound_horizon():
    """Get the sound horizon at drag epoch from CAMB (Planck 2018)."""
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0_PLANCK, ombh2=0.02237, omch2=0.1200,
        mnu=0.06, omk=0, tau=0.0544
    )
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)
    pars.set_matter_power(redshifts=[0], kmax=2.0)
    results = camb.get_results(pars)
    rs = results.get_derived_params()['rdrag']
    print(f"  Sound horizon (CAMB): rs = {rs:.2f} Mpc")
    return rs


# ============================================================
# COSMOLOGICAL DISTANCE COMPUTATIONS
# ============================================================

def H_lcdm(z, H0=H0_PLANCK, Om=OM_PLANCK):
    """Hubble parameter for flat LCDM."""
    OL = 1.0 - Om
    return H0 * np.sqrt(Om * (1 + z)**3 + OL)


def H_w0wa(z, H0=H0_PLANCK, Om=OM_PLANCK, w0=-1.0, wa=0.0):
    """Hubble parameter for flat w0-wa (CPL) dark energy."""
    ODE = 1.0 - Om
    a = 1.0 / (1.0 + z)
    # DE density: ODE * a^{-3(1+w0+wa)} * exp(-3*wa*(1-a))
    de_density = ODE * (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    val = H0**2 * (Om * (1 + z)**3 + de_density)
    if val < 0:
        return 1e10  # unphysical region
    return np.sqrt(val)


def H_threebody(z, H0=H0_PLANCK, Om=OM_PLANCK, alpha=0.0):
    """
    Hubble parameter for Three-Body Architecture.

    The geometric friction modifies the effective expansion rate:
      H_eff(z) = H_LCDM(z) * sqrt(1 + alpha * g(z))

    where g(z) is the activation profile:
      g(z) = (1+z)/(1+z_onset) for z < z_onset, 0 otherwise

    The (1+z) scaling comes from the Gamma_geo ~ 1/a dependence.
    Normalized by (1+z_onset) so g(z_onset) = 1.
    """
    H_base = H_lcdm(z, H0, Om)
    if z > Z_ONSET:
        return H_base
    g_z = (1 + z) / (1 + Z_ONSET)
    return H_base * np.sqrt(1.0 + alpha * g_z)


def compute_distances(H_func, z_arr, quantities, rs, **kwargs):
    """
    Compute theory prediction vector for given H(z) function.
    Returns DM/rs, DH/rs, or DV/rs matching the data ordering.
    """
    # Cache distances at unique redshifts
    unique_z = sorted(set(z_arr))
    dist_cache = {}

    for z in unique_z:
        Hz = H_func(z, **kwargs)
        DH = c_km_s / Hz

        integrand = lambda zp: c_km_s / H_func(zp, **kwargs)
        DM, _ = quad(integrand, 0, z, limit=200, epsrel=1e-10)

        DV = (z * DM**2 * DH)**(1.0 / 3.0)

        dist_cache[z] = {
            'DM_over_rs': DM / rs,
            'DH_over_rs': DH / rs,
            'DV_over_rs': DV / rs
        }

    # Build theory vector in data ordering
    theory = np.array([dist_cache[z][q] for z, q in zip(z_arr, quantities)])
    return theory


def chi2(data_vec, theory_vec, cov_inv):
    """Compute chi-squared statistic."""
    diff = data_vec - theory_vec
    return float(diff @ cov_inv @ diff)


# ============================================================
# MODEL 1: LCDM
# ============================================================

def compute_lcdm_chi2(z_arr, data_vec, quantities, cov_inv, rs):
    """Compute chi-squared for standard LCDM."""
    print("\n" + "=" * 70)
    print("MODEL 1: Standard LCDM (Planck 2018)")
    print("=" * 70)

    theory = compute_distances(H_lcdm, z_arr, quantities, rs,
                               H0=H0_PLANCK, Om=OM_PLANCK)

    chi2_val = chi2(data_vec, theory, cov_inv)
    ndof = len(data_vec)  # 12 data points, 0 free params for fixed Planck

    print(f"\n  chi2 = {chi2_val:.2f}")
    print(f"  N_data = {ndof}")
    print(f"  chi2/N = {chi2_val / ndof:.3f}")

    # Print per-point residuals
    print("\n  Per-point residuals:")
    errors = np.sqrt(np.diag(np.linalg.inv(cov_inv)))  # approximate
    for i, (z, q, d, t) in enumerate(zip(z_arr, quantities, data_vec, theory)):
        resid_pct = (d - t) / t * 100
        print(f"    [{i:2d}] z={z:.3f} {q:12s}: data={d:.4f}  theory={t:.4f}  "
              f"resid={resid_pct:+.2f}%")

    return chi2_val, theory


# ============================================================
# MODEL 2: w0-wa Grid Scan
# ============================================================

def scan_w0wa(z_arr, data_vec, quantities, cov_inv, rs):
    """Scan w0-wa parameter space to find best-fit."""
    print("\n" + "=" * 70)
    print("MODEL 2: w0-wa CPL Dark Energy (Grid Scan)")
    print("=" * 70)

    # Coarse grid
    w0_range = np.linspace(-1.8, -0.2, 33)
    wa_range = np.linspace(-3.5, 1.5, 41)

    chi2_grid = np.full((len(w0_range), len(wa_range)), np.inf)

    print(f"  Scanning {len(w0_range)} x {len(wa_range)} = "
          f"{len(w0_range) * len(wa_range)} points...")

    for i, w0 in enumerate(w0_range):
        for j, wa in enumerate(wa_range):
            try:
                theory = compute_distances(H_w0wa, z_arr, quantities, rs,
                                           H0=H0_PLANCK, Om=OM_PLANCK,
                                           w0=w0, wa=wa)
                if np.all(np.isfinite(theory)) and np.all(theory > 0):
                    chi2_grid[i, j] = chi2(data_vec, theory, cov_inv)
            except Exception:
                pass

        if (i + 1) % 10 == 0:
            print(f"    ... {i + 1}/{len(w0_range)} rows done")

    # Find minimum
    idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    w0_best = w0_range[idx[0]]
    wa_best = wa_range[idx[1]]
    chi2_min = chi2_grid[idx]

    print(f"\n  Best-fit: w0 = {w0_best:.3f}, wa = {wa_best:.3f}")
    print(f"  chi2_min = {chi2_min:.2f}")
    print(f"  chi2/N = {chi2_min / len(data_vec):.3f}")

    # Refine with scipy
    print("  Refining with Nelder-Mead...")

    def neg_chi2_w0wa(params):
        w0, wa = params
        try:
            theory = compute_distances(H_w0wa, z_arr, quantities, rs,
                                       H0=H0_PLANCK, Om=OM_PLANCK,
                                       w0=w0, wa=wa)
            if np.all(np.isfinite(theory)) and np.all(theory > 0):
                return chi2(data_vec, theory, cov_inv)
        except Exception:
            pass
        return 1e10

    result = minimize(neg_chi2_w0wa, [w0_best, wa_best], method='Nelder-Mead',
                      options={'xatol': 0.001, 'fatol': 0.01})

    if result.fun < chi2_min:
        w0_best, wa_best = result.x
        chi2_min = result.fun
        print(f"  Refined: w0 = {w0_best:.4f}, wa = {wa_best:.4f}")
        print(f"  chi2_min = {chi2_min:.3f}")

    theory_best = compute_distances(H_w0wa, z_arr, quantities, rs,
                                    H0=H0_PLANCK, Om=OM_PLANCK,
                                    w0=w0_best, wa=wa_best)

    return chi2_min, w0_best, wa_best, chi2_grid, w0_range, wa_range, theory_best


# ============================================================
# MODEL 3: Three-Body Architecture
# ============================================================

def scan_threebody(z_arr, data_vec, quantities, cov_inv, rs):
    """Scan the Three-Body friction coupling strength alpha."""
    print("\n" + "=" * 70)
    print("MODEL 3: Three-Body Architecture (Geometric Friction)")
    print("=" * 70)
    print(f"  Onset redshift: z = {Z_ONSET:.1f}")
    print(f"  Profile: H_eff = H_LCDM * sqrt(1 + alpha * (1+z)/(1+z_onset))")

    # Scan alpha
    alpha_range = np.linspace(-0.5, 0.5, 201)
    chi2_alpha = np.full_like(alpha_range, np.inf)

    for i, alpha in enumerate(alpha_range):
        try:
            theory = compute_distances(H_threebody, z_arr, quantities, rs,
                                       H0=H0_PLANCK, Om=OM_PLANCK, alpha=alpha)
            if np.all(np.isfinite(theory)) and np.all(theory > 0):
                chi2_alpha[i] = chi2(data_vec, theory, cov_inv)
        except Exception:
            pass

    # Find minimum
    idx_best = np.argmin(chi2_alpha)
    alpha_best = alpha_range[idx_best]
    chi2_best = chi2_alpha[idx_best]

    # Refine
    def chi2_func(alpha):
        try:
            theory = compute_distances(H_threebody, z_arr, quantities, rs,
                                       H0=H0_PLANCK, Om=OM_PLANCK, alpha=alpha)
            if np.all(np.isfinite(theory)) and np.all(theory > 0):
                return chi2(data_vec, theory, cov_inv)
        except Exception:
            pass
        return 1e10

    result = minimize_scalar(chi2_func, bounds=(alpha_best - 0.05, alpha_best + 0.05),
                             method='bounded')
    if result.fun < chi2_best:
        alpha_best = result.x
        chi2_best = result.fun

    print(f"\n  Best-fit alpha = {alpha_best:.6f}")
    print(f"  chi2 = {chi2_best:.3f}")
    print(f"  chi2/N = {chi2_best / len(data_vec):.3f}")

    # Physical interpretation
    H_mod_z0 = H_threebody(0, alpha=alpha_best)
    H_mod_z1 = H_threebody(1.0, alpha=alpha_best)
    print(f"\n  H(z=0) modification: {(H_mod_z0/H_lcdm(0) - 1)*100:+.2f}%")
    print(f"  H(z=1) modification: {(H_mod_z1/H_lcdm(1.0) - 1)*100:+.2f}%")

    # Map to effective growth suppression
    # beta_growth = alpha * (effective mapping factor)
    # For the perturbation-level friction Gamma_geo = beta/a,
    # beta relates to alpha via the Friedmann equation consistency:
    # H_eff^2 = H^2 + delta_H^2 => extra friction ~ delta_H / H ~ alpha/2
    beta_eff = alpha_best * H0_PLANCK * 3.0 / 2.0  # rough mapping
    print(f"  Effective perturbation friction beta ~ {beta_eff:.2f}")

    theory_best = compute_distances(H_threebody, z_arr, quantities, rs,
                                    H0=H0_PLANCK, Om=OM_PLANCK, alpha=alpha_best)

    # Per-point residuals
    print("\n  Per-point residuals (Three-Body best-fit):")
    for i, (z, q, d, t) in enumerate(zip(z_arr, quantities, data_vec, theory_best)):
        resid_pct = (d - t) / t * 100
        print(f"    [{i:2d}] z={z:.3f} {q:12s}: data={d:.4f}  theory={t:.4f}  "
              f"resid={resid_pct:+.2f}%")

    return chi2_best, alpha_best, chi2_alpha, alpha_range, theory_best


# ============================================================
# FIGURES
# ============================================================

def make_figures(z_arr, data_vec, quantities, cov,
                 theory_lcdm, theory_w0wa, theory_3body,
                 chi2_lcdm, chi2_w0wa, chi2_3body,
                 w0_best, wa_best, alpha_best,
                 chi2_grid, w0_range, wa_range,
                 chi2_alpha, alpha_range):
    """Generate publication figures."""

    print("\n" + "=" * 70)
    print("Generating Figures")
    print("=" * 70)

    errors = np.sqrt(np.diag(cov))
    paths = []

    # ================================================================
    # Figure 1: w0-wa chi-squared contour
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(9, 7))

    chi2_min_grid = np.min(chi2_grid[chi2_grid < 1e9])
    delta_chi2 = chi2_grid - chi2_min_grid
    delta_chi2[delta_chi2 > 100] = 100

    levels = [2.30, 6.18, 11.83]  # 1sigma, 2sigma, 3sigma for 2 params
    labels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$']

    W0, WA = np.meshgrid(w0_range, wa_range, indexing='ij')

    cs = ax1.contourf(W0, WA, delta_chi2, levels=np.linspace(0, 30, 31),
                      cmap='RdYlBu_r', alpha=0.8)
    plt.colorbar(cs, ax=ax1, label=r'$\Delta\chi^2$')

    for lev, lab in zip(levels, labels):
        ax1.contour(W0, WA, delta_chi2, levels=[lev],
                    colors='black', linewidths=1.5, linestyles='-')

    # Mark key points
    ax1.plot(w0_best, wa_best, 'k*', markersize=15, label=f'Best-fit ({w0_best:.2f}, {wa_best:.2f})')
    ax1.plot(-1.0, 0.0, 'bs', markersize=10, label=r'$\Lambda$CDM ($-1, 0$)')

    # DESI 2024 published best-fit (approximate from their paper)
    ax1.plot(-0.45, -1.79, 'r^', markersize=10, label='DESI 2024 (approx)')

    ax1.set_xlabel(r'$w_0$', fontsize=14)
    ax1.set_ylabel(r'$w_a$', fontsize=14)
    ax1.set_title(r'$\chi^2$ Contours: $w_0$-$w_a$ vs DESI DR1 BAO'
                  '\n(Planck 2018 cosmology, BAO-only)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    path1 = os.path.join(RESULTS_DIR, 'desi_chi2_w0wa_contour.png')
    plt.tight_layout()
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    paths.append(path1)
    print(f"  Saved: {path1}")

    # ================================================================
    # Figure 2: Three-Body alpha scan
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    valid = chi2_alpha < 1e9
    ax2.plot(alpha_range[valid], chi2_alpha[valid], 'b-', linewidth=2)
    ax2.axhline(chi2_lcdm, color='gray', linestyle='--', linewidth=1,
                label=fr'$\Lambda$CDM: $\chi^2 = {chi2_lcdm:.2f}$')
    ax2.axvline(alpha_best, color='red', linestyle=':', linewidth=1,
                label=fr'Best-fit $\alpha = {alpha_best:.4f}$, $\chi^2 = {chi2_3body:.2f}$')
    ax2.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax2.set_xlabel(r'Geometric friction coupling $\alpha$', fontsize=14)
    ax2.set_ylabel(r'$\chi^2$', fontsize=14)
    ax2.set_title('Three-Body Architecture: Friction Coupling vs DESI DR1 BAO\n'
                  r'$H_{\rm eff}(z) = H_{\Lambda\rm CDM}(z)\sqrt{1 + \alpha\,(1+z)/(1+z_{\rm onset})}$',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    path2 = os.path.join(RESULTS_DIR, 'desi_chi2_threebody_alpha.png')
    plt.tight_layout()
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    paths.append(path2)
    print(f"  Saved: {path2}")

    # ================================================================
    # Figure 3: Residual comparison — all three models
    # ================================================================
    fig3, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Separate DM/rs and DH/rs measurements
    dm_idx = [i for i, q in enumerate(quantities) if q == 'DM_over_rs']
    dh_idx = [i for i, q in enumerate(quantities) if q == 'DH_over_rs']
    dv_idx = [i for i, q in enumerate(quantities) if q == 'DV_over_rs']

    def plot_residuals(ax, indices, ylabel, title):
        z_pts = [z_arr[i] for i in indices]
        data_pts = data_vec[indices]
        err_pts = errors[indices]

        # Residuals: (data - theory) / theory * 100
        for theory, label, color, marker in [
            (theory_lcdm, r'$\Lambda$CDM', 'blue', 's'),
            (theory_w0wa, f'$w_0w_a$ best-fit', 'green', '^'),
            (theory_3body, 'Three-Body', 'red', 'o')
        ]:
            th_pts = theory[indices]
            resid = (data_pts - th_pts) / th_pts * 100
            resid_err = err_pts / th_pts * 100
            offset = {'blue': -0.015, 'green': 0, 'red': 0.015}[color]
            z_offset = [z + offset for z in z_pts]
            ax.errorbar(z_offset, resid, yerr=resid_err,
                       fmt=marker, color=color, markersize=8,
                       capsize=4, linewidth=1.5, label=label)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # DM/rs residuals
    plot_residuals(axes[0], dm_idx,
                   r'$({\rm data} - {\rm theory}) / {\rm theory}$ [%]',
                   r'$D_M/r_s$ Residuals')

    # DH/rs residuals
    plot_residuals(axes[1], dh_idx,
                   r'$({\rm data} - {\rm theory}) / {\rm theory}$ [%]',
                   r'$D_H/r_s$ Residuals')

    axes[1].set_xlabel(r'Redshift $z$', fontsize=14)

    fig3.suptitle('DESI DR1 BAO Residuals: Three Models Compared\n'
                  fr'$\Lambda$CDM ($\chi^2={chi2_lcdm:.1f}$)  |  '
                  fr'$w_0w_a$ ($\chi^2={chi2_w0wa:.1f}$)  |  '
                  fr'Three-Body ($\chi^2={chi2_3body:.1f}$)',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path3 = os.path.join(RESULTS_DIR, 'desi_residuals_three_models.png')
    plt.savefig(path3, dpi=200, bbox_inches='tight')
    paths.append(path3)
    print(f"  Saved: {path3}")

    # ================================================================
    # Figure 4: Distance-redshift comparison
    # ================================================================
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

    z_fine = np.linspace(0.05, 3.0, 200)

    for ax, dist_type, title in [
        (ax4a, 'DM_over_rs', r'$D_M / r_s$'),
        (ax4b, 'DH_over_rs', r'$D_H / r_s$')
    ]:
        # Theory curves
        rs = 147.10  # will be passed properly in main
        for H_func, kwargs, label, color, ls in [
            (H_lcdm, {'H0': H0_PLANCK, 'Om': OM_PLANCK},
             r'$\Lambda$CDM', 'blue', '-'),
            (H_w0wa, {'H0': H0_PLANCK, 'Om': OM_PLANCK, 'w0': w0_best, 'wa': wa_best},
             f'$w_0w_a$ ({w0_best:.2f}, {wa_best:.2f})', 'green', '--'),
            (H_threebody, {'H0': H0_PLANCK, 'Om': OM_PLANCK, 'alpha': alpha_best},
             fr'Three-Body ($\alpha={alpha_best:.3f}$)', 'red', ':')
        ]:
            vals = []
            for z in z_fine:
                Hz = H_func(z, **kwargs)
                DH = c_km_s / Hz
                integrand = lambda zp: c_km_s / H_func(zp, **kwargs)
                DM, _ = quad(integrand, 0, z, limit=100)
                if dist_type == 'DM_over_rs':
                    vals.append(DM / rs)
                else:
                    vals.append(DH / rs)
            ax.plot(z_fine, vals, color=color, linestyle=ls, linewidth=2, label=label)

        # Data points
        indices = [i for i, q in enumerate(quantities) if q == dist_type]
        z_pts = [z_arr[i] for i in indices]
        d_pts = data_vec[indices]
        e_pts = errors[indices]
        ax.errorbar(z_pts, d_pts, yerr=e_pts, fmt='ko', markersize=7,
                    capsize=4, linewidth=1.5, label='DESI DR1', zorder=5)

        ax.set_xlabel(r'Redshift $z$', fontsize=14)
        ax.set_ylabel(title, fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig4.suptitle('BAO Distance-Redshift Relations: Data vs Models',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()

    path4 = os.path.join(RESULTS_DIR, 'desi_distance_redshift_models.png')
    plt.savefig(path4, dpi=200, bbox_inches='tight')
    paths.append(path4)
    print(f"  Saved: {path4}")

    plt.close('all')
    return paths


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  PHASE 3C: CHI-SQUARED DESI BAO CONFRONTATION")
    print("=" * 70)

    # Load data
    print("\n--- Loading Data ---")
    z_arr, data_vec, quantities, cov, cov_inv = load_desi_dr1()
    rs = get_sound_horizon()

    n_data = len(data_vec)

    # Model 1: LCDM
    chi2_lcdm, theory_lcdm = compute_lcdm_chi2(
        z_arr, data_vec, quantities, cov_inv, rs)

    # Model 2: w0-wa scan
    chi2_w0wa, w0_best, wa_best, chi2_grid, w0_range, wa_range, theory_w0wa = \
        scan_w0wa(z_arr, data_vec, quantities, cov_inv, rs)

    # Model 3: Three-Body
    chi2_3body, alpha_best, chi2_alpha, alpha_range, theory_3body = \
        scan_threebody(z_arr, data_vec, quantities, cov_inv, rs)

    # ================================================================
    # STATISTICAL COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    delta_chi2_w0wa = chi2_lcdm - chi2_w0wa
    delta_chi2_3body = chi2_lcdm - chi2_3body

    # Significance: delta_chi2 follows chi2 distribution with delta_dof degrees of freedom
    from scipy.stats import chi2 as chi2_dist

    # w0-wa has 2 extra parameters vs LCDM
    p_value_w0wa = 1 - chi2_dist.cdf(delta_chi2_w0wa, df=2)
    sigma_w0wa = np.sqrt(chi2_dist.ppf(1 - p_value_w0wa, df=1)) if p_value_w0wa > 0 else 99

    # Three-Body has 1 extra parameter vs LCDM
    p_value_3body = 1 - chi2_dist.cdf(delta_chi2_3body, df=1)
    sigma_3body = np.sqrt(chi2_dist.ppf(1 - p_value_3body, df=1)) if p_value_3body > 0 else 99

    print(f"""
  Model             | chi2    | chi2/N  | Delta_chi2 | p-value  | Significance
  ------------------|---------|---------|------------|----------|-------------
  LCDM              | {chi2_lcdm:7.2f} | {chi2_lcdm/n_data:7.3f} |     ---    |   ---    |    ---
  w0-wa (best-fit)  | {chi2_w0wa:7.2f} | {chi2_w0wa/n_data:7.3f} | {delta_chi2_w0wa:10.2f} | {p_value_w0wa:.2e} | {sigma_w0wa:.1f} sigma
  Three-Body (best) | {chi2_3body:7.2f} | {chi2_3body/n_data:7.3f} | {delta_chi2_3body:10.2f} | {p_value_3body:.2e} | {sigma_3body:.1f} sigma

  w0-wa best-fit: w0 = {w0_best:.4f}, wa = {wa_best:.4f}
  Three-Body best-fit: alpha = {alpha_best:.6f}
  N_data = {n_data}, rs = {rs:.2f} Mpc
    """)

    # AIC/BIC comparison
    # AIC = chi2 + 2*k, BIC = chi2 + k*ln(N)
    k_lcdm, k_w0wa, k_3body = 0, 2, 1
    ln_n = np.log(n_data)

    aic_lcdm = chi2_lcdm + 2 * k_lcdm
    aic_w0wa = chi2_w0wa + 2 * k_w0wa
    aic_3body = chi2_3body + 2 * k_3body

    bic_lcdm = chi2_lcdm + k_lcdm * ln_n
    bic_w0wa = chi2_w0wa + k_w0wa * ln_n
    bic_3body = chi2_3body + k_3body * ln_n

    print(f"  Information Criteria (lower = better):")
    print(f"  Model             | AIC     | BIC")
    print(f"  ------------------|---------|--------")
    print(f"  LCDM              | {aic_lcdm:7.2f} | {bic_lcdm:7.2f}")
    print(f"  w0-wa             | {aic_w0wa:7.2f} | {bic_w0wa:7.2f}")
    print(f"  Three-Body        | {aic_3body:7.2f} | {bic_3body:7.2f}")

    # ================================================================
    # FIGURES
    # ================================================================
    figure_paths = make_figures(
        z_arr, data_vec, quantities, cov,
        theory_lcdm, theory_w0wa, theory_3body,
        chi2_lcdm, chi2_w0wa, chi2_3body,
        w0_best, wa_best, alpha_best,
        chi2_grid, w0_range, wa_range,
        chi2_alpha, alpha_range
    )

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    output = {
        'n_data': n_data,
        'rs_drag_Mpc': rs,
        'lcdm': {
            'chi2': chi2_lcdm,
            'chi2_per_dof': chi2_lcdm / n_data,
            'n_params': 0
        },
        'w0wa': {
            'chi2': chi2_w0wa,
            'chi2_per_dof': chi2_w0wa / n_data,
            'w0': w0_best,
            'wa': wa_best,
            'n_params': 2,
            'delta_chi2_vs_lcdm': delta_chi2_w0wa,
            'p_value': p_value_w0wa,
            'sigma': sigma_w0wa
        },
        'three_body': {
            'chi2': chi2_3body,
            'chi2_per_dof': chi2_3body / n_data,
            'alpha': alpha_best,
            'z_onset': Z_ONSET,
            'n_params': 1,
            'delta_chi2_vs_lcdm': delta_chi2_3body,
            'p_value': p_value_3body,
            'sigma': sigma_3body
        },
        'information_criteria': {
            'aic': {'lcdm': aic_lcdm, 'w0wa': aic_w0wa, 'three_body': aic_3body},
            'bic': {'lcdm': bic_lcdm, 'w0wa': bic_w0wa, 'three_body': bic_3body}
        },
        'figures': figure_paths
    }

    json_path = os.path.join(RESULTS_DIR, 'chi_squared_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {json_path}")

    print("\n" + "=" * 70)
    print("  PHASE 3C COMPLETE")
    print("=" * 70)
    print(f"""
  KEY RESULTS:
  - LCDM chi2 = {chi2_lcdm:.2f} ({chi2_lcdm/n_data:.2f} per data point)
  - w0-wa improves by Delta_chi2 = {delta_chi2_w0wa:.2f} ({sigma_w0wa:.1f} sigma, 2 params)
  - Three-Body improves by Delta_chi2 = {delta_chi2_3body:.2f} ({sigma_3body:.1f} sigma, 1 param)
  - Three-Body coupling: alpha = {alpha_best:.6f}
  - 4 publication figures generated
    """)

    return output


if __name__ == "__main__":
    main()
