"""
Phase 3B: DESI Data Confrontation
===================================
Overlay the Three-Body Architecture predictions against actual DESI DR1
BAO distance measurements.

DESI measures DM/rs (comoving angular diameter distance / sound horizon)
and DH/rs (Hubble distance / sound horizon) at multiple redshifts.

We compute these same quantities from CAMB for both:
1. Standard LCDM (Planck 2018)
2. Three-Body Architecture (with growth modification)

Then overlay on the actual DESI data points with error bars.
"""

import numpy as np
import camb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
DESI_DIR = os.path.join(os.path.dirname(RESULTS_DIR), 'desi_data', 'bao_data')


def load_desi_data():
    """Load all DESI DR1 BAO measurements and covariance matrices."""
    print("=" * 70)
    print("Loading DESI DR1 BAO Measurements")
    print("=" * 70)

    # Parse the ALL combined mean file
    data_file = os.path.join(DESI_DIR, 'desi_2024_gaussian_bao_ALL_GCcomb_mean.txt')
    cov_file = os.path.join(DESI_DIR, 'desi_2024_gaussian_bao_ALL_GCcomb_cov.txt')

    z_vals = []
    values = []
    quantities = []

    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            z_vals.append(float(parts[0]))
            values.append(float(parts[1]))
            quantities.append(parts[2])

    # Load covariance matrix
    cov = np.loadtxt(cov_file)
    errors = np.sqrt(np.diag(cov))

    # Organize by type
    dm_data = {'z': [], 'val': [], 'err': []}
    dh_data = {'z': [], 'val': [], 'err': []}
    dv_data = {'z': [], 'val': [], 'err': []}

    for i, (z, v, q) in enumerate(zip(z_vals, values, quantities)):
        if q == 'DM_over_rs':
            dm_data['z'].append(z)
            dm_data['val'].append(v)
            dm_data['err'].append(errors[i])
        elif q == 'DH_over_rs':
            dh_data['z'].append(z)
            dh_data['val'].append(v)
            dh_data['err'].append(errors[i])
        elif q == 'DV_over_rs':
            dv_data['z'].append(z)
            dv_data['val'].append(v)
            dv_data['err'].append(errors[i])

    for name, d in [('DM/rs', dm_data), ('DH/rs', dh_data), ('DV/rs', dv_data)]:
        for z, v, e in zip(d['z'], d['val'], d['err']):
            print(f"  {name} at z={z:.3f}: {v:.4f} +/- {e:.4f}")

    return dm_data, dh_data, dv_data


def compute_camb_distances():
    """
    Compute theoretical DM/rs, DH/rs, DV/rs from CAMB for standard LCDM.
    """
    print("\n" + "=" * 70)
    print("Computing CAMB Theoretical Distances")
    print("=" * 70)

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=67.36, ombh2=0.02237, omch2=0.1200,
        mnu=0.06, omk=0, tau=0.0544
    )
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)

    # Compute at fine redshift grid
    z_grid = np.linspace(0.01, 3.0, 300)
    pars.set_matter_power(redshifts=[0], kmax=2.0)

    results = camb.get_results(pars)

    # Get the sound horizon at drag epoch
    rs_drag = results.get_derived_params()['rdrag']
    print(f"\n  Sound horizon at drag: rs = {rs_drag:.2f} Mpc")

    # Compute distances at the DESI redshifts
    H0 = 67.36
    c_km_s = 299792.458  # km/s

    # Get background evolution
    bg = camb.get_background(pars)

    # DM(z) = comoving angular diameter distance
    # DH(z) = c / H(z)
    # DV(z) = [z * DM(z)^2 * DH(z)]^(1/3)

    def get_distances(z_arr):
        dm_theory = []
        dh_theory = []
        dv_theory = []
        for z in z_arr:
            # Comoving distance in Mpc
            dm = bg.comoving_radial_distance(z)
            # Hubble parameter at z in km/s/Mpc
            hz = bg.hubble_parameter(z)
            dh = c_km_s / hz

            dm_over_rs = dm / rs_drag
            dh_over_rs = dh / rs_drag
            dv = (z * dm**2 * dh)**(1.0/3.0)
            dv_over_rs = dv / rs_drag

            dm_theory.append(dm_over_rs)
            dh_theory.append(dh_over_rs)
            dv_theory.append(dv_over_rs)

        return np.array(dm_theory), np.array(dh_theory), np.array(dv_theory)

    # Fine grid for smooth curves
    dm_grid, dh_grid, dv_grid = get_distances(z_grid)

    # At specific DESI redshifts
    desi_z = [0.295, 0.510, 0.706, 0.930, 1.317, 1.491, 2.330]
    dm_desi, dh_desi, dv_desi = get_distances(desi_z)

    print(f"\n  Theoretical predictions at DESI redshifts:")
    for z, dm, dh, dv in zip(desi_z, dm_desi, dh_desi, dv_desi):
        print(f"    z={z:.3f}: DM/rs={dm:.4f}, DH/rs={dh:.4f}, DV/rs={dv:.4f}")

    return z_grid, dm_grid, dh_grid, dv_grid, rs_drag, bg


def make_desi_figures(dm_data, dh_data, dv_data, z_grid, dm_grid, dh_grid, dv_grid):
    """Generate the DESI confrontation figures."""

    print("\n" + "=" * 70)
    print("Generating DESI Confrontation Figures")
    print("=" * 70)

    # ================================================================
    # Figure: DM/rs and DH/rs vs redshift — DESI data + LCDM theory
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # DM/rs panel
    ax1.plot(z_grid, dm_grid, 'b-', linewidth=2,
             label=r'$\Lambda$CDM (CAMB, Planck 2018)')
    ax1.errorbar(dm_data['z'], dm_data['val'], yerr=dm_data['err'],
                 fmt='ro', markersize=8, capsize=5, linewidth=2,
                 label='DESI DR1', zorder=5)
    ax1.set_xlabel(r'Redshift $z$', fontsize=14)
    ax1.set_ylabel(r'$D_M / r_s$', fontsize=14)
    ax1.set_title(r'Comoving Distance / Sound Horizon', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # DH/rs panel
    ax2.plot(z_grid, dh_grid, 'b-', linewidth=2,
             label=r'$\Lambda$CDM (CAMB, Planck 2018)')
    ax2.errorbar(dh_data['z'], dh_data['val'], yerr=dh_data['err'],
                 fmt='ro', markersize=8, capsize=5, linewidth=2,
                 label='DESI DR1', zorder=5)
    ax2.set_xlabel(r'Redshift $z$', fontsize=14)
    ax2.set_ylabel(r'$D_H / r_s$', fontsize=14)
    ax2.set_title(r'Hubble Distance / Sound Horizon', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('DESI DR1 BAO Measurements vs Standard $\\Lambda$CDM\n'
                 'Three-Body Architecture: Baseline Comparison',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    path1 = os.path.join(RESULTS_DIR, 'desi_bao_distances.png')
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path1}")

    # ================================================================
    # Figure: Residuals — (DESI - LCDM) / LCDM
    # ================================================================
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

    # Compute residuals for DM
    from scipy.interpolate import interp1d
    dm_interp = interp1d(z_grid, dm_grid, kind='cubic')
    dh_interp = interp1d(z_grid, dh_grid, kind='cubic')

    dm_z = np.array(dm_data['z'])
    dm_val = np.array(dm_data['val'])
    dm_err = np.array(dm_data['err'])
    dm_theory_at_data = dm_interp(dm_z)
    dm_residual = (dm_val - dm_theory_at_data) / dm_theory_at_data * 100
    dm_residual_err = dm_err / dm_theory_at_data * 100

    dh_z = np.array(dh_data['z'])
    dh_val = np.array(dh_data['val'])
    dh_err = np.array(dh_data['err'])
    dh_theory_at_data = dh_interp(dh_z)
    dh_residual = (dh_val - dh_theory_at_data) / dh_theory_at_data * 100
    dh_residual_err = dh_err / dh_theory_at_data * 100

    # DM residuals
    ax3.axhline(0, color='blue', linewidth=1, linestyle='-', alpha=0.5,
                label=r'$\Lambda$CDM')
    ax3.errorbar(dm_z, dm_residual, yerr=dm_residual_err,
                 fmt='ro', markersize=8, capsize=5, linewidth=2,
                 label='DESI DR1')
    ax3.set_xlabel(r'Redshift $z$', fontsize=14)
    ax3.set_ylabel(r'$(D_M^{\rm obs} - D_M^{\rm theory}) / D_M^{\rm theory}$ [%]',
                   fontsize=11)
    ax3.set_title(r'$D_M/r_s$ Residuals', fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 5)

    # DH residuals
    ax4.axhline(0, color='blue', linewidth=1, linestyle='-', alpha=0.5,
                label=r'$\Lambda$CDM')
    ax4.errorbar(dh_z, dh_residual, yerr=dh_residual_err,
                 fmt='ro', markersize=8, capsize=5, linewidth=2,
                 label='DESI DR1')
    ax4.set_xlabel(r'Redshift $z$', fontsize=14)
    ax4.set_ylabel(r'$(D_H^{\rm obs} - D_H^{\rm theory}) / D_H^{\rm theory}$ [%]',
                   fontsize=11)
    ax4.set_title(r'$D_H/r_s$ Residuals', fontsize=13)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 5)

    fig2.suptitle('DESI DR1 BAO Residuals vs Planck 2018 $\\Lambda$CDM\n'
                  'Searching for Systematic Deviations',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path2 = os.path.join(RESULTS_DIR, 'desi_bao_residuals.png')
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path2}")

    plt.close('all')

    # Print residual analysis
    print("\n  --- Residual Analysis ---")
    print("  DM/rs residuals (DESI vs LCDM):")
    for z, r, e in zip(dm_z, dm_residual, dm_residual_err):
        sigma = abs(r / e) if e > 0 else 0
        print(f"    z={z:.3f}: {r:+.2f}% +/- {e:.2f}% ({sigma:.1f} sigma)")

    print("  DH/rs residuals (DESI vs LCDM):")
    for z, r, e in zip(dh_z, dh_residual, dh_residual_err):
        sigma = abs(r / e) if e > 0 else 0
        print(f"    z={z:.3f}: {r:+.2f}% +/- {e:.2f}% ({sigma:.1f} sigma)")

    return [path1, path2], {
        'dm_residuals': list(zip(dm_z.tolist(), dm_residual.tolist(),
                                  dm_residual_err.tolist())),
        'dh_residuals': list(zip(dh_z.tolist(), dh_residual.tolist(),
                                  dh_residual_err.tolist()))
    }


def main():
    print("\n" + "=" * 70)
    print("  PHASE 3B: DESI DATA CONFRONTATION")
    print("=" * 70 + "\n")

    # Step 1: Load DESI data
    dm_data, dh_data, dv_data = load_desi_data()

    # Step 2: Compute CAMB theoretical distances
    z_grid, dm_grid, dh_grid, dv_grid, rs_drag, bg = compute_camb_distances()

    # Step 3: Generate figures and residual analysis
    paths, residuals = make_desi_figures(
        dm_data, dh_data, dv_data,
        z_grid, dm_grid, dh_grid, dv_grid
    )

    # Step 4: Save results
    output = {
        'rs_drag_Mpc': rs_drag,
        'dm_residuals': residuals['dm_residuals'],
        'dh_residuals': residuals['dh_residuals'],
        'figures': paths
    }

    json_path = os.path.join(RESULTS_DIR, 'desi_confrontation_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {json_path}")

    print("\n" + "=" * 70)
    print("  PHASE 3B COMPLETE")
    print("=" * 70)
    print(f"""
  DESI DR1 BAO data loaded: 12 measurements across 7 redshifts
  CAMB LCDM baseline: rs_drag = {rs_drag:.2f} Mpc
  Residual analysis complete — see plots and JSON for details.

  NEXT: Overlay Three-Body modified distance predictions
  and quantify chi-squared improvement.
""")

    return output


if __name__ == "__main__":
    main()
