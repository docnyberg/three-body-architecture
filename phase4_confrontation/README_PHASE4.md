# Phase 4 — Four-Tier Cosmological Confrontation

Reproducibility deposit for Section VI of *Spontaneous Localization from
Cosmological Geometry: A Parameter-Free Derivation of the Collapse Kernel and
the Observational Exclusion of Its Cosmological Signature* (B. C. Nyberg, 2026).

Every number in the paper's Table (four-tier confrontation verdicts)
regenerates from a script in this directory run against a public dataset
fetched by `fetch_data.sh`. The campaign is deterministic; outputs
(`script*_output.txt`, `script*_results.json`) are included for byte-level
comparison.

## Result summary

| Γ₀ | ε | DESI joint Δχ² | Planck PR4 Δχ² | verdict |
|---|---|---|---|---|
| 0.003 | 1e-6 | −0.0 | −0.8 | survives (both) |
| 0.01 | 1e-6 | +0.4 | +1.3 | survives (both) |
| 0.03 | 1e-6 | +5.1 | +32.5 | excluded (Planck lensing) |
| 0.01 | 1e-5 | +35.7 | +0.8 | excluded (DESI shape) |
| 0.10 | 1e-5 | +3538 | +249 | excluded (both) |

Surviving region: **Γ₀ ≲ 0.01, ε ≲ 1e-6** — empirically indistinguishable from
ΛCDM by construction; predicts a uniform 1.5–5% C_L^φφ deficit (the Simons
Observatory / CMB-S4 target). An earlier "chirped residual" conjecture is
retracted in the paper (Sec. V): the signature is monotonic, scale-dependent
growth suppression.

## Pipeline

- **Tier 1 — gated growth scan** (`script5_kdep_growth.py`): k-resolved
  integration of the corrected growth equation (EdS friction 3/(2a); full form
  3/a + dlnH/da) with the mode-dependent proper-r trigger
  `3·rc³·(Ω_m + 4Ω_Λ a³) > ε`, `rc = (2π/k)·H0`. Entry gate: Γ₀=0 must
  reproduce CAMB growth to <0.5% (passes at 0.049%; the first configuration
  FAILED the gate at 0.58% from a massive-ν mismatch and reported nothing —
  by design).
- **Tier 2 — full Boltzmann** (`class_tier2_patch.py`, `run_tier2*.sh`,
  `script6`, `script7`): drag applied to θ_cdm in CLASS (commit `e858083`,
  Newtonian gauge), env-var controlled (`GEO_GAMMA0`, `GEO_EPS`,
  `GEO_SUBHORIZON`). Off-state proven **bitwise identical** to vanilla CLASS.
  The causal sub-horizon guard (k > aH) was added under a numerical regression
  gate; it reduced the strongest coupling's low-ℓ TT distortion from 1800% to
  725% without changing any verdict.
- **Tier 3 — DESI DR1 full-shape filter** (`script8_desi_template_filter.py`):
  window-convolved comparison against the published DESI DR1 likelihood
  bundles (P0/P2 vectors, systematic covariances, window matrices), two
  Kaiser-block amplitudes (A0, A2) marginalized by GLS. Note: the bundles'
  theory-side fiducial arrays are zero placeholders; the fiducial is built
  from the guarded CLASS baseline at each bin redshift. Baseline GOF:
  48.9/70, 75.7/70, 75.1/70.
- **Tier 4 — hardening** (`script9`, `script10`): three-bin joint filter;
  ν-restored robustness (verdicts shift ≤19% on near-zero entries, ≤2% on
  decisive ones); formal Planck PR4 lensing bandpower χ² (amplitude method vs
  the release fiducial; null sanity 9.33/9).

## Stated limitations (filter grade)

No EFT counterterms / AP / shot-noise freedom in the DESI filter; mid-bin
effective redshifts (0.5 / 0.7 / 0.95); CDM-only drag; Planck test omits the
PR4 linear correction and exact bin windows. None of these plausibly reverses
a +3538 or +249. A desilike-grade joint likelihood is warranted only on
positive filter-grade signal, which no case showed.

## Reproduction

1. `bash fetch_data.sh` — downloads the DESI DR1 LRG full-shape likelihood
   bundles (public, data.desi.lbl.gov) into `desi_dr1/` and the Planck PR4
   lensing bandpowers/covariance (github.com/carronj/planck_PR4_lensing) into
   `planck_pr4/`.
2. Build CLASS: `git clone https://github.com/lesgourg/class_public && cd
   class_public && git checkout e858083 && make -j4 class`. Apply the patch:
   `python3 class_tier2_patch.py <class_public_dir>`; rebuild.
3. Run `run_tier4.sh` (adjusting the two path variables at the top) to produce
   the guarded ν-less + ν-restored spectra, then `script9` and `script10`.
   Tier-1/2/3 scripts reproduce the earlier stages.

Environment used: Python 3.13 (numpy, scipy, matplotlib, h5py, camb 1.6.6),
CLASS at `e858083` under WSL2 Ubuntu 24.04 / gcc 13.3.

## Data credits

DESI DR1 Full Shape and BAO clustering VAC (DESI Collaboration,
arXiv:2411.12021); Planck PR4 lensing (Carron, Mirmelstein & Lewis,
arXiv:2206.07773). This deposit redistributes no survey data; all data are
fetched from the official public sources.
