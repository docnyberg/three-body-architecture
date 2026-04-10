# Three-Body Architecture

**Spontaneous Localization from Cosmological Geometry: A Parameter-Free Derivation of the Collapse Kernel, Thermodynamic Viability, and Falsifiable Cosmological Signature**

Brian C. Nyberg, M.D.
Independent Researcher, Cedar City, Utah

---

## Overview

This repository contains the complete computational verification, manuscript source, and companion philosophical essay for the Three-Body Architecture — a geometric framework linking quantum wavefunction collapse to cosmological expansion.

### Key Results

- The collapse kernel is derived from high-dimensional measure concentration (no free parameters in the microscopic dynamics)
- The critical localization scale matches the electron Compton wavelength: sigma_crit = 3.72 x 10^-13 m
- DESI DR1 BAO confrontation yields Delta-chi2 = 4.28 improvement over Lambda-CDM with a single parameter
- Onset redshift z ~ 24 (Cosmic Dawn) emerges from the Ricci scalar threshold

## Repository Structure

```
manuscript/          LaTeX source (RevTeX4-2, PRD format)
scripts/             Computational verification (Python)
  phase1_*.py        Real-J Reconstruction Theorem
  phase2_*.py        Thermodynamic balance + CAMB integration
  phase3_*.py        Boltzmann integration (chirped residual)
  phase3b_*.py       DESI DR1 BAO baseline
  phase3c_*.py       Chi-squared confrontation (LCDM vs w0wa vs Three-Body)
results/             Numerical results (JSON)
figures/             Publication figures (PNG)
companion_essay/     "Resonance of the Subtended Heart in the Relational Field"
```

## Companion Essay

*Resonance of the Subtended Heart in the Relational Field* is a philosophical companion to the physics paper. It addresses the method by which the framework was derived, the nature of AI-amplified discovery, and the relationship between the observer and the observed.

Available in this repository as Markdown, HTML, and PDF.

## Computational Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib
- CAMB (`pip install camb`)
- DESI DR1 BAO data (Cobaya likelihood repository)

## Citation

```bibtex
@article{Nyberg2026,
  author  = {Nyberg, Brian C.},
  title   = {Spontaneous Localization from Cosmological Geometry},
  journal = {Submitted to Physical Review D},
  year    = {2026}
}
```

## Acknowledgments

This work was developed through sustained engagement with the foundational literature and computational verification using AI systems (Anthropic Claude, Google Gemini). The author gratefully acknowledges the open, published work of Ghirardi, Rimini, Weber, Pearle, Popescu, Winter, Levy, Busch, Adler, and the DESI and LISA Pathfinder collaborations.

All physical claims and interpretations are the sole responsibility of the author.

## License

Copyright 2026 Brian C. Nyberg, M.D. All rights reserved.

The computational scripts in `scripts/` are released under the MIT License for reproducibility purposes. The manuscript and companion essay are copyrighted works; permission to redistribute requires written consent from the author.
