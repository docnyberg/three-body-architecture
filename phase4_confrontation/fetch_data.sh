#!/usr/bin/env bash
# fetch_data.sh — pull the public survey data used by the Phase-4 confrontation.
# DESI DR1 full-shape likelihood bundles (public, no auth) + Planck PR4 lensing.
set -euo pipefail

mkdir -p desi_dr1 planck_pr4

DESI_BASE="https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/data/likelihood"
for Z in "0.4-0.6" "0.6-0.8" "0.8-1.1"; do
  F="likelihood_spectrum-poles-rotated_syst-rotation-hod-photo_LRG_GCcomb_z${Z}_thetacut0.05.h5"
  echo "fetching $F"
  curl -sS -f -o "desi_dr1/$F" "$DESI_BASE/$F"
done

PR4_BASE="https://raw.githubusercontent.com/carronj/planck_PR4_lensing/main/planckpr4lensing/data_pr4"
for F in \
  "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_bandpowers.dat" \
  "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_cov.dat" \
  "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1.dataset" \
  "FFP10_wdipole_lenspotentialCls.dat"; do
  echo "fetching $F"
  curl -sS -f -o "planck_pr4/$F" "$PR4_BASE/$F"
done

echo "DONE — desi_dr1/ ($(ls desi_dr1 | wc -l) files), planck_pr4/ ($(ls planck_pr4 | wc -l) files)"
