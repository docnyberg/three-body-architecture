#!/usr/bin/env bash
# run_tier4.sh — referee-hardening CLASS runs (Decider-ratified 2026-07-06).
#   Set 1 (t4*):  nu-less pair,   z_pk = 0.5, 0.7, 0.95, 0   (3-bin joint filter)
#   Set 2 (t4n*): nu-RESTORED (N_ncdm=1, m=0.06 eV, N_ur=2.0328), same z_pk
# All guarded (GEO_SUBHORIZON=1). Bin z_eff approximated by mid-bin values
# 0.5 / 0.7 / 0.95 — stated filter-grade approximation (S varies slowly in z).
set -euo pipefail

WIN='/mnt/c/Users/docny/Projects/THREE_BODY/SOVEREIGN_PRACTICE/holographic_DP_scripts_Spontaneous Localization'
CP="$HOME/class_public"
OUT="$CP/geo_runs"
cd "$CP"

tr -d '\r' < "$WIN/geo_base.ini" | sed 's/^z_pk = .*/z_pk = 0.5, 0.7, 0.95, 0./' > "$OUT/_t4_template.ini"
# nu-restored template
sed -e 's/^N_ur = .*/N_ur = 2.0328/' -e 's/^N_ncdm = .*/N_ncdm = 1\nm_ncdm = 0.06/' \
    "$OUT/_t4_template.ini" > "$OUT/_t4n_template.ini"

run_case () {  # template name gamma eps
  local tmpl="$1" name="$2" gamma="$3" eps="$4"
  cp "$OUT/$tmpl" "$OUT/$name.ini"
  echo "root = geo_runs/${name}_" >> "$OUT/$name.ini"
  echo "--- $name (G=$gamma eps=$eps tmpl=$tmpl) ---"
  if [ "$gamma" = "OFF" ]; then
    env -u GEO_GAMMA0 -u GEO_EPS -u GEO_SUBHORIZON ./class "$OUT/$name.ini" > "$OUT/${name}.log" 2>&1
  else
    GEO_GAMMA0="$gamma" GEO_EPS="$eps" GEO_SUBHORIZON=1 ./class "$OUT/$name.ini" > "$OUT/${name}.log" 2>&1
  fi
}

for spec in "t4base OFF -" "t4A 0.03 1e-6" "t4B 0.1 1e-5" "t4D 0.01 1e-6" "t4E 0.003 1e-6" "t4F 0.01 1e-5"; do
  run_case _t4_template.ini $spec
done
for spec in "t4nbase OFF -" "t4nA 0.03 1e-6" "t4nB 0.1 1e-5" "t4nD 0.01 1e-6" "t4nE 0.003 1e-6" "t4nF 0.01 1e-5"; do
  run_case _t4n_template.ini $spec
done

DEST="$WIN/class_tier2"
cp "$OUT"/t4*_pk.dat "$OUT"/t4*_cl.dat "$OUT"/t4*.log "$DEST"/ 2>/dev/null || true
echo "collected: $(ls "$DEST" | grep -c '^t4')"
echo "DONE"
