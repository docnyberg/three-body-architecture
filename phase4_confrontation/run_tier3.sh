#!/usr/bin/env bash
# run_tier3.sh — Tier-3 CLASS runs: guarded S(k) at the LRG3 bin redshift.
# z_pk = 0.95, 0  (z_eff ~ LRG 0.8-1.1 bin; z=0 kept for continuity checks).
# Cases: base + {A:0.03,1e-6} {B:0.1,1e-5} {D:0.01,1e-6} {E:0.003,1e-6} {F:0.01,1e-5}
set -euo pipefail

WIN='/mnt/c/Users/docny/Projects/THREE_BODY/SOVEREIGN_PRACTICE/holographic_DP_scripts_Spontaneous Localization'
CP="$HOME/class_public"
OUT="$CP/geo_runs"
cd "$CP"

# template with two output redshifts
tr -d '\r' < "$WIN/geo_base.ini" | sed 's/^z_pk = .*/z_pk = 0.95, 0./' > "$OUT/_t3_template.ini"
grep -q "z_pk = 0.95" "$OUT/_t3_template.ini" || { echo "template sed failed"; exit 2; }

run_case () {  # name gamma eps  (all guarded; gamma=OFF => vanilla env)
  local name="$1" gamma="$2" eps="$3"
  cp "$OUT/_t3_template.ini" "$OUT/$name.ini"
  echo "root = geo_runs/${name}_" >> "$OUT/$name.ini"
  echo "--- run $name (GEO_GAMMA0=$gamma GEO_EPS=$eps SUBHORIZON=1) ---"
  if [ "$gamma" = "OFF" ]; then
    env -u GEO_GAMMA0 -u GEO_EPS -u GEO_SUBHORIZON ./class "$OUT/$name.ini" > "$OUT/${name}.log" 2>&1
  else
    GEO_GAMMA0="$gamma" GEO_EPS="$eps" GEO_SUBHORIZON=1 ./class "$OUT/$name.ini" > "$OUT/${name}.log" 2>&1
  fi
}

run_case t3base OFF -
run_case t3A 0.03  1e-6
run_case t3B 0.1   1e-5
run_case t3D 0.01  1e-6
run_case t3E 0.003 1e-6
run_case t3F 0.01  1e-5

DEST="$WIN/class_tier2"
cp "$OUT"/t3*_pk.dat "$OUT"/t3*_cl.dat "$OUT"/t3*.log "$DEST"/ 2>/dev/null || true
ls "$DEST" | grep t3 | head -30
echo "DONE"
