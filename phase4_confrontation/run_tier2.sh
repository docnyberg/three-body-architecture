#!/usr/bin/env bash
# run_tier2.sh — Tier-2 CLASS runs (patch -> rebuild -> integrity -> 3 cases)
# Executed inside WSL. Artifacts copied back to the Windows-side class_tier2/.
set -euo pipefail

WIN='/mnt/c/Users/docny/Projects/THREE_BODY/SOVEREIGN_PRACTICE/holographic_DP_scripts_Spontaneous Localization'
CP="$HOME/class_public"
OUT="$CP/geo_runs"

echo "=== [1/6] patch ==="
python3 "$WIN/class_tier2_patch.py" "$CP"

echo "=== [2/6] rebuild ==="
cd "$CP"
make -j4 class 2>&1 | tail -2

mkdir -p "$OUT"
tr -d '\r' < "$WIN/geo_base.ini" > "$OUT/_template.ini"

run_case () {  # name gamma eps
  local name="$1" gamma="$2" eps="$3"
  cp "$OUT/_template.ini" "$OUT/$name.ini"
  echo "root = geo_runs/${name}_" >> "$OUT/$name.ini"
  echo "--- run $name (GEO_GAMMA0=$gamma GEO_EPS=$eps) ---"
  if [ "$gamma" = "OFF" ]; then
    env -u GEO_GAMMA0 -u GEO_EPS ./class "$OUT/$name.ini" > "$OUT/${name}.log" 2>&1
  else
    GEO_GAMMA0="$gamma" GEO_EPS="$eps" ./class "$OUT/$name.ini" > "$OUT/${name}.log" 2>&1
  fi
  grep -i "sigma8" "$OUT/${name}.log" | head -2 || true
}

echo "=== [3/6] baseline (env unset) ==="
run_case base OFF -

echo "=== [4/6] patch-integrity: GEO_GAMMA0=0 must equal baseline ==="
run_case zero 0 0
# CLASS embeds root+timestamp in .dat header comments — compare numerics only
if cmp -s <(grep -v '^#' "$OUT/base_pk.dat") <(grep -v '^#' "$OUT/zero_pk.dat"); then
  echo "INTEGRITY PASS: Gamma0=0 pk numerically identical to vanilla-env pk"
else
  echo "INTEGRITY FAIL: Gamma0=0 differs from baseline numerically — ABORT"
  diff <(grep -v '^#' "$OUT/base_pk.dat") <(grep -v '^#' "$OUT/zero_pk.dat") | head -5
  exit 3
fi

echo "=== [5/6] cases ==="
run_case caseA 0.03 1e-6      # surviving corner (Tier-1: -9.2% @ k=0.1)
run_case caseB 0.1  1e-5      # surviving corner (Tier-1: -6.3% @ k=0.1)
run_case caseC 6    1         # pushed-cliff ISW probe (invisible in P(k) window)

echo "=== [6/6] collect ==="
DEST="$WIN/class_tier2"
mkdir -p "$DEST"
cp "$OUT"/*_pk.dat "$OUT"/*_cl.dat "$OUT"/*_cl_lensed.dat "$OUT"/*.log "$DEST"/ 2>/dev/null || true
ls -la "$DEST" | head -20
echo "DONE"
