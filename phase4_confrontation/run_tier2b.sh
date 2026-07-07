#!/usr/bin/env bash
# run_tier2b.sh — sub-horizon-guard rerun (v2 patch).
# restore vanilla -> apply v2 patch -> rebuild -> REGRESSION GATE (unguarded
# caseB must reproduce stored v1 caseB numerically) -> guarded A/B/C -> collect.
set -euo pipefail

WIN='/mnt/c/Users/docny/Projects/THREE_BODY/SOVEREIGN_PRACTICE/holographic_DP_scripts_Spontaneous Localization'
CP="$HOME/class_public"
OUT="$CP/geo_runs"

echo "=== [1/6] restore vanilla perturbations.c ==="
git -C "$CP" checkout -- source/perturbations.c

echo "=== [2/6] apply v2 patch ==="
python3 "$WIN/class_tier2_patch.py" "$CP"

echo "=== [3/6] rebuild ==="
cd "$CP"
make -j4 class 2>&1 | tail -1

mkdir -p "$OUT"
tr -d '\r' < "$WIN/geo_base.ini" > "$OUT/_template.ini"

run_case () {  # name gamma eps guard(0|1)
  local name="$1" gamma="$2" eps="$3" guard="$4"
  cp "$OUT/_template.ini" "$OUT/$name.ini"
  echo "root = geo_runs/${name}_" >> "$OUT/$name.ini"
  echo "--- run $name (GEO_GAMMA0=$gamma GEO_EPS=$eps GEO_SUBHORIZON=$guard) ---"
  if [ "$guard" = "1" ]; then
    GEO_GAMMA0="$gamma" GEO_EPS="$eps" GEO_SUBHORIZON=1 ./class "$OUT/$name.ini" > "$OUT/${name}.log" 2>&1
  else
    GEO_GAMMA0="$gamma" GEO_EPS="$eps" ./class "$OUT/$name.ini" > "$OUT/${name}.log" 2>&1
  fi
}

echo "=== [4/6] REGRESSION GATE: v2-unguarded caseB vs stored v1 caseB ==="
run_case caseBv2 0.1 1e-5 0
if cmp -s <(grep -v '^#' "$OUT/caseB_00_pk.dat") <(grep -v '^#' "$OUT/caseBv2_00_pk.dat"); then
  echo "REGRESSION PASS: v2 binary with guard OFF reproduces v1 caseB pk numerically"
else
  echo "REGRESSION FAIL: v2-unguarded differs from v1 — patch v2 broke something. ABORT."
  diff <(grep -v '^#' "$OUT/caseB_00_pk.dat") <(grep -v '^#' "$OUT/caseBv2_00_pk.dat") | head -5
  exit 3
fi

echo "=== [5/6] guarded runs ==="
run_case caseA_sh 0.03 1e-6 1
run_case caseB_sh 0.1  1e-5 1
run_case caseC_sh 6    1    1

echo "=== [6/6] collect ==="
DEST="$WIN/class_tier2"
mkdir -p "$DEST"
cp "$OUT"/caseBv2_*.dat "$OUT"/case*_sh_*.dat "$OUT"/caseBv2.log "$OUT"/case*_sh.log "$DEST"/ 2>/dev/null || true
ls "$DEST" | grep -E "sh|v2" | head -20
echo "DONE"
