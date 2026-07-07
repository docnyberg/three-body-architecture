#!/usr/bin/env python3
"""
class_tier2_patch.py — inject the GEO_FRICTION drag into class_public.

Tier 2 of PHASE3_BOLTZMANN_DESI_SCOPE_2026-07-06.md. Minimal-diff, env-var
controlled (no input.c plumbing):

    GEO_GAMMA0      drag strength Gamma0 (unset/0 => binary behaves vanilla)
    GEO_EPS         trigger threshold eps
    GEO_SUBHORIZON  "1" => causal guard: drag only on sub-horizon modes
                    (k > aH = conformal Hubble). v2 addition, 2026-07-06:
                    super-horizon drag exceeded the physical premise
                    (localization is sub-horizon physics) and may have
                    inflated the low-l TT numbers in the v1 runs.

Physics (matches script5_kdep_growth.py exactly):
    trigger (proper-r):  3 * rc^3 * (Om_m + 4*Om_L*a^3) > eps,
                         rc = (2*pi/k) * H0     [k and H0 both in 1/Mpc]
    drag:                theta_cdm' += -Gamma0 * (a'/a) * theta_cdm
                         (i.e. Hubble drag multiplied by (1+Gamma0) while on)

SCOPED DEVIATION (stated): CDM-only in this pass. The baryon Euler is
entangled with photon-coupling terms; coupling-case enumeration is a
follow-up run.

Idempotent: refuses to double-patch (marker check). Anchors asserted unique.
Usage:  python3 class_tier2_patch.py <path-to-class_public>
"""
import sys
from pathlib import Path

MARKER = "GEO_FRICTION PATCH"

HELPER = r'''
/* === GEO_FRICTION PATCH (Ring A, Tier-2, 2026-07-06) ==================
   Scale-dependent late-time drag on CDM, env-var controlled:
     GEO_GAMMA0 (drag strength), GEO_EPS (trigger threshold).
   Trigger (proper-r convention, cf. script5_kdep_growth.py):
     3*rc^3*(Om_m + 4*Om_L*a^3) > eps,  rc = (2*pi/k)*H0  [1/Mpc units]
   Drag: Hubble drag on theta_cdm multiplied by (1+Gamma0) while triggered.
   Init via gcc constructor => single-threaded, no OpenMP race.
   ==================================================================== */
#include <stdlib.h>
static double geo_gamma0 = 0.0;
static double geo_eps = 0.0;
static int geo_subhorizon = 0;
__attribute__((constructor)) static void geo_init(void) {
  const char *g = getenv("GEO_GAMMA0");
  const char *e = getenv("GEO_EPS");
  const char *s = getenv("GEO_SUBHORIZON");
  geo_gamma0 = (g == NULL) ? 0.0 : atof(g);
  if (geo_gamma0 < 0.0) geo_gamma0 = 0.0;
  geo_eps = (e == NULL) ? 0.0 : atof(e);
  geo_subhorizon = (s != NULL && s[0] == '1') ? 1 : 0;
  if (geo_gamma0 > 0.0)
    printf("# GEO_FRICTION active: Gamma0=%g eps=%g subhorizon_guard=%d (CDM-only)\n",
           geo_gamma0, geo_eps, geo_subhorizon);
}
static double geo_drag_factor(double a, double k, double aH,
                              double H0, double Om_m, double Om_L) {
  double rc, trig;
  if (geo_gamma0 <= 0.0) return 0.0;
  /* v2 causal guard: localization is sub-horizon physics; no drag on
     super-horizon modes. aH here is the conformal Hubble a'/a [1/Mpc],
     same units as k. */
  if (geo_subhorizon == 1 && k <= aH) return 0.0;
  rc = 2.0 * _PI_ / k * H0;
  trig = 3.0 * rc * rc * rc * (Om_m + 4.0 * Om_L * a * a * a);
  return (trig > geo_eps) ? geo_gamma0 : 0.0;
}
/* === end GEO_FRICTION PATCH =========================================== */
'''

ANCHOR_INCLUDE = '#include "perturbations.h"'

ANCHOR_CDM = ("        dy[pv->index_pt_theta_cdm] = - a_prime_over_a"
              "*y[pv->index_pt_theta_cdm] + metric_euler; /* cdm velocity */")

PATCHED_CDM = ("        dy[pv->index_pt_theta_cdm] = - (1.0+geo_drag_factor("
               "a,k,a_prime_over_a,pba->H0,pba->Omega0_b+pba->Omega0_cdm,"
               "pba->Omega0_lambda))"
               "*a_prime_over_a*y[pv->index_pt_theta_cdm] + metric_euler; "
               "/* cdm velocity -- GEO_FRICTION patched v2 */")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: class_tier2_patch.py <class_public dir>")
        return 1
    src = Path(sys.argv[1]) / "source" / "perturbations.c"
    text = src.read_text(encoding="utf-8")

    if MARKER in text:
        print("already patched — nothing to do (idempotent).")
        return 0

    n_inc = text.count(ANCHOR_INCLUDE)
    n_cdm = text.count(ANCHOR_CDM)
    if n_inc < 1 or n_cdm != 1:
        print(f"ANCHOR FAILURE: include x{n_inc}, cdm-euler x{n_cdm} "
              "(need >=1 and ==1). CLASS source drifted — re-derive anchors.")
        return 2

    # insert helper after the FIRST include line
    idx = text.index(ANCHOR_INCLUDE) + len(ANCHOR_INCLUDE)
    text = text[:idx] + "\n" + HELPER + text[idx:]
    # patch the CDM Euler
    text = text.replace(ANCHOR_CDM, PATCHED_CDM)

    src.write_text(text, encoding="utf-8")
    print("patched OK: helper block + CDM Euler drag installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
