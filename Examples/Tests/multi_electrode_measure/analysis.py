#!/usr/bin/env python3
"""Analysis for the multi-electrode voltage-measurement test.

Checks that the flux + capacitance measurement recovers the known per-electrode
target voltages V1, V2 from the induced charges (V_eff = C^{-1}(Q - Q_g)), even
with the plasma screening charge present. The decisive number is the *first*
(t=0, screened) measurement; later steps show the drift as the field evolves.
"""

import csv
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else "multi_electrode_Veff.csv"
TOL = 0.05  # 5% -- generous, staircase/particle-noise limited

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append({k: (float(v) if v not in ("", None) else None) for k, v in r.items()})

if not rows:
    print("FAIL: no rows in", CSV)
    sys.exit(1)

# Electrode names from the header (Veff_<name> / Vtarget_<name>).
names = [k[len("Veff_"):] for k in rows[0] if k.startswith("Veff_")]

first = rows[0]  # t=0 screened static measurement -- the decisive check
print(f"=== multi-electrode measurement check ({CSV}) ===")
print(f"{'electrode':10s} {'V_target':>12s} {'V_eff(t=0)':>12s} {'rel.err':>9s}")
ok = True
for nm in names:
    vt = first[f"Vtarget_{nm}"]
    ve = first[f"Veff_{nm}"]
    qg = first.get(f"Qg_{nm}")
    rel = abs(ve - vt) / max(abs(vt), 1.0)
    flag = "OK" if rel <= TOL else "FAIL"
    ok = ok and rel <= TOL
    print(f"{nm:10s} {vt:12.2f} {ve:12.2f} {rel:8.1%}  {flag}   (Q_g={qg:.3e})")

# Report the drift over the logged steps (context, not a pass/fail).
if len(rows) > 1:
    print("\ndrift over logged steps:")
    print("  " + "  ".join(["step"] + [f"Veff_{n}" for n in names]))
    for row in rows:
        print("  " + "  ".join(
            [f"{int(row['step']):5d}"] + [f"{row[f'Veff_{n}']:11.2f}" for n in names]
        ))

print()
if ok:
    print(f"PASS: all electrodes recovered within {TOL:.0%} at t=0")
    sys.exit(0)
print(f"FAIL: some electrode V_eff off by more than {TOL:.0%}")
sys.exit(1)
