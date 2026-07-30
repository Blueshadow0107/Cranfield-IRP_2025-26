"""Validate rd_train_xor_phi_fast.py (cropped domain) against the frozen
full-grid references BEFORE any training campaign.  Mandatory gate.

Checks (all must pass):
  1. Cropped-domain protocol (4 uniform-phi0 reference sims, recomputed
     here) vs the frozen figures/rd_train_xor_protocol.json:
       |tA_fast - 31.95| < 0.3 t.u.,  |tB_fast - 26.95| < 0.3 t.u.
       A0 within 2% of the frozen A0 (~0.707)
       baseline windowed truth-table peaks within 0.02 absolute
  2. One random candidate (seed 0) through BOTH the debug script's and
     the fast script's objective(): losses must agree within 0.01.

Run from Analysis/RD:  ../../.venv/bin/python validate_fast.py
Exit code 0 = all checks passed, 1 = at least one failed.

The debug script, rd_core.py, config.json and the frozen protocol JSON are
treated read-only.  The only files written are
figures/rd_train_fast_protocol.json (by the fast module's
compute_protocol) and figures/rd_train_fast_validate_eval_log.jsonl.
"""

import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # the debug module reads config.json from the cwd
sys.path.insert(0, SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import numpy as np

import rd_train_xor_phi_debug as dbg
import rd_train_xor_phi_fast as fast

FROZEN_JSON = os.path.join(fast.FIG, 'rd_train_xor_protocol.json')
VALIDATE_LOG = os.path.join(fast.FIG, 'rd_train_fast_validate_eval_log.jsonl')

TOL_T = 0.3        # t.u. tolerance on arrival times
TOL_A0 = 0.02      # relative tolerance on A0
TOL_PEAK = 0.02    # absolute tolerance on baseline windowed peaks
TOL_LOSS = 0.01    # debug-vs-fast objective agreement


def check(name, value, reference, passed, detail=''):
    status = 'PASS' if passed else 'FAIL'
    print(f'  [{status}] {name:34s} fast={value!r:>24} frozen={reference!r:>24} {detail}')
    return bool(passed)


def main():
    t_start = time.time()
    with open(FROZEN_JSON) as fh:
        frozen = json.load(fh)

    # Both modules' objectives append eval rows; send them to a dedicated
    # validation log so campaign logs stay clean.
    dbg.EVAL_LOG = VALIDATE_LOG
    fast.EVAL_LOG = VALIDATE_LOG

    print('=' * 78)
    print('CHECK 1: cropped-domain protocol vs frozen full-grid protocol')
    print('=' * 78)
    t0 = time.time()
    proto_fast = fast.compute_protocol()   # 4 sims x 1000 steps, cropped
    print(f'  (protocol sims took {time.time() - t0:.0f}s on the cropped domain)')

    ok = True
    ok &= check('tA_arrival', round(proto_fast['tA_arrival'], 4),
                round(frozen['tA_arrival'], 4),
                abs(proto_fast['tA_arrival'] - frozen['tA_arrival']) < TOL_T,
                f'|d|={abs(proto_fast["tA_arrival"] - frozen["tA_arrival"]):.4f} < {TOL_T}')
    ok &= check('tB_arrival', round(proto_fast['tB_arrival'], 4),
                round(frozen['tB_arrival'], 4),
                abs(proto_fast['tB_arrival'] - frozen['tB_arrival']) < TOL_T,
                f'|d|={abs(proto_fast["tB_arrival"] - frozen["tB_arrival"]):.4f} < {TOL_T}')
    a0_rel = abs(proto_fast['A0_lone_peak'] - frozen['A0_lone_peak']) / frozen['A0_lone_peak']
    ok &= check('A0_lone_peak', round(proto_fast['A0_lone_peak'], 6),
                round(frozen['A0_lone_peak'], 6),
                a0_rel < TOL_A0, f'rel diff={a0_rel:.2e} < {TOL_A0}')
    for tag in ('00', '10', '01', '11'):
        pf = proto_fast['baseline_truth_windowed'][tag]
        pz = frozen['baseline_truth_windowed'][tag]
        ok &= check(f'baseline peak {tag}', round(pf, 6), round(pz, 6),
                    abs(pf - pz) < TOL_PEAK,
                    f'|d|={abs(pf - pz):.2e} < {TOL_PEAK}')
    print(f"  window(fast)={proto_fast['window_tu']}  window(frozen)={frozen['window_tu']}")
    print(f"  nsteps_train: fast={proto_fast['nsteps_train']} frozen={frozen['nsteps_train']}")

    print()
    print('=' * 78)
    print('CHECK 2: random candidate (seed 0) through debug vs fast objective')
    print('=' * 78)
    rng = np.random.default_rng(0)
    x = rng.uniform(dbg.PHI0, dbg.PHI_MAX, dbg.NB * dbg.NB)

    # Same frozen protocol for both objectives: isolates the geometry as
    # the only difference (window, A0, nsteps_train identical).
    t0 = time.time()
    loss_dbg = dbg.objective(x.copy(), frozen)
    t_dbg = time.time() - t0
    print(f'  debug objective: loss={loss_dbg:.6f}  ({t_dbg:.0f}s, full grid)')
    t0 = time.time()
    loss_fast = fast.objective(x.copy(), frozen)
    t_fast = time.time() - t0
    print(f'  fast  objective: loss={loss_fast:.6f}  ({t_fast:.0f}s, cropped)')
    ok &= check('objective loss', round(loss_fast, 6), round(loss_dbg, 6),
                abs(loss_fast - loss_dbg) < TOL_LOSS,
                f'|d|={abs(loss_fast - loss_dbg):.2e} < {TOL_LOSS}')

    print()
    print('=' * 78)
    print(f'VALIDATION {"PASSED" if ok else "FAILED"} '
          f'(total wall time {(time.time() - t_start) / 60:.1f} min)')
    print('=' * 78)
    if not ok:
        print('DO NOT LAUNCH THE CAMPAIGN -- debug the failing checks first.')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
