"""
DE-HACK VERIFICATION -- re-run key transfer-test cases with the de-hacked
rd_core (no np.clip; adaptive reaction subcycling + blow-up guard) and
compare against the pre-de-hack JSONs.

Re-runs:
  - Test 1 (rd_transfer_channel.py, Barkley): free medium + W=16 and W=8
    channels.
  - Test 4a (rd_transfer_aniso.py): r = 1 and r = 2 axis speeds.

Outputs (suffix _dehacked preserves the old files):
  figures/rd_transfer_channel_barkley_dehacked.json / .png
  figures/rd_transfer_aniso_dehacked.json / .png

HEADLINE RESULT (2026-07-20)
----------------------------
The de-hacked core does NOT reproduce the old absolute pulse speeds --
and it should not: the old runs integrated the stiff Barkley reaction
(dt/eps = 2.5) with a single explicit Euler step whose overshoots were
silently reset by np.clip.  That combination underestimates the Barkley
free-medium speed by ~17%:
    old clipped core:   3.446 cells/t.u.  (rd_transfer_channel_barkley.json)
    de-hacked core:     4.151 cells/t.u.  (dt-converged: 4.14-4.16 at
                                         dt = 0.05 / 0.02 / 0.01)
The de-hacked value is corroborated by plain unclipped Euler at fine
dt = 0.01 (no clips needed, dt/eps = 0.5): 4.11 cells/t.u., and the
point kinetics match scipy LSODA to ~1e-6 on the slow phases.  The
PHYSICS of the old tests survives: the sqrt(r) anisotropy ratio holds
(r=2: 1.479 vs 1.414, 4.6% off) and channels transmit cleanly.
"""

import json
import math
import numpy as np
import matplotlib.pyplot as plt

from rd_core import RDSubstrate
from rd_transfer_channel import (run_case, analyse, KINETICS, X_PORT,
                                 disk_mask, X_MID, X_FAR, U_THRESH)
from rd_transfer_aniso import run_4a

OUT_CH = 'figures/rd_transfer_channel_barkley_dehacked.json'
OUT_CH_PNG = 'figures/rd_transfer_channel_barkley_dehacked.png'
OUT_AN = 'figures/rd_transfer_aniso_dehacked.json'
OUT_AN_PNG = 'figures/rd_transfer_aniso_dehacked.png'
OLD_CH = 'figures/rd_transfer_channel_barkley.json'
OLD_AN = 'figures/rd_transfer_aniso.json'


def test1():
    print('=== Test 1 re-run (Barkley, de-hacked core) ===')
    out = {}
    for width, label in [(None, 'free'), (16, 'W=16'), (8, 'W=8')]:
        data = run_case('barkley', width=width)
        r = analyse(data, label)
        key = 'free_medium' if width is None else label
        out[key] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in r.items()}
        print(f"  {label}: speed={r['speed']:.5f} cells/step "
              f"({r['speed']/0.05:.4f} cells/t.u.), peak_far={r['peak_far']:.4f}, "
              f"transmitted={r['transmitted']}")
    with open(OLD_CH) as fh:
        old = json.load(fh)
    out['comparison_with_pre_dehack'] = {
        'old_free_medium_speed_cells_per_step': old['free_medium']['speed'],
        'new_free_medium_speed_cells_per_step': out['free_medium']['speed'],
        'abs_speed_change_pct': 100 * (out['free_medium']['speed']
                                       / old['free_medium']['speed'] - 1),
        'note': ('Old number was produced by unstable coarse explicit '
                 'Euler (dt/eps=2.5) patched by np.clip; de-hacked value '
                 'is dt-converged (4.14-4.16 cells/t.u. at dt=0.05/0.02/'
                 '0.01) and matches unclipped fine-dt Euler (4.11).'),
    }
    with open(OUT_CH, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'Saved {OUT_CH}')
    return out


def test4a():
    print('=== Test 4a re-run (Barkley, de-hacked core) ===')
    out = {}
    for r, dt, nsteps in [(1, 0.05, 900), (2, 0.05, 700)]:
        c_par, c_perp, snaps, track = run_4a(r, dt, nsteps)
        ratio = c_par / c_perp
        pred = math.sqrt(r)
        out[str(r)] = {'c_par': c_par, 'c_perp': c_perp, 'ratio': ratio,
                       'sqrt_r': pred,
                       'deviation_pct': abs(ratio - pred) / pred * 100}
        print(f'  r={r}: c_par={c_par:.4f}, c_perp={c_perp:.4f}, '
              f'ratio={ratio:.4f} vs sqrt(r)={pred:.4f} '
              f'({out[str(r)]["deviation_pct"]:.2f}% off)')
    with open(OLD_AN) as fh:
        old = json.load(fh)['4a']
    out['comparison_with_pre_dehack'] = {
        'old_r2': old['2'], 'new_r2': out['2'],
        'abs_speed_change_pct_r2_perp':
            100 * (out['2']['c_perp'] / old['2']['c_perp'] - 1),
        'ratio_law': ('sqrt(r) ratio survives de-hacking: r=2 ratio '
                      f"{out['2']['ratio']:.3f} (old {old['2']['ratio']:.3f}, "
                      'theory 1.414)'),
    }
    with open(OUT_AN, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'Saved {OUT_AN}')
    return out, old


def plot(ch, an, old_an):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    labels = ['free', 'W=16', 'W=8']
    speeds = [ch['free_medium']['speed'], ch['W=16']['speed'],
              ch['W=8']['speed']]
    old_free = ch['comparison_with_pre_dehack'][
        'old_free_medium_speed_cells_per_step']
    ax.bar(labels, [s / 0.05 for s in speeds], color='steelblue',
           label='de-hacked')
    ax.axhline(old_free / 0.05, color='r', ls='--',
               label=f'old clipped free medium ({old_free/0.05:.3f})')
    ax.set_ylabel('speed (cells / t.u.)')
    ax.set_title('Test 1 (Barkley): pulse speed, de-hacked core')
    ax.legend()

    ax = axes[1]
    rs = ['1', '2']
    ax.plot(rs, [an[r]['c_par'] for r in rs], 'o-', label='c_par (de-hacked)')
    ax.plot(rs, [an[r]['c_perp'] for r in rs], 's-',
            label='c_perp (de-hacked)')
    ax.plot(rs, [old_an[r]['c_par'] for r in rs], 'o--', color='gray',
            label='c_par (old clipped)')
    ax.plot(rs, [old_an[r]['c_perp'] for r in rs], 's--', color='lightgray',
            label='c_perp (old clipped)')
    ax.set_xlabel('anisotropy ratio r')
    ax.set_ylabel('speed (cells / t.u.)')
    ax.set_title('Test 4a: axis speeds')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_CH_PNG.replace('channel_barkley', 'aniso'), dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_CH_PNG.replace('channel_barkley', 'aniso')}")

    # channel figure: probe traces not stored; plot speed bars alone
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [s / 0.05 for s in speeds], color='steelblue')
    ax.axhline(old_free / 0.05, color='r', ls='--',
               label=f'old clipped free medium ({old_free/0.05:.3f})')
    ax.set_ylabel('speed (cells / t.u.)')
    ax.set_title('Test 1 (Barkley) de-hacked: free + channel speeds')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_CH_PNG, dpi=150)
    plt.close(fig)
    print(f'Saved {OUT_CH_PNG}')


if __name__ == '__main__':
    ch = test1()
    an, old_an = test4a()
    plot(ch, an, old_an)
