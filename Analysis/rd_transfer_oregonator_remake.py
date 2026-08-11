"""
Oregonator-only remakes of the TEST 1 (channel) and TEST 4a (aniso)
transfer figures, from the EXISTING final2 JSON results -- no reruns.

The thesis goes Oregonator-only (Barkley dropped), so these figures
replot only the 'oregonator' entry of
  figures/rd_transfer_channel_final2.json  ->  rd_transfer_channel_oregonator.png
  figures/rd_transfer_aniso_final2.json    ->  rd_transfer_aniso_oregonator.png
in the same visual style as rd_transfer_channel_final2.py /
rd_transfer_aniso_final2.py.  The final2 files are NOT touched.

NOTE (aniso): the final2 JSON stores only the fitted axis-speed ratios
('cases'), not the front-extent time tracks (those lived only in
memory in rd_transfer_aniso_final2.py), so the Oregonator-only aniso
remake contains the sqrt(r) check panel only.

Usage:
    ../.venv/bin/python rd_transfer_oregonator_remake.py   (from Analysis/)
"""

import json

import numpy as np
import matplotlib.pyplot as plt

U_THRESH = 0.5
XTICKS = [1, 2, 4, 8, 16, 32, 60]


def remake_channel():
    with open('figures/rd_transfer_channel_final2.json') as fh:
        res = json.load(fh)['oregonator']
    sweep, free, traces = res['sweep'], res['free_medium'], res['traces']
    thr = res['block_threshold']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for W, d in traces.items():
        ax.plot(d['t'], d['far'], label=f'W={W}')
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
    ax.set_xlabel('t (t.u.)')
    ax.set_ylabel('mean u at far probe')
    ax.set_title('oregonator: far-probe traces')
    ax.legend(fontsize=8)

    ax = axes[1]
    sp = [(r['W'], r['speed_cells_per_tu']) for r in sweep
          if r['speed_cells_per_tu'] is not None]
    ax.plot([s[0] for s in sp], [s[1] for s in sp], 'o-', ms=4,
            label='channel (3-probe fit)')
    ax.axhline(free['speed_cells_per_tu'], color='r', ls='--',
               label=f"free medium ({free['speed_cells_per_tu']:.3f})")
    ax.set_xscale('log')
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XTICKS)
    ax.set_xlabel('channel width W (cells)')
    ax.set_ylabel('speed (cells / t.u.)')
    ax.set_title('oregonator: speed vs width')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot([r['W'] for r in sweep], [r['peak_far'] for r in sweep],
            's-', ms=4, color='darkgreen')
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8, label='threshold')
    ax.set_xscale('log')
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XTICKS)
    ax.set_xlabel('channel width W (cells)')
    ax.set_ylabel('peak u at far probe')
    ax.set_title('oregonator: transmission amplitude '
                 f"(transmits W>={thr['min_width_transmitted']})")
    ax.legend(fontsize=8)

    fig.suptitle('TEST 1 FINAL2 (clamp-rest fix), Oregonator A only: '
                 'pulse transfer through a walled channel', fontsize=13)
    fig.tight_layout()
    figpath = 'figures/rd_transfer_channel_oregonator.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')


def remake_aniso():
    with open('figures/rd_transfer_aniso_final2.json') as fh:
        res = json.load(fh)['oregonator']

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    rs = sorted(float(k) for k in res['cases'])
    ratios = [res['cases'][str(int(r))]['ratio'] for r in rs]
    ax.plot(rs, ratios, 'o', color='tab:red', ms=8,
            label='measured c_par/c_perp')
    rf = np.linspace(1, 4, 100)
    ax.plot(rf, np.sqrt(rf), 'k--', label='sqrt(r)')
    for r, ratio in zip(rs, ratios):
        c = res['cases'][str(int(r))]
        ax.annotate(f"{c['deviation_pct']:.1f}%", (r, ratio),
                    textcoords='offset points', xytext=(8, -12),
                    fontsize=9)
    ax.set_xlabel('anisotropy ratio r = D_par/D_perp')
    ax.set_ylabel('c_par / c_perp')
    ax.set_title('oregonator: sqrt(r) check')
    ax.legend()
    fig.suptitle('TEST 4a FINAL2, Oregonator A only:\n'
                 'anisotropic target wave', fontsize=12)
    fig.tight_layout()
    figpath = 'figures/rd_transfer_aniso_oregonator.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')


if __name__ == '__main__':
    remake_channel()
    remake_aniso()
