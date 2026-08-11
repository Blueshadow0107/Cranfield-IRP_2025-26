"""
rd_pce_sensitivity.py

Lightweight Sobol sensitivity analysis for the Oregonator BZ channel-pulse
QoIs.  Perturbs four scalar parameters around the working point and reports
first-order and total-effect Sobol indices for pulse speed and peak amplitude.

Parameters (isotropic diffusion):
    eps  -- activator/recovery timescale separation
    f    -- stoichiometric factor
    phi  -- uniform background light-suppression field
    Du   -- activator diffusivity (scalar, isotropic)

QoIs:
    c    -- pulse speed between two small probes (cells / t.u.)
    peak -- maximum activator value at the downstream probe

Outputs:
    Analysis/figures/rd_pce_sensitivity/results.json
    Analysis/figures/rd_pce_sensitivity/{sobol_speed,sobol_peak,hist_speed,hist_peak,pairwise}.png
    Analysis/figures/rd_pce_sensitivity/{sobol_speed,sobol_peak}.pdf

Usage:
    cd Analysis
    ../.venv/bin/python rd_pce_sensitivity.py
"""

import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from SALib.sample import sobol as saltelli
from SALib.analyze import sobol as sobol

sys.path.insert(0, os.path.dirname(__file__))
from rd_core import RDSubstrate

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Working point and parameter ranges (uniform +/- 10 %)
# ---------------------------------------------------------------------------
NOM = {
    'eps': 0.05014844822490394,
    'f': 1.4375,
    'phi': 0.010,
    'Du': 1.0,
}

FRACTION = 0.10

PROBLEM = {
    'num_vars': 4,
    'names': ['eps', 'f', 'phi', 'Du'],
    'bounds': [
        [NOM['eps'] * (1 - FRACTION), NOM['eps'] * (1 + FRACTION)],
        [NOM['f']   * (1 - FRACTION), NOM['f']   * (1 + FRACTION)],
        [NOM['phi'] * (1 - FRACTION), NOM['phi'] * (1 + FRACTION)],
        [NOM['Du']  * (1 - FRACTION), NOM['Du']  * (1 + FRACTION)],
    ],
}

U_STAR = 0.0030821
THRESH = 0.30
NX, NY = 120, 40
DX = 1.0
DT = 0.05
DURATION_STEPS = 30
TOTAL_STEPS = 500
N_SOBOL = 64           # saltelli sample size (N*(D+2) evaluations with calc_second_order=False)
N_WORKERS = 8

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'rd_pce_sensitivity')
os.makedirs(FIG_DIR, exist_ok=True)


def _build_rd(eps, f, phi0, Du):
    """Return a configured RDSubstrate for the channel-pulse experiment."""
    rd = RDSubstrate(
        nx=NX, ny=NY, dx=DX, dt=DT,
        kinetics='oregonator',
        eps=eps, q=0.002, f=f,
        Du=Du, Dv=0.0,
        clamp_rest=(U_STAR, U_STAR),
    )
    # straight channel, width 20 cells, centred in y
    wall = np.zeros((NX, NY), bool)
    wall[:, :10] = True
    wall[:, 30:] = True
    rd.set_walls(wall)
    rd.set_phi(np.full((NX, NY), phi0))
    rd.u[:, :] = U_STAR
    rd.v[:, :] = U_STAR

    port = np.zeros((NX, NY), bool)
    port[5:15, 10:30] = True
    rd.add_port('in', port)

    # three small centre-line probes for speed measurement
    for name, x in [('p40', 40), ('p80', 80), ('p100', 100)]:
        mask = np.zeros((NX, NY), bool)
        mask[x:x + 2, 19:21] = True
        rd.add_probe(name, mask)

    rd.fire('in', value=0.8, v_value=0.2, duration=DURATION_STEPS)
    return rd


def evaluate(params):
    """Evaluate one parameter vector. Returns (speed, peak, arrival_time)."""
    eps, f, phi, Du = params
    try:
        rd = _build_rd(eps, f, phi, Du)
        data = rd.run(TOTAL_STEPS)
        t = data['t']
        s40 = data['p40']
        s100 = data['p100']

        idx40 = np.where(s40 > THRESH)[0]
        idx100 = np.where(s100 > THRESH)[0]
        if len(idx40) == 0 or len(idx100) == 0:
            return 0.0, 0.0, np.nan

        t40 = t[idx40[0]]
        t100 = t[idx100[0]]
        dt = t100 - t40
        if dt <= 0:
            return 0.0, 0.0, np.nan

        speed = 60.0 / dt          # p40 -> p100 is 60 cells
        peak = float(s100.max())
        arrival = float(t100)
        return speed, peak, arrival
    except Exception as exc:
        # Return sentinel values for failed runs so SALib still has finite output.
        return 0.0, 0.0, np.nan


def main():
    print(f'[PCE] Generating Saltelli sample: N={N_SOBOL}, D={PROBLEM["num_vars"]}')
    param_values = saltelli.sample(PROBLEM, N_SOBOL, calc_second_order=False)
    n_eval = param_values.shape[0]
    print(f'[PCE] Total evaluations: {n_eval}')

    print(f'[PCE] Running forward solves on {N_WORKERS} workers...')
    with Pool(N_WORKERS) as pool:
        results = pool.map(evaluate, param_values, chunksize=1)

    results = np.asarray(results, dtype=float)
    speed = results[:, 0]
    peak = results[:, 1]
    arrival = results[:, 2]

    n_fail = np.sum(speed == 0.0)
    print(f'[PCE] Completed: {n_eval - n_fail}/{n_eval} successful propagations')

    # Basic statistics
    def stats(y):
        mask = np.isfinite(y) & (y > 0)
        yf = y[mask]
        return {
            'mean': float(yf.mean()) if len(yf) else None,
            'std': float(yf.std()) if len(yf) else None,
            'p5': float(np.percentile(yf, 5)) if len(yf) else None,
            'p95': float(np.percentile(yf, 95)) if len(yf) else None,
        }

    stats_speed = stats(speed)
    stats_peak = stats(peak)
    stats_arrival = stats(arrival)

    print(f'[PCE] Speed stats: {stats_speed}')
    print(f'[PCE] Peak stats:  {stats_peak}')

    # Sobol analysis
    print('[PCE] Sobol analysis for speed...')
    Si_speed = sobol.analyze(PROBLEM, speed, calc_second_order=False)
    print('[PCE] Sobol analysis for peak...')
    Si_peak = sobol.analyze(PROBLEM, peak, calc_second_order=False)

    def si_dict(si):
        return {
            'S1': {n: float(v) for n, v in zip(PROBLEM['names'], si['S1'])},
            'ST': {n: float(v) for n, v in zip(PROBLEM['names'], si['ST'])},
            'S1_conf': {n: float(v) for n, v in zip(PROBLEM['names'], si['S1_conf'])},
            'ST_conf': {n: float(v) for n, v in zip(PROBLEM['names'], si['ST_conf'])},
        }

    out = {
        'problem': PROBLEM,
        'nominal': NOM,
        'N_sobol': N_SOBOL,
        'n_eval': n_eval,
        'n_fail': int(n_fail),
        'speed': stats_speed,
        'peak': stats_peak,
        'arrival': stats_arrival,
        'sobol_speed': si_dict(Si_speed),
        'sobol_peak': si_dict(Si_peak),
    }

    json_path = os.path.join(FIG_DIR, 'results.json')
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'[PCE] Saved {json_path}')

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    # Histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.hist(speed[speed > 0], bins=40, color='steelblue', edgecolor='white')
    ax.axvline(stats_speed['mean'], color='crimson', linestyle='--', label=f"mean={stats_speed['mean']:.2f}")
    ax.set_xlabel('Pulse speed (cells / t.u.)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of pulse speed')
    ax.legend()

    ax = axes[1]
    ax.hist(peak[peak > 0], bins=40, color='seagreen', edgecolor='white')
    ax.axvline(stats_peak['mean'], color='crimson', linestyle='--', label=f"mean={stats_peak['mean']:.3f}")
    ax.set_xlabel('Peak probe activator')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of probe peak')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'hist_speed_peak.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'hist_speed_peak.pdf'))
    plt.close(fig)

    # Sobol index bar charts
    def plot_sobol(si, title, fname):
        names = PROBLEM['names']
        s1 = [si['S1'][n] for n in names]
        st = [si['ST'][n] for n in names]
        s1c = [si['S1_conf'][n] for n in names]
        stc = [si['ST_conf'][n] for n in names]

        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, s1, width, yerr=s1c, label='S1 (first-order)', color='steelblue', capsize=3)
        ax.bar(x + width / 2, st, width, yerr=stc, label='ST (total-effect)', color='coral', capsize=3)
        ax.set_ylabel('Sobol index')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.axhline(0, color='k', linewidth=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f'{fname}.png'), dpi=200)
        fig.savefig(os.path.join(FIG_DIR, f'{fname}.pdf'))
        plt.close(fig)

    plot_sobol(out['sobol_speed'], 'Sobol indices: pulse speed', 'sobol_speed')
    plot_sobol(out['sobol_peak'], 'Sobol indices: probe peak', 'sobol_peak')

    # Pairwise scatter matrix
    valid = speed > 0
    labels = PROBLEM['names']
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    for i, name in enumerate(labels):
        ax = axes[i]
        ax.scatter(param_values[valid, i], speed[valid], alpha=0.3, s=8, color='steelblue')
        ax.set_xlabel(name)
        ax.set_ylabel('Pulse speed')
        ax.set_title(f'{name} vs speed')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'pairwise_speed.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'pairwise_speed.pdf'))
    plt.close(fig)

    print('[PCE] Figures saved to', FIG_DIR)


if __name__ == '__main__':
    main()
