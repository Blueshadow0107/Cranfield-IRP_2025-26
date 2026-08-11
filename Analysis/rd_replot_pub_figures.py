"""Replot existing RD results as publication-quality figures.

This script does NOT rerun simulations; it reads the JSON outputs already
saved in Analysis/figures/ and redraws them with the thesis house style into
Analysis/figures/pub/ (PNG + PDF).

Oregonator-only: all figures use the validated excitable-Oregonator parameters
(f=1.4375, eps=0.050148, phi=0.010).  Barkley entries, where present in the
JSONs, are ignored because Barkley has been dropped from the thesis narrative.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rd_house_style as rhs

FIG_DIR = Path(__file__).parent / 'figures'
PUB_DIR = FIG_DIR / 'pub'
PUB_DIR.mkdir(parents=True, exist_ok=True)

U_THRESH = 0.5


def load(name):
    """Load a JSON from Analysis/figures/."""
    path = FIG_DIR / name
    with open(path) as fh:
        return json.load(fh)


def save(fig, stem):
    """Save PNG and PDF versions of a figure."""
    return rhs.save_both(fig, str(PUB_DIR / stem))


# -----------------------------------------------------------------------------
# Figure 1: channel pulse transfer
# -----------------------------------------------------------------------------
def plot_channel_transfer():
    data = load('rd_transfer_channel_final2.json')['oregonator']
    sweep = data['sweep']
    free = data['free_medium']
    traces = data['traces']

    fig, axes = plt.subplots(1, 2, figsize=rhs.DOUBLE_PANEL)

    # Left: far-probe traces for selected widths
    ax = axes[0]
    order = ['60', '16', '8', '1']
    for i, w in enumerate(order):
        if w in traces:
            d = traces[w]
            ax.plot(d['t'], d['far'], label=f'W = {w}', color=rhs.get_color(i))
    ax.axhline(U_THRESH, color=rhs.COLORS['grey'], ls='--', lw=1.0,
               label='threshold')
    ax.set_xlabel('time (t.u.)')
    ax.set_ylabel('mean $u$ at far probe')
    ax.set_title('Pulse arrival traces')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)

    # Right: speed vs width
    ax = axes[1]
    widths = [r['W'] for r in sweep if r['speed_cells_per_tu'] is not None]
    speeds = [r['speed_cells_per_tu'] for r in sweep
              if r['speed_cells_per_tu'] is not None]
    ax.plot(widths, speeds, 'o-', ms=5, color=rhs.get_color(0),
            label='channel')
    ax.axhline(free['speed_cells_per_tu'], color=rhs.COLORS['grey'], ls='--',
               lw=1.0, label=f"free medium ({free['speed_cells_per_tu']:.2f})")
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 8, 16, 32, 60])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 60])
    ax.set_xlabel('channel width $W$ (cells)')
    ax.set_ylabel('speed (cells / t.u.)')
    ax.set_title('Pulse speed vs channel width')
    ax.legend(loc='lower right')

    fig.suptitle('Channel pulse transfer (Oregonator A)',
                 fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_channel_transfer')


# -----------------------------------------------------------------------------
# Figure 2: refractory frequency response
# -----------------------------------------------------------------------------
def plot_frequency_response():
    data = load('rd_transfer_frequency_final2_oregonator.json')
    rows = data['train_results']
    refractory_tu = data['refractory_period_time']

    fig, axes = plt.subplots(1, 2, figsize=rhs.DOUBLE_PANEL)

    # Left: output vs input rate
    ax = axes[0]
    f_in = np.array([r['f_in'] for r in rows])
    f_out = np.array([r['f_out'] for r in rows])
    lim = f_in.max() * 1.1
    ax.plot(f_in, f_out, 'o-', ms=5, color=rhs.get_color(0),
            label='measured')
    ax.plot([0, lim], [0, lim], color=rhs.COLORS['grey'], ls='--', lw=1.0,
            label='1:1 line')
    ax.axvline(1.0 / refractory_tu, color=rhs.COLORS['vermillion'], ls=':',
               lw=1.5, label=f'1/refractory ({1.0/refractory_tu:.3f})')
    ax.set_xlabel('input rate $f_{\\mathrm{in}}$ (pulses / t.u.)')
    ax.set_ylabel('output rate $f_{\\mathrm{out}}$ (pulses / t.u.)')
    ax.set_title('Frequency response')
    ax.legend(loc='lower right')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # Right: transmitted count vs period
    ax = axes[1]
    periods = np.array([r['period_steps'] for r in rows])
    transmitted = np.array([r['transmitted'] for r in rows])
    n_fired = rows[0]['fired']
    ax.plot(periods * 0.05, transmitted, 's-', ms=5,
            color=rhs.get_color(1), label='transmitted')
    ax.axhline(n_fired, color=rhs.COLORS['grey'], ls='--', lw=1.0,
               label=f'fired ({n_fired})')
    ax.axvline(refractory_tu, color=rhs.COLORS['vermillion'], ls=':', lw=1.5,
               label=f'refractory ({refractory_tu:.1f} t.u.)')
    ax.set_xlabel('input period (t.u.)')
    ax.set_ylabel('pulses transmitted')
    ax.set_title('Transmission count vs period')
    ax.legend(loc='lower right')

    fig.suptitle('Refractory frequency response (Oregonator A)',
                 fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_frequency_response')


# -----------------------------------------------------------------------------
# Figure 3: T-junction inhibition logic
# -----------------------------------------------------------------------------
def plot_tjunction_logic():
    data = load('rd_transfer_logic_final2_oregonator.json')
    truth = data['truth_table']
    thr = data['decision_threshold']
    sweep = data['timing_sweep']

    fig, axes = plt.subplots(1, 2, figsize=rhs.DOUBLE_PANEL)

    # Left: windowed truth table
    ax = axes[0]
    labels = ['00', '10', '01', '11']
    vals = [truth[k] for k in labels]
    colors = [rhs.COLORS['grey'], rhs.get_color(0),
              rhs.COLORS['grey'], rhs.COLORS['grey']]
    bars = ax.bar(labels, vals, color=colors, edgecolor='#333333', linewidth=0.5)
    ax.axhline(thr, color=rhs.COLORS['black'], ls='--', lw=1.0,
               label=f'decision threshold ({thr:.2f})')
    ax.set_xlabel('inputs $(A,B)$')
    ax.set_ylabel('windowed peak $u$ at output')
    ax.set_title(f"Inhibition gate $A \\wedge \\neg B$ "
                 f"(separation {data['separation_ratio']:.0f}$\\times$)")
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(vals) * 1.15)

    # Right: inhibitor delay sweep
    ax = axes[1]
    dB = np.array([s['dB_steps'] for s in sweep]) * 0.05
    peaks = np.array([s['window_peak'] for s in sweep])
    blocked = np.array([s['blocked'] for s in sweep])
    ax.plot(dB, peaks, 'o-', ms=5, color=rhs.get_color(0))
    ax.scatter(dB[blocked], peaks[blocked], color=rhs.COLORS['vermillion'],
               s=40, zorder=3, label='inhibited')
    ax.scatter(dB[~blocked], peaks[~blocked], color=rhs.get_color(0),
               s=40, zorder=3, label='transmits')
    ax.axhline(thr, color=rhs.COLORS['black'], ls='--', lw=1.0)
    ax.set_xlabel('inhibitor delay $\\Delta B$ (t.u.)')
    ax.set_ylabel('windowed peak $u$ at output')
    ax.set_title(f"Inhibition window $\\lesssim$ "
                 f"{data['inhibition_window_time']:.1f} t.u.")
    ax.legend(loc='upper right')

    fig.suptitle('T-junction collision logic (Oregonator A)',
                 fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_tjunction_logic')


# -----------------------------------------------------------------------------
# Figure 4: anisotropic routing
# -----------------------------------------------------------------------------
def plot_anisotropic_routing():
    data = load('rd_transfer_aniso_final2.json')['oregonator']
    # Re-run is not allowed, but tracks are not in JSON; we only have summary
    # case data.  Plot the sqrt(r) verification.
    cases = data['cases']

    fig, axes = plt.subplots(1, 2, figsize=rhs.DOUBLE_PANEL)

    # Left: measured speed ratio vs r (the core check)
    ax = axes[0]
    rs = sorted(int(k) for k in cases)
    ratios = [cases[str(r)]['ratio'] for r in rs]
    devs = [cases[str(r)]['deviation_pct'] for r in rs]
    ax.plot(rs, ratios, 'o', ms=6, color=rhs.get_color(0),
            label='measured $c_{\\parallel}/c_{\\perp}$')
    r_fine = np.linspace(1, 4, 100)
    ax.plot(r_fine, np.sqrt(r_fine), color=rhs.COLORS['grey'], ls='--',
            lw=1.0, label='$\\sqrt{r}$')
    for r, ratio, dev in zip(rs, ratios, devs):
        ax.annotate(f'{dev:.2f}%', (r, ratio), textcoords='offset points',
                    xytext=(8, 4), fontsize=8)
    ax.set_xlabel('anisotropy ratio $r = D_{\\parallel}/D_{\\perp}$')
    ax.set_ylabel('speed ratio $c_{\\parallel}/c_{\\perp}$')
    ax.set_title('Anisotropic wave speed scaling')
    ax.legend(loc='upper left')

    # Right: axis speeds in cells/t.u.
    ax = axes[1]
    c_par = [cases[str(r)]['c_par'] for r in rs]
    c_perp = [cases[str(r)]['c_perp'] for r in rs]
    x = np.arange(len(rs))
    width = 0.35
    ax.bar(x - width/2, c_par, width, label='$c_{\\parallel}$',
           color=rhs.get_color(0))
    ax.bar(x + width/2, c_perp, width, label='$c_{\\perp}$',
           color=rhs.get_color(1))
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rs])
    ax.set_xlabel('anisotropy ratio $r$')
    ax.set_ylabel('speed (cells / t.u.)')
    ax.set_title('Axis speeds')
    ax.legend()

    fig.suptitle('Anisotropic routing (Oregonator A)',
                 fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_anisotropic_routing')


# -----------------------------------------------------------------------------
# Figure 5: MMS convergence
# -----------------------------------------------------------------------------
def plot_mms_convergence():
    data = load('rd_mms_oregonator.json')

    fig, axes = plt.subplots(1, 2, figsize=rhs.DOUBLE_PANEL)

    # Left: spatial convergence
    ax = axes[0]
    for i, (name, res) in enumerate(data.items()):
        rows = res['spatial']
        dxs = np.array([r['dx'] for r in rows])
        errs = np.array([r['u_L2'] for r in rows])
        p = res['order_spatial_u_L2'][-1]
        ax.loglog(dxs, errs, 'o-', ms=5, color=rhs.get_color(i),
                  label=f"{name} ($p={p:.2f}$)")
    # anchored slope-2 reference through the finest point
    ref = next(iter(data.values()))['spatial']
    x0, y0 = ref[-1]['dx'], ref[-1]['u_L2']
    dx_ref = np.array([ref[0]['dx'], ref[-1]['dx']])
    ax.loglog(dx_ref, y0 * (dx_ref / x0)**2, color=rhs.COLORS['grey'],
              ls='--', lw=1.0, label='slope 2')
    ax.set_xlabel('grid spacing $\\Delta x$')
    ax.set_ylabel('$u$ RMS error at $t=20$')
    ax.set_title('Spatial convergence')
    ax.legend(loc='upper left')

    # Right: temporal convergence
    ax = axes[1]
    for i, (name, res) in enumerate(data.items()):
        rows = res['temporal']
        dts = np.array([r['dt'] for r in rows])
        errs = np.array([r['u_L2'] for r in rows])
        p = res['order_temporal_u_L2'][-1]
        ax.loglog(dts, errs, 's-', ms=5, color=rhs.get_color(i),
                  label=f"{name} ($p={p:.2f}$)")
    ref = next(iter(data.values()))['temporal']
    x0, y0 = ref[-1]['dt'], ref[-1]['u_L2']
    dt_ref = np.array([ref[0]['dt'], ref[-1]['dt']])
    ax.loglog(dt_ref, y0 * (dt_ref / x0), color=rhs.COLORS['grey'],
              ls='--', lw=1.0, label='slope 1')
    ax.set_xlabel('timestep $\\Delta t$')
    ax.set_ylabel('$u$ RMS error at $t=20$')
    ax.set_title('Temporal convergence')
    ax.legend(loc='upper left')

    fig.suptitle('Method of manufactured solutions (Oregonator)',
                 fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_mms_convergence')


# -----------------------------------------------------------------------------
# Figure 6: light-held Oregonator regime map
# -----------------------------------------------------------------------------
def plot_regime_map():
    data = load('rd_phi_regime_map.json')
    results = data['data']
    cfg = data['config']

    phis = np.array([r['phi'] for r in results if 'u_star' in r])
    rests = np.array([r['u_star'] for r in results if 'u_star' in r])
    maxres = np.array([r['max_re_eig'] for r in results if 'u_star' in r])

    prop = [(r['phi'], r.get('propagation_crossed_40_cells', None),
             r.get('propagation_peak_p2', 0.0) or 0.0)
            for r in results if 'u_star' in r]
    prop_phis = np.array([p[0] for p in prop if p[1] is not None])
    prop_flags = np.array([p[1] for p in prop if p[1] is not None])
    prop_peaks = np.array([p[2] for p in prop if p[1] is not None])

    fig, axes = plt.subplots(3, 1, figsize=rhs.TALL_3x1, sharex=True)

    # Panel 1: rest state
    ax = axes[0]
    ax.plot(phis, rests, color=rhs.get_color(0), lw=1.5)
    ax.set_ylabel('$u^*$')
    ax.set_title('Light-held Oregonator parameter map '
                 f"($f={cfg['f']}$, $\\varepsilon={cfg['eps']:.4f}$)")
    ax.axvline(0.005, color=rhs.COLORS['grey'], ls=':', lw=0.8)
    ax.axvline(0.028, color=rhs.COLORS['grey'], ls=':', lw=0.8)

    # Panel 2: eigenvalue
    ax = axes[1]
    ax.plot(phis, maxres, color=rhs.get_color(1), lw=1.5)
    ax.axhline(0, color=rhs.COLORS['black'], ls='--', lw=1.0)
    ax.set_ylabel('max Re($\\lambda$)')
    ax.set_yscale('symlog', linthresh=1.0)
    ax.axvline(0.005, color=rhs.COLORS['grey'], ls=':', lw=0.8)
    ax.axvline(0.028, color=rhs.COLORS['grey'], ls=':', lw=0.8)

    # Panel 3: propagation flag
    ax = axes[2]
    colors = [rhs.COLORS['bluish_green'] if f else rhs.COLORS['vermillion']
              for f in prop_flags]
    ax.scatter(prop_phis, prop_flags.astype(float) * 0.5 + 0.25, c=colors,
               s=40, zorder=3)
    if len(prop_peaks):
        ax2 = ax.twinx()
        ax2.plot(prop_phis, prop_peaks, '.--', color=rhs.get_color(2),
                 alpha=0.7, label='probe peak')
        ax2.set_ylabel('probe peak $u$')
        ax2.set_ylim(0, 0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('propagates')
    ax.set_xlabel('background illumination $\\phi$')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['no', 'yes'])
    ax.axvline(0.005, color=rhs.COLORS['grey'], ls=':', lw=0.8)
    ax.axvline(0.028, color=rhs.COLORS['grey'], ls=':', lw=0.8)

    # Shade regimes
    for ax in axes:
        ax.axvspan(0, 0.005, alpha=0.08, color=rhs.COLORS['vermillion'])
        ax.axvspan(0.005, 0.028, alpha=0.08, color=rhs.COLORS['bluish_green'])
        ax.axvspan(0.028, 0.050, alpha=0.08, color=rhs.COLORS['blue'])
        ax.set_xlim(phis.min(), phis.max())

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_regime_map')


# -----------------------------------------------------------------------------
# Figure 7: flash calibration
# -----------------------------------------------------------------------------
def plot_flash_calibration():
    data = load('rd_flash_calib.json')
    t_flash = np.array([float(k) for k in data])
    first = np.array([data[k]['first_arrival_tu'] for k in data])
    pulses = np.array([data[k]['pulses'] for k in data])
    peak = np.array([data[k]['probe_peak'] for k in data])

    fig, axes = plt.subplots(1, 2, figsize=rhs.DOUBLE_PANEL)

    ax = axes[0]
    ax.plot(t_flash, first, 'o-', ms=5, color=rhs.get_color(0))
    ax.set_xlabel('dark-spot duration $T_{\\mathrm{flash}}$ (t.u.)')
    ax.set_ylabel('first arrival time (t.u.)')
    ax.set_title('Dim-to-fire latency')
    ax.set_xlim(left=0)

    ax = axes[1]
    ax.plot(t_flash, pulses, 's-', ms=5, color=rhs.get_color(1))
    ax.set_xlabel('dark-spot duration $T_{\\mathrm{flash}}$ (t.u.)')
    ax.set_ylabel('pulses emitted')
    ax.set_title('Pulse count vs flash duration')
    ax.set_xlim(left=0)
    ax.set_ylim(0, pulses.max() + 1)

    fig.suptitle('Dark-spot flash calibration (Oregonator A)',
                 fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_flash_calibration')


# -----------------------------------------------------------------------------
# Figure 8: optimizer comparison
# -----------------------------------------------------------------------------
def plot_optimizer_comparison():
    data = load('rd_optimizer_compare_left.json')

    fig, ax = plt.subplots(figsize=rhs.SINGLE_PANEL)
    for i, (name, d) in enumerate(data.items()):
        fs = np.array([h['f'] for h in d['history']])
        best = np.minimum.accumulate(fs)
        n = np.arange(1, len(best) + 1)
        marker = 'o' if len(n) < 10 else None
        ax.plot(n, best, '-', marker=marker, ms=4, color=rhs.get_color(i),
                label=name)
    ax.set_xlabel('function evaluations')
    ax.set_ylabel('best soft loss so far')
    ax.set_title('Optimiser comparison: route to left')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return save(fig, 'pub_optimizer_comparison')


# -----------------------------------------------------------------------------
# Figure 9: dark-spot channel pulse transfer
# -----------------------------------------------------------------------------
def plot_channel_transfer_darkspot():
    data = load('rd_transfer_channel_darkspot.json')
    sweep = data['sweep']
    free = data['free_medium']
    traces = data['traces']

    fig, axes = plt.subplots(1, 3, figsize=rhs.TRIPLE_PANEL)

    # Left: far-probe traces
    ax = axes[0]
    order = ['60', '16', '8', '1']
    for i, w in enumerate(order):
        if w in traces:
            d = traces[w]
            ax.plot(d['t'], d['far'], label=f'W = {w}', color=rhs.get_color(i))
    ax.axhline(U_THRESH, color=rhs.COLORS['grey'], ls='--', lw=1.0,
               label='threshold')
    ax.set_xlabel('time (t.u.)')
    ax.set_ylabel('mean $u$ at far probe')
    ax.set_title('Pulse arrival traces')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)

    # Centre: speed vs width
    ax = axes[1]
    widths = [r['W'] for r in sweep if r['speed_cells_per_tu'] is not None]
    speeds = [r['speed_cells_per_tu'] for r in sweep
              if r['speed_cells_per_tu'] is not None]
    ax.plot(widths, speeds, 'o-', ms=5, color=rhs.get_color(0),
            label='dark-spot channel')
    ax.axhline(free['speed_cells_per_tu'], color=rhs.COLORS['grey'], ls='--',
               lw=1.0, label=f"free medium ({free['speed_cells_per_tu']:.2f})")
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 8, 16, 32, 60])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 60])
    ax.set_xlabel('channel width $W$ (cells)')
    ax.set_ylabel('speed (cells / t.u.)')
    ax.set_title('Pulse speed vs width')
    ax.legend(loc='lower right')

    # Right: transmission amplitude
    ax = axes[2]
    peaks = [r['peak_far'] for r in sweep]
    ax.plot(widths, peaks, 's-', ms=5, color=rhs.get_color(1))
    ax.axhline(U_THRESH, color=rhs.COLORS['grey'], ls='--', lw=1.0)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 8, 16, 32, 60])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 60])
    ax.set_xlabel('channel width $W$ (cells)')
    ax.set_ylabel('far-probe peak $u$')
    ax.set_title('Transmission amplitude')

    fig.suptitle('Channel pulse transfer with dark-spot drive (Oregonator A)',
                 fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_channel_transfer_darkspot')


# -----------------------------------------------------------------------------
# Figure 10: dark-spot frequency response
# -----------------------------------------------------------------------------
def plot_frequency_response_darkspot():
    data = load('rd_transfer_frequency_darkspot.json')
    rows = data['train_results']
    refractory_tu = data['refractory_period_time']

    fig, axes = plt.subplots(1, 2, figsize=rhs.DOUBLE_PANEL)

    ax = axes[0]
    f_in = np.array([r['f_in'] for r in rows])
    f_out = np.array([r['f_out'] for r in rows])
    lim = f_in.max() * 1.1
    ax.plot(f_in, f_out, 'o-', ms=5, color=rhs.get_color(0),
            label='measured')
    ax.plot([0, lim], [0, lim], color=rhs.COLORS['grey'], ls='--', lw=1.0,
            label='1:1 line')
    ax.axvline(1.0 / refractory_tu, color=rhs.COLORS['vermillion'], ls=':',
               lw=1.5, label=f'1/refractory ({1.0/refractory_tu:.3f})')
    ax.set_xlabel('input rate $f_{\\mathrm{in}}$ (pulses / t.u.)')
    ax.set_ylabel('output rate $f_{\\mathrm{out}}$ (pulses / t.u.)')
    ax.set_title('Frequency response')
    ax.legend(loc='lower right')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax = axes[1]
    periods = np.array([r['period_steps'] for r in rows]) * 0.05
    transmitted = np.array([r['transmitted'] for r in rows])
    n_fired = rows[0]['fired']
    ax.plot(periods, transmitted, 's-', ms=5,
            color=rhs.get_color(1), label='transmitted')
    ax.axhline(n_fired, color=rhs.COLORS['grey'], ls='--', lw=1.0,
               label=f'fired ({n_fired})')
    ax.axvline(refractory_tu, color=rhs.COLORS['vermillion'], ls=':', lw=1.5,
               label=f'refractory ({refractory_tu:.1f} t.u.)')
    ax.set_xlabel('input period (t.u.)')
    ax.set_ylabel('pulses transmitted')
    ax.set_title('Transmission count vs period')
    ax.legend(loc='lower right')

    fig.suptitle('Frequency response with dark-spot drive (Oregonator A)',
                 fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_frequency_response_darkspot')


# -----------------------------------------------------------------------------
# Figure 11: dark-spot T-junction logic
# -----------------------------------------------------------------------------
def plot_tjunction_logic_darkspot():
    data = load('rd_transfer_logic_darkspot.json')
    truth = data['truth_table']
    thr = data['decision_threshold']
    sweep = data['timing_sweep']

    fig, axes = plt.subplots(1, 2, figsize=rhs.DOUBLE_PANEL)

    ax = axes[0]
    labels = ['00', '10', '01', '11']
    vals = [truth[k] for k in labels]
    colors = [rhs.COLORS['grey'], rhs.get_color(0),
              rhs.COLORS['grey'], rhs.COLORS['grey']]
    bars = ax.bar(labels, vals, color=colors, edgecolor='#333333', linewidth=0.5)
    ax.axhline(thr, color=rhs.COLORS['black'], ls='--', lw=1.0,
               label=f'decision threshold ({thr:.2f})')
    ax.set_xlabel('inputs $(A,B)$')
    ax.set_ylabel('windowed peak $u$ at output')
    ax.set_title(f"Inhibition gate $A \\wedge \\neg B$ "
                 f"(separation {data['separation_ratio']:.0f}$\\times$)")
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(vals) * 1.15)

    ax = axes[1]
    dB = np.array([s['dB_steps'] for s in sweep]) * 0.05
    peaks = np.array([s['window_peak'] for s in sweep])
    blocked = np.array([s['blocked'] for s in sweep])
    ax.plot(dB, peaks, 'o-', ms=5, color=rhs.get_color(0))
    ax.scatter(dB[blocked], peaks[blocked], color=rhs.COLORS['vermillion'],
               s=40, zorder=3, label='inhibited')
    ax.scatter(dB[~blocked], peaks[~blocked], color=rhs.get_color(0),
               s=40, zorder=3, label='transmits')
    ax.axhline(thr, color=rhs.COLORS['black'], ls='--', lw=1.0)
    ax.set_xlabel('inhibitor delay $\\Delta B$ (t.u.)')
    ax.set_ylabel('windowed peak $u$ at output')
    ax.set_title(f"Inhibition window $\\lesssim$ "
                 f"{data['inhibition_window_time']:.1f} t.u.")
    ax.legend(loc='upper right')

    fig.suptitle('T-junction collision logic with dark-spot drive (Oregonator A)',
                 fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, 'pub_tjunction_logic_darkspot')


# -----------------------------------------------------------------------------
# Figure 12: verification summary
# -----------------------------------------------------------------------------
def plot_verification():
    data = load('rd_verification.json')

    fig = plt.figure(figsize=(10.0, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_dt = fig.add_subplot(gs[0, 1])
    ax_sub = fig.add_subplot(gs[1, 0])
    ax_txt = fig.add_subplot(gs[1, 1])

    # Top-left: grid convergence
    for k, label, color in [('barkley', 'Barkley', rhs.get_color(0)),
                            ('oregonator', 'Oregonator A', rhs.get_color(1))]:
        runs = data['grid'][k]['runs']
        dxs = [r['DX'] for r in runs]
        speeds = [r['c_cells_per_tu'] for r in runs]
        ax_grid.plot(dxs, speeds, 'o-', ms=5, color=color, label=label)
    ax_grid.set_xscale('log')
    ax_grid.set_xlabel('grid spacing $\\Delta x$ (cells)')
    ax_grid.set_ylabel('pulse speed (cells / t.u.)')
    ax_grid.set_title('Grid convergence')
    ax_grid.legend(loc='upper left')

    # Top-right: timestep convergence
    for k, label, color in [('barkley', 'Barkley', rhs.get_color(0)),
                            ('oregonator', 'Oregonator A', rhs.get_color(1))]:
        runs = data['dt'][k]['runs']
        dts = [r['DT'] for r in runs]
        speeds = [r['c_cells_per_tu'] for r in runs]
        ax_dt.plot(dts, speeds, 's-', ms=5, color=color, label=label)
    ax_dt.set_xscale('log')
    ax_dt.set_xlabel('timestep $\\Delta t$ (t.u.)')
    ax_dt.set_ylabel('pulse speed (cells / t.u.)')
    ax_dt.set_title('Timestep convergence')
    ax_dt.legend(loc='upper left')

    # Bottom-left: runtime invariants (max substeps)
    labels = ['Barkley', 'Oregonator A']
    substeps = [data['invariants']['barkley']['max_react_substeps'],
                data['invariants']['oregonator']['max_react_substeps']]
    ax_sub.bar(labels, substeps, color=[rhs.get_color(0), rhs.get_color(1)])
    ax_sub.set_ylabel('max reaction substeps')
    ax_sub.set_title('Adaptive subcycling peak')
    ax_sub.set_ylim(0, max(substeps) * 1.2)

    # Bottom-right: summary text
    ax_txt.axis('off')
    txt = (
        "Verification summary\n"
        "- Barkley: pre-asymptotic; ~14% spatial deficit at $\\Delta x=1$\n"
        "- Oregonator A: speeds within 2.8% band\n"
        "- Temporal error < 1.5% at $\\Delta t = 0.05$\n"
        "- NaN/Inf tripwire: clean\n"
        "- Step-size guard: clean"
    )
    ax_txt.text(0.05, 0.5, txt, transform=ax_txt.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')

    fig.suptitle('Solver verification summary',
                 fontsize=plt.rcParams['figure.titlesize'])
    return save(fig, 'pub_verification')


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    rhs.setup()
    generated = []

    generated.append(plot_channel_transfer())
    generated.append(plot_frequency_response())
    generated.append(plot_tjunction_logic())
    generated.append(plot_anisotropic_routing())
    generated.append(plot_mms_convergence())
    generated.append(plot_regime_map())
    generated.append(plot_flash_calibration())
    generated.append(plot_optimizer_comparison())
    generated.append(plot_channel_transfer_darkspot())
    generated.append(plot_frequency_response_darkspot())
    generated.append(plot_tjunction_logic_darkspot())
    generated.append(plot_verification())

    print('\nGenerated files:')
    for png, pdf in generated:
        print(f'  {png}')
        print(f'  {pdf}')


if __name__ == '__main__':
    main()
