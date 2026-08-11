"""
Remake key figures for the audit report with a consistent style.

Reads existing JSON data where available and writes *_audit.png figures.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_style
from audit_style import COLORS, save

FIG = os.path.dirname(os.path.abspath(__file__))

def regime_map():
    with open(os.path.join(FIG, 'rd_phi_regime_map.json')) as fh:
        data = json.load(fh)['data']
    phis = np.array([r['phi'] for r in data if 'u_star' in r])
    rests = np.array([r['u_star'] for r in data if 'u_star' in r])
    maxres = np.array([r['max_re_eig'] for r in data if 'u_star' in r])
    prop = [(r['phi'], r.get('propagation_crossed_40_cells', None),
             r.get('propagation_peak_p2', 0.0)) for r in data if 'u_star' in r]
    prop_phis = np.array([p[0] for p in prop if p[1] is not None])
    prop_peaks = np.array([p[2] for p in prop if p[1] is not None])
    prop_flags = np.array([p[1] for p in prop if p[1] is not None])

    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    axes[0].plot(phis, rests, '.-', color=COLORS['blue'])
    axes[0].set_ylabel(r'rest state $u^*$')
    axes[1].plot(phis, maxres, '.-', color=COLORS['orange'])
    axes[1].axhline(0, color=COLORS['red'], ls='--', lw=1)
    axes[1].set_ylabel(r'max Re($\lambda$)')
    axes[1].set_yscale('symlog', linthresh=1.0)
    colors = [COLORS['green'] if f else COLORS['red'] for f in prop_flags]
    axes[2].scatter(prop_phis, prop_flags.astype(float) * 0.5 + 0.25, c=colors, s=40, zorder=3)
    if len(prop_peaks):
        ax2 = axes[2].twinx()
        ax2.plot(prop_phis, prop_peaks, '.--', color=COLORS['cyan'], alpha=0.7)
        ax2.set_ylabel(r'probe peak $u$')
        ax2.set_ylim(0, 0.8)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('propagates')
    axes[2].set_xlabel(r'background illumination $\phi$')
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['no', 'yes'])
    for ax in axes:
        ax.axvspan(0, 0.005, alpha=0.08, color=COLORS['red'])
        ax.axvspan(0.005, 0.028, alpha=0.08, color=COLORS['green'])
        ax.axvspan(0.028, 0.050, alpha=0.08, color=COLORS['blue'])
        ax.set_xlim(phis.min(), phis.max())
    plt.tight_layout()
    save(fig, 'rd_phi_regime_map_audit')
    plt.close(fig)


def tjunc_hand():
    sys.path.insert(0, os.path.join(FIG, '..'))
    from rd_tjunc_router import build_phi, WALL, NX, NY
    with open(os.path.join(FIG, 'rd_tjunc_hand_tests.json')) as fh:
        data = json.load(fh)
    names = ['uniform_phi0', 'left_block_high', 'right_block_high', 'top_block_high']
    titles = ['uniform', 'left block', 'right block', 'top block']
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes = axes.ravel()
    for ax, name, title in zip(axes, names, titles):
        blocks = data[name]['blocks']
        phi = build_phi(blocks)
        im = ax.imshow(phi.T, origin='lower', vmin=0.010, vmax=0.040,
                       cmap='viridis', extent=[0, NX, 0, NY])
        wall_x, wall_y = np.where(WALL)
        ax.scatter(wall_x, wall_y, c='k', s=0.5, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    save(fig, 'rd_tjunc_hand_phi_maps_audit')
    plt.close(fig)


def tjunc_trained():
    sys.path.insert(0, os.path.join(FIG, '..'))
    from rd_tjunc_router import build_phi, WALL, NX, NY, run_pattern
    with open(os.path.join(FIG, 'rd_optimizer_compare_left.json')) as fh:
        data = json.load(fh)
    best_name = min(data, key=lambda k: data[k]['best_f'])
    blocks = np.array(data[best_name]['best_x'])
    phi = build_phi(blocks)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ax = axes[0]
    im = ax.imshow(phi.T, origin='lower', vmin=0.010, vmax=0.040,
                   cmap='viridis', extent=[0, NX, 0, NY])
    wall_x, wall_y = np.where(WALL)
    ax.scatter(wall_x, wall_y, c='k', s=0.5, alpha=0.4)
    ax.set_title(f'trained phi ({best_name})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046)
    t, left, right = run_pattern(phi, True, False)
    ax = axes[1]
    ax.plot(t, left, label='left probe', color=COLORS['blue'])
    ax.plot(t, right, label='right probe', color=COLORS['orange'])
    ax.axhline(0.5, color=COLORS['red'], ls='--', lw=1, label='threshold')
    ax.set_xlabel('time (t.u.)')
    ax.set_ylabel(r'max $u$ in probe')
    ax.set_title('probe response to input A')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, 'rd_tjunc_trained_audit')
    plt.close(fig)


def optimizer_compare():
    with open(os.path.join(FIG, 'rd_optimizer_compare_left.json')) as fh:
        data = json.load(fh)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in zip(['random', 'DE', 'dual_annealing', 'CMA-ES'],
                           [COLORS['grey'], COLORS['blue'], COLORS['orange'], COLORS['green']]):
        hist = data[name]['history']
        fs = [h['f'] for h in hist]
        best = np.minimum.accumulate(fs)
        ax.plot(range(1, len(best)+1), best, 'o-', label=name, color=color, ms=3)
    ax.set_xlabel('forward evaluations')
    ax.set_ylabel('best soft loss')
    ax.set_title('optimiser comparison: route input A to left probe')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, 'rd_optimizer_compare_left_audit')
    plt.close(fig)


def tjunc_frequency():
    with open(os.path.join(FIG, 'rd_tjunc_frequency.json')) as fh:
        data = json.load(fh)['results']
    periods = np.array([r['period_tu'] for r in data])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(periods, [r['left_peak'] for r in data], 'o-', label='left probe', color=COLORS['blue'])
    axes[0].plot(periods, [r['right_peak'] for r in data], 's-', label='right probe', color=COLORS['orange'])
    axes[0].axhline(0.5, color=COLORS['red'], ls='--', lw=1)
    axes[0].set_xlabel('input pulse period (t.u.)')
    axes[0].set_ylabel('peak $u$')
    axes[0].set_title('T-junction frequency response')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(periods, [r['left_crossings'] for r in data], 'o-', label='left crossings', color=COLORS['blue'])
    axes[1].plot(periods, [r['right_crossings'] for r in data], 's-', label='right crossings', color=COLORS['orange'])
    axes[1].set_xlabel('input pulse period (t.u.)')
    axes[1].set_ylabel('threshold crossings')
    axes[1].set_title('pulse count vs input period')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, 'rd_tjunc_frequency_audit')
    plt.close(fig)


def tjunc_frequency_tflash_2d():
    with open(os.path.join(FIG, 'rd_tjunc_frequency_tflash_2d.json')) as fh:
        d = json.load(fh)
    periods = d['periods']
    tflashes = d['tflashes']
    matrix = np.array([[r['left_crossings'] for r in row] for row in d['results']])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(matrix, aspect='auto', origin='lower',
                   extent=[periods[0]-0.5, periods[-1]+0.5, tflashes[0]-0.5, tflashes[-1]+0.5],
                   vmin=0, vmax=6, cmap='RdYlGn')
    ax.set_xlabel('input period (t.u.)')
    ax.set_ylabel(r'$T_{\mathrm{flash}}$ (t.u.)')
    ax.set_title('left-probe pulse count vs period and flash duration')
    for i, tf in enumerate(tflashes):
        for j, p in enumerate(periods):
            ax.text(p, tf, str(matrix[i, j]), ha='center', va='center', color='k', fontsize=8)
    plt.colorbar(im, ax=ax, label='left crossings', ticks=range(7))
    plt.tight_layout()
    save(fig, 'rd_tjunc_frequency_tflash_2d_audit')
    plt.close(fig)


def transfer_channel():
    with open(os.path.join(FIG, 'rd_transfer_channel_oregonator.json')) as fh:
        d = json.load(fh)
    sweep = d['sweep']
    widths = np.array([r['W'] for r in sweep])
    speeds = np.array([r['speed'] for r in sweep])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(widths, speeds, 'o-', color=COLORS['blue'])
    ax.axhline(d['free_medium']['speed'], color=COLORS['red'], ls='--', lw=1,
               label='free medium speed')
    ax.set_xlabel('channel width $W$ (cells)')
    ax.set_ylabel('pulse speed (cells / t.u.)')
    ax.set_title('Oregonator channel pulse transfer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, 'rd_transfer_channel_audit')
    plt.close(fig)


def transfer_frequency():
    with open(os.path.join(FIG, 'rd_transfer_frequency_final2_oregonator.json')) as fh:
        d = json.load(fh)
    train = d['train_results']
    f_in = np.array([r['f_in'] for r in train])
    f_out = np.array([r['f_out'] for r in train])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(f_in, f_out, 'o-', color=COLORS['blue'], label='measured')
    ax.plot([0, f_in.max()], [0, f_in.max()], 'k--', lw=1, label='1:1')
    ax.set_xlabel('input frequency (1 / t.u.)')
    ax.set_ylabel('output frequency (1 / t.u.)')
    ax.set_title('Oregonator frequency response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, 'rd_transfer_frequency_audit')
    plt.close(fig)


def transfer_logic():
    with open(os.path.join(FIG, 'rd_transfer_logic_final2_oregonator.json')) as fh:
        d = json.load(fh)
    labels = ['00', '10', '01', '11']
    vals = [d['truth_table'][l] for l in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=[COLORS['grey'], COLORS['green'], COLORS['red'], COLORS['red']])
    ax.axhline(d['decision_threshold'], color='k', ls='--', lw=1, label='decision threshold')
    ax.set_ylabel('windowed probe peak $u$')
    ax.set_title('Oregonator T-junction logic: A AND (NOT B)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save(fig, 'rd_transfer_logic_audit')
    plt.close(fig)


def transfer_aniso():
    with open(os.path.join(FIG, 'rd_transfer_aniso_final2.json')) as fh:
        d = json.load(fh)
    oreg = d['oregonator']['cases']
    ratios = np.array([oreg[k]['sqrt_r'] for k in oreg])
    speed_ratios = np.array([oreg[k]['ratio'] for k in oreg])
    devs = np.array([oreg[k]['deviation_pct'] for k in oreg])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(ratios, speed_ratios, 'o-', color=COLORS['blue'], label='measured')
    axes[0].plot(ratios, ratios, 'k--', lw=1, label=r'$c_\parallel / c_\perp = \sqrt{r}$')
    axes[0].set_xlabel(r'anisotropy ratio $\sqrt{r}$')
    axes[0].set_ylabel(r'speed ratio $c_\parallel / c_\perp$')
    axes[0].set_title('Oregonator anisotropic speed law')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(range(len(devs)), devs, color=COLORS['orange'])
    axes[1].set_xticks(range(len(devs)))
    axes[1].set_xticklabels([f'$\\sqrt{{r}}={r:.2f}$' for r in ratios])
    axes[1].set_ylabel(r'deviation from $\sqrt{r}$ (%)')
    axes[1].set_title('eikonal law deviation')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save(fig, 'rd_transfer_aniso_audit')
    plt.close(fig)


if __name__ == '__main__':
    audit_style.setup()
    regime_map()
    tjunc_hand()
    tjunc_trained()
    optimizer_compare()
    tjunc_frequency()
    tjunc_frequency_tflash_2d()
    transfer_channel()
    transfer_frequency()
    transfer_logic()
    transfer_aniso()
    print('[done] all audit figures remade')
