"""
2D FitzHugh-Nagumo reaction-diffusion demo in a channel geometry.

FHN produces robust traveling pulses, making it a good proof-of-concept
for wave-based computation in Project 11.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NX, NY = 300, 150          # grid size
DX = 1.0                   # spatial step
DT = 0.05                  # time step
NSTEPS = 6000              # total steps
PLOT_EVERY = 1500          # snapshot interval

Du = 1.0                   # diffusion of activator u
Dv = 0.1                   # diffusion of inhibitor v
A = 0.1                    # excitation threshold
B = 0.5                    # recovery scale
EPS = 0.01                 # time-scale separation

# Channel geometry: walls as boolean mask
mask = np.zeros((NX, NY), dtype=bool)
mask[:, 0] = True
mask[:, -1] = True

# Two central obstacles to force routing
mask[100:110, 30:70] = True
mask[100:110, 80:120] = True

# Input ports on left boundary
port1_y = range(35, 55)
port2_y = range(95, 115)
input_amp = 1.0            # u amplitude injected
input_width = 50           # number of steps input is active

# Readout region on right boundary
probe_y = range(60, 90)
probe_x = range(290, 300)


def laplacian(c, mask):
    """5-point Laplacian with Neumann boundaries and no-flux walls."""
    out = np.zeros_like(c)
    out[1:-1, 1:-1] = (
        c[:-2, 1:-1] + c[2:, 1:-1] +
        c[1:-1, :-2] + c[1:-1, 2:] -
        4 * c[1:-1, 1:-1]
    ) / DX**2

    # Neumann on outer boundaries
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]

    out[mask] = 0.0
    return out


def step(u, v, mask):
    """One explicit Euler step of FHN."""
    Lu = laplacian(u, mask)
    Lv = laplacian(v, mask)

    reaction = u * (A - u) * (u - 1.0)

    du = Du * Lu + reaction - v
    dv = Dv * Lv + EPS * (u - B * v)

    u += DT * du
    v += DT * dv

    u[mask] = 0.0
    v[mask] = 0.0

    return u, v


def run_case(port1_on=False, port2_on=False):
    """Run one case and return final field, probe trace, snapshots."""
    u = np.zeros((NX, NY), dtype=float)
    v = np.zeros((NX, NY), dtype=float)

    probe_trace = []
    snapshots = []
    snapshot_times = []

    for n in range(NSTEPS):
        # Stimulate input ports
        if n < input_width:
            if port1_on:
                u[0:5, port1_y] = input_amp
            if port2_on:
                u[0:5, port2_y] = input_amp

        u, v = step(u, v, mask)

        probe_trace.append(float(u[probe_x, :][:, probe_y].mean()))

        if (n + 1) % PLOT_EVERY == 0:
            snapshots.append(u.copy())
            snapshot_times.append((n + 1) * DT)

    return u, np.array(probe_trace), snapshots, snapshot_times


def plot_case(u, trace, snapshots, times, title, filename):
    """Create a figure with snapshots and probe trace."""
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 3)

    for i, (snap, t) in enumerate(zip(snapshots, times)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow(snap.T, origin='lower', cmap='hot', vmin=-0.1, vmax=1.0)
        ax.set_title(f't = {t:.0f}')
        ax.set_xticks([])
        ax.set_yticks([])
        wall_y, wall_x = np.where(mask.T)
        ax.scatter(wall_x, wall_y, c='cyan', s=1, alpha=0.4)

    ax_trace = fig.add_subplot(gs[:, 2])
    ax_trace.plot(trace, color='tab:red', linewidth=1.5)
    ax_trace.axhline(0.2, color='k', linestyle='--', alpha=0.4)
    ax_trace.set_xlabel('time step')
    ax_trace.set_ylabel('probe u')
    ax_trace.set_title(title)
    ax_trace.set_xlim(0, len(trace))
    ax_trace.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f'Saved {filename}')


if __name__ == '__main__':
    cases = [
        (False, False, 'no input'),
        (True, False, 'port 1 ON'),
        (False, True, 'port 2 ON'),
        (True, True, 'both ports ON'),
    ]

    results = {}
    for p1, p2, label in cases:
        print(f'Running: {label}')
        u, trace, snaps, times = run_case(p1, p2)
        results[label] = (u, trace, snaps, times)
        final_probe = trace[-1]
        print(f'  final probe u = {final_probe:.4f}')

        plot_case(
            u, trace, snaps, times,
            title=f'{label}: final probe u = {final_probe:.3f}',
            filename=f'Analysis/fhn_{label.replace(" ", "_")}.png'
        )

    # Summary
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [f'P1={int(p1)} P2={int(p2)}' for p1, p2, _ in cases]
    finals = [results[label][1][-1] for _, _, label in cases]
    colors = ['gray', 'tab:green', 'tab:green', 'tab:red']
    ax.bar(labels, finals, color=colors)
    ax.set_ylabel('final probe u')
    ax.set_title('FHN channel readout (target: high for single input)')
    ax.axhline(0.2, color='k', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('Analysis/fhn_summary.png', dpi=150)
    print('Saved Analysis/fhn_summary.png')
