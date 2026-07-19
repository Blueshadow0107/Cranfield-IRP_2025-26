"""
2D Gray-Scott reaction-diffusion demo in a channel geometry.

This is a quick proof-of-concept for Project 11: a spatial, continuous
reaction-diffusion medium whose patterns can be read out as computation.

The domain is a rectangular channel with no-flux walls.  Two input ports
on the left can inject the autocatalyst V.  The readout is the average V
concentration at a probe region on the right.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NX, NY = 256, 128          # grid size
DX = 1.0                   # spatial step
DT = 1.0                   # time step
NSTEPS = 8000              # total steps
PLOT_EVERY = 2000          # snapshot interval

Du = 0.16                  # diffusion of U
Dv = 0.08                  # diffusion of V
F = 0.035                  # feed rate
K = 0.065                  # kill rate

# Channel geometry: walls as a boolean mask (True = wall/no-flux)
mask = np.zeros((NX, NY), dtype=bool)
# Top and bottom walls
mask[:, 0] = True
mask[:, -1] = True
# A few obstacles / channel constrictions to make routing non-trivial
mask[80:90, 20:60] = True
mask[80:90, 70:108] = True

# Input ports (left boundary)
port1_y = range(30, 46)
port2_y = range(82, 98)
input_value = 0.5          # V concentration injected when port is ON

# Readout region (right boundary)
probe_y = range(50, 78)
probe_x = range(245, 256)


def laplacian(c, mask):
    """5-point Laplacian with Neumann (no-flux) boundaries and walls."""
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

    # No flux at walls: copy Laplacian from nearest fluid cell
    # (simplest treatment for this demo)
    out[mask] = 0.0
    return out


def step(U, V, mask):
    """One explicit Euler step of Gray-Scott."""
    Lu = laplacian(U, mask)
    Lv = laplacian(V, mask)

    uvv = U * V * V

    dU = Du * Lu - uvv + F * (1.0 - U)
    dV = Dv * Lv + uvv - (F + K) * V

    U += DT * dU
    V += DT * dV

    # Enforce no-flux walls by mirroring
    U[mask] = 0.0
    V[mask] = 0.0

    return U, V


def run_case(port1_on=False, port2_on=False):
    """Run a single simulation case and return trajectory + final field."""
    U = np.ones((NX, NY), dtype=float)
    V = np.zeros((NX, NY), dtype=float)

    # Seed a small region near left ports with V
    if port1_on:
        U[0:5, port1_y] = 0.5
        V[0:5, port1_y] = input_value
    if port2_on:
        U[0:5, port2_y] = 0.5
        V[0:5, port2_y] = input_value

    probe_trace = []
    snapshots = []
    snapshot_times = []

    for n in range(NSTEPS):
        U, V = step(U, V, mask)

        # Inject sustained input at ports
        if port1_on:
            V[0:3, port1_y] = input_value
            U[0:3, port1_y] = 0.5
        if port2_on:
            V[0:3, port2_y] = input_value
            U[0:3, port2_y] = 0.5

        probe_trace.append(float(V[probe_x, :][:, probe_y].mean()))

        if (n + 1) % PLOT_EVERY == 0:
            snapshots.append(V.copy())
            snapshot_times.append((n + 1) * DT)

    return V, np.array(probe_trace), snapshots, snapshot_times


def plot_case(V, trace, snapshots, times, title, filename):
    """Create a figure with snapshots and probe trace."""
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 3)

    for i, (snap, t) in enumerate(zip(snapshots, times)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow(snap.T, origin='lower', cmap='inferno', vmin=0, vmax=0.5)
        ax.set_title(f't = {t:.0f}')
        ax.set_xticks([])
        ax.set_yticks([])
        # Draw walls
        wall_y, wall_x = np.where(mask.T)
        ax.scatter(wall_x, wall_y, c='cyan', s=1, alpha=0.3)

    ax_trace = fig.add_subplot(gs[:, 2])
    ax_trace.plot(trace, color='tab:red', linewidth=1.5)
    ax_trace.axhline(0.1, color='k', linestyle='--', alpha=0.4)
    ax_trace.set_xlabel('time step')
    ax_trace.set_ylabel('probe V')
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
        V, trace, snaps, times = run_case(p1, p2)
        results[label] = (V, trace, snaps, times)
        final_probe = trace[-1]
        print(f'  final probe V = {final_probe:.4f}')

        plot_case(
            V, trace, snaps, times,
            title=f'{label}: final probe V = {final_probe:.3f}',
            filename=f'Analysis/gray_scott_{label.replace(" ", "_")}.png'
        )

    # Summary truth-table style plot
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [f'P1={int(p1)} P2={int(p2)}' for p1, p2, _ in cases]
    finals = [results[label][1][-1] for _, _, label in cases]
    ax.bar(labels, finals, color=['gray', 'tab:green', 'tab:green', 'tab:red'])
    ax.set_ylabel('final probe V')
    ax.set_title('Gray-Scott channel readout (XOR-like target: high for single input)')
    ax.axhline(0.1, color='k', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('Analysis/gray_scott_summary.png', dpi=150)
    print('Saved Analysis/gray_scott_summary.png')
