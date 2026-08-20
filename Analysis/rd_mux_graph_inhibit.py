"""
Length-aware inhibition-based 2-to-1 multiplexer graph simulator.

This version models the PDE physics more faithfully than the state-machine MUX:
- A and S meet at an inhibition node (the T-junction).
- If S arrives within the refractory window before A, A is blocked.
- B joins the output downstream and is never affected by S.

The script:
1. Runs the full truth table for a given set of channel lengths.
2. Sweeps the S and A channel lengths to find the feasible inhibition window.

Channel lengths are in cells; delays are length / channel_speed.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Measured dark-spot transfer parameters (Oregonator A, phi0=0.010)
# ---------------------------------------------------------------------------
C_SPEED = 6.46411888154886      # cells / t.u.
TAU_REFRACT = 3.0               # t.u. (junction stays refractory after A passes)
W_INHIBIT = 1.0                 # t.u. (control must lead data by this much)
PULSE_AMP = 0.7264304595123359

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


@dataclass
class Edge:
    src: str
    dst: str
    length: float          # cells
    speed: float = C_SPEED

    def delay(self) -> float:
        return self.length / self.speed


def run_mux_inhibit(s: bool, a: bool, b: bool,
                    ls: float, la: float, lb: float, lo: float,
                    t_max: float = 80.0):
    """Run one input pattern on the inhibition-based MUX.

    Geometry:
        S --ls-->|
                 | inhibition node --lo--> O
        A --la-->|
        B -------------lb----------------> O
    """
    # Build events at t=0 injections
    events = []
    if s:
        events.append(('S', ls / C_SPEED))
    if a:
        events.append(('A', la / C_SPEED))
    if b:
        events.append(('B', lb / C_SPEED))

    # Sort by arrival time at the inhibition node (S, A) or output (B)
    events.sort(key=lambda x: x[1])

    refractory_until = 0.0
    output_events = []

    for src, t_arr in events:
        if t_arr > t_max:
            continue

        if src == 'S':
            # S arms inhibition: node is refractory until t_arr + TAU_REFRACT
            refractory_until = max(refractory_until, t_arr + TAU_REFRACT)

        elif src == 'A':
            # A passes only if it arrives outside the refractory shadow
            if t_arr >= refractory_until:
                output_events.append({
                    't': t_arr + lo / C_SPEED,
                    'src': 'A',
                    'amp': PULSE_AMP
                })
                refractory_until = max(refractory_until, t_arr + TAU_REFRACT)

        elif src == 'B':
            # B is routed around the inhibition node
            output_events.append({
                't': t_arr + lo / C_SPEED,
                'src': 'B',
                'amp': PULSE_AMP
            })

    # Output peak within expected arrival window
    expected_a = (la + lo) / C_SPEED
    peak = max((ev['amp'] for ev in output_events
                if abs(ev['t'] - expected_a) <= 3.0), default=0.0)
    return peak, output_events


def truth_table(ls: float, la: float, lb: float, lo: float) -> dict:
    """Run all eight patterns and return structured result.

    NOTE: This model only gates A with S.  B always passes, so it is NOT a
    true MUX for arbitrary (A,B).  We therefore judge correctness on the
    one-hot cases plus the A-block case:
      S=0, A=1 -> A passes
      S=1, A=1 -> A blocked
      any S, B=1 -> B passes
    """
    patterns = []
    for s in [0, 1]:
        for a in [0, 1]:
            for b in [0, 1]:
                peak, events = run_mux_inhibit(bool(s), bool(a), bool(b),
                                               ls, la, lb, lo)
                patterns.append({
                    'S': s, 'A': a, 'B': b,
                    'output_peak': peak,
                    'output_fires': peak > 0.5,
                    'events': events
                })

    # Correctness for an A-inhibit / priority-router building block
    checks = [
        (0, 1, 0, True),   # S=0, A=1 -> A passes
        (1, 1, 0, False),  # S=1, A=1 -> A blocked
        (0, 0, 1, True),   # S=0, B=1 -> B passes
        (1, 0, 1, True),   # S=1, B=1 -> B passes
    ]
    correct = True
    for s, a, b, expected in checks:
        p = next((x for x in patterns if x['S'] == s and x['A'] == a and x['B'] == b), None)
        if p is None or int(p['output_fires']) != int(expected):
            correct = False
            break

    return {
        'ls': ls, 'la': la, 'lb': lb, 'lo': lo,
        'delta_tu': (la - ls) / C_SPEED,
        'correct': correct,
        'patterns': patterns
    }


def sweep(ls_min: float = 5.0, ls_max: float = 40.0, n_ls: int = 36,
          la_min: float = 15.0, la_max: float = 60.0, n_la: int = 46,
          lb: float = 30.0, lo: float = 20.0):
    """Sweep S and A channel lengths, mark correct/incorrect MUX behaviour."""
    ls_vals = np.linspace(ls_min, ls_max, n_ls)
    la_vals = np.linspace(la_min, la_max, n_la)

    correct = np.zeros((n_la, n_ls), dtype=bool)
    delta = np.zeros((n_la, n_ls))

    for i, la in enumerate(la_vals):
        for j, ls in enumerate(ls_vals):
            res = truth_table(ls, la, lb, lo)
            correct[i, j] = res['correct']
            delta[i, j] = res['delta_tu']

    return ls_vals, la_vals, correct, delta


def main():
    # ------------------------------------------------------------------
    # Single design-point truth table
    # ------------------------------------------------------------------
    ls, la, lb, lo = 20.0, 35.0, 30.0, 20.0
    single = truth_table(ls, la, lb, lo)
    print('Single design-point truth table:')
    print(json.dumps(single, indent=2))

    json_path = os.path.join(FIG, 'rd_mux_graph_inhibit.json')
    with open(json_path, 'w') as fh:
        json.dump(single, fh, indent=2)
    print(f'Saved {json_path}')

    # ------------------------------------------------------------------
    # Length sweep
    # ------------------------------------------------------------------
    ls_vals, la_vals, correct, delta = sweep()

    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [ls_vals[0], ls_vals[-1], la_vals[0], la_vals[-1]]
    im = ax.imshow(correct.astype(float), origin='lower', extent=extent,
                   aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    cs = ax.contour(ls_vals, la_vals, delta, levels=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                    colors='k', linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt='Δt=%1.1f')
    ax.set_xlabel('S channel length L_s (cells)')
    ax.set_ylabel('A channel length L_a (cells)')
    ax.set_title('MUX correctness vs channel lengths (green = correct)')
    fig.colorbar(im, ax=ax, label='correct (1) / incorrect (0)')
    fig.tight_layout()
    sweep_path = os.path.join(FIG, 'rd_mux_graph_inhibit_sweep.png')
    fig.savefig(sweep_path, dpi=150)
    plt.close(fig)
    print(f'Saved {sweep_path}')

    # Save sweep data
    sweep_json = os.path.join(FIG, 'rd_mux_graph_inhibit_sweep.json')
    with open(sweep_json, 'w') as fh:
        json.dump({
            'ls_vals': ls_vals.tolist(),
            'la_vals': la_vals.tolist(),
            'correct': correct.tolist(),
            'delta_tu': delta.tolist(),
            'parameters': {
                'channel_speed': C_SPEED,
                'refractory_time': TAU_REFRACT,
                'lb': lb,
                'lo': lo
            }
        }, fh, indent=2)
    print(f'Saved {sweep_json}')


if __name__ == '__main__':
    main()
