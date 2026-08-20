"""
2-to-1 multiplexer (MUX) in the event-driven graph simulator.

What a 2-to-1 MUX does:
- Two data inputs: A and B.
- One select input: S.
- One output: O.
- O = A if S = 0, O = B if S = 1.

In pulse logic, S is a control pulse that arrives at a switch node before the
data pulses.  The switch node remembers which input was selected and routes the
corresponding data pulse to O.

This script tests all eight input patterns (S, A, B) and reports the truth
table.  It is a pure graph-simulation test; no PDE is run.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Measured dark-spot transfer parameters (Oregonator A, phi0=0.010)
# ---------------------------------------------------------------------------
C_SPEED = 6.46411888154886      # cells / t.u.
TAU_REFRACT = 3.0               # t.u.
W_INHIBIT = 1.0                 # t.u.
PULSE_AMP = 0.7264304595123359

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


@dataclass
class Pulse:
    t_arrive: float
    src: str
    dst: str
    amp: float = PULSE_AMP


@dataclass
class Edge:
    name: str
    src: str
    dst: str
    length: float
    speed: float = C_SPEED

    def transit(self) -> float:
        return self.length / self.speed


@dataclass
class SwitchNode:
    name: str
    state: str = 'A'   # 'A' or 'B': which data input is currently selected
    refractory_until: float = 0.0
    recent_arrivals: List[Pulse] = field(default_factory=list)


def run_mux(s: bool, a: bool, b: bool, t_max: float = 60.0):
    """Run one input pattern.  s, a, b are booleans indicating whether a
    pulse is injected at that input at t=0."""
    # Nodes
    switch = SwitchNode('switch')
    output_log = []

    # Edges: control path is shorter so S arrives first
    edges = {
        'S->switch': Edge('S->switch', 'S', 'switch', 10.0),
        'A->switch': Edge('A->switch', 'A', 'switch', 30.0),
        'B->switch': Edge('B->switch', 'B', 'switch', 30.0),
        'switch->O': Edge('switch->O', 'switch', 'O', 20.0),
    }

    # Initial events
    events: List[Pulse] = []
    if s:
        e = edges['S->switch']
        events.append(Pulse(e.transit(), 'S', 'switch'))
    if a:
        e = edges['A->switch']
        events.append(Pulse(e.transit(), 'A', 'switch'))
    if b:
        e = edges['B->switch']
        events.append(Pulse(e.transit(), 'B', 'switch'))

    events.sort(key=lambda p: p.t_arrive)

    while events:
        p = events.pop(0)
        if p.t_arrive > t_max:
            continue

        if p.dst == 'switch':
            # Control pulse arms the switch to select B
            if p.src == 'S':
                switch.state = 'B'
                continue

            # Data pulse: forward only if it matches the selected input
            if p.t_arrive < switch.refractory_until:
                continue
            if p.src != switch.state:
                continue

            # Forward to output
            switch.refractory_until = p.t_arrive + TAU_REFRACT
            e = edges['switch->O']
            output_log.append({
                't': p.t_arrive + e.transit(),
                'selected': p.src,
                'amp': p.amp
            })

    # Peak in output window (around expected arrival)
    expected = edges['A->switch'].transit() + edges['switch->O'].transit()
    peak = max((ev['amp'] for ev in output_log
                if abs(ev['t'] - expected) <= 2.5), default=0.0)
    return peak, output_log, switch.state


def main():
    patterns = []
    for s in [0, 1]:
        for a in [0, 1]:
            for b in [0, 1]:
                peak, log, final_state = run_mux(bool(s), bool(a), bool(b))
                patterns.append({
                    'S': s, 'A': a, 'B': b,
                    'selected': final_state,
                    'output_peak': peak,
                    'output_fires': peak > 0.5,
                    'events': log
                })

    out = {
        'test': '2-to-1 multiplexer (graph simulator)',
        'parameters': {
            'channel_speed': C_SPEED,
            'refractory_time': TAU_REFRACT,
            'inhibition_window': W_INHIBIT,
            'pulse_amplitude': PULSE_AMP
        },
        'truth_table': patterns
    }

    json_path = os.path.join(FIG, 'rd_mux_graph.json')
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=2)

    # Plot truth table
    rows = [(p['S'], p['A'], p['B'], int(p['output_fires']), p['output_peak'])
            for p in patterns]
    fig, ax = plt.subplots(figsize=(8, 4))
    table_data = [['S', 'A', 'B', 'O fires', 'O peak']] + rows
    ax.axis('off')
    ax.table(cellText=table_data, loc='center', cellLoc='center')
    ax.set_title('2-to-1 multiplexer truth table (graph simulator)')
    fig.savefig(os.path.join(FIG, 'rd_mux_graph.png'), dpi=150)
    plt.close(fig)

    print(json.dumps(out, indent=2))
    print(f'Saved {json_path}')


if __name__ == '__main__':
    main()
