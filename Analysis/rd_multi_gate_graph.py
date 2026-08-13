"""
rd_multi_gate_graph.py -- shared-control selective gate (multi-junction network).

Network topology::

    A (control) ------+--------------+
                      |              |
                      v              v
                     J1             J2
                    /  \\           /  \\
    B1 (data) ---->/    O1   B2 -->/    O2

Function:
- A quiet, B1/B2 fire: their pulses pass to O1/O2.
- A fires: it reaches J1 and J2 first, making them refractory; B1/B2 are blocked.

This tests whether the measured inhibition operator composes across a
network with one control signal gating two independent data paths.

Uses parameters from rd_graph_sim.py (dark-spot Oregonator, phi0=0.010).
"""

import json
import os

from rd_graph_sim import Circuit, C_SPEED, TAU_REFRACT, W_INHIBIT, PULSE_AMP

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


def build_shared_control_gate(a_to_j1: float = 50.0,
                              a_to_j2: float = 60.0,
                              b1_to_j1: float = 70.0,
                              b2_to_j2: float = 80.0,
                              j1_to_o1: float = 40.0,
                              j2_to_o2: float = 40.0) -> Circuit:
    """
    Build the shared-control gate.  Lengths are chosen so that A reaches
    each junction before the corresponding data pulse arrives, giving a
    refractory block when A is active.
    """
    circ = Circuit(tau_refract=TAU_REFRACT, w_inhibit=W_INHIBIT)
    for n in ('A', 'B1', 'B2', 'J1', 'J2', 'O1', 'O2'):
        circ.add_node(n)

    # control fan-out
    circ.add_edge('A->J1', 'A', 'J1', a_to_j1)
    circ.add_edge('A->J2', 'A', 'J2', a_to_j2)
    # data paths
    circ.add_edge('B1->J1', 'B1', 'J1', b1_to_j1)
    circ.add_edge('B2->J2', 'B2', 'J2', b2_to_j2)
    # outputs
    circ.add_edge('J1->O1', 'J1', 'O1', j1_to_o1)
    circ.add_edge('J2->O2', 'J2', 'O2', j2_to_o2)

    # routing tables: data passes through; control just blocks
    circ.nodes['J1'].routing = {
        'A->J1': [],
        'B1->J1': ['J1->O1'],
        'J1->O1': [],
    }
    circ.nodes['J2'].routing = {
        'A->J2': [],
        'B2->J2': ['J2->O2'],
        'J2->O2': [],
    }
    return circ


def run_all_patterns(circ):
    """Run all 8 input patterns for (A, B1, B2)."""
    patterns = [
        (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1),
        (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1),
    ]
    results = {}
    for a, b1, b2 in patterns:
        circ.reset()
        if a:
            circ.schedule(0.0, 'A', 'J1')
            circ.schedule(0.0, 'A', 'J2')
        if b1:
            circ.schedule(0.0, 'B1', 'J1')
        if b2:
            circ.schedule(0.0, 'B2', 'J2')

        log = circ.run(t_max=200.0)
        o1 = [e for e in log if e['event'] == 'arrived' and e['dst'] == 'O1']
        o2 = [e for e in log if e['event'] == 'arrived' and e['dst'] == 'O2']

        label = f'{a}{b1}{b2}'
        results[label] = {
            'O1_fires': bool(o1),
            'O1_arrival_tu': o1[0]['t'] if o1 else None,
            'O2_fires': bool(o2),
            'O2_arrival_tu': o2[0]['t'] if o2 else None,
        }
    return results


def main():
    # Choose lengths so A arrives well before B at each junction and the
    # junction is still refractory when B arrives (refractory time 3.0 t.u.).
    # A at J1: 35/c ~ 5.4 t.u. -> refractory until 8.4 t.u.
    # B1 at J1: 45/c ~ 7.0 t.u. -> within refractory window -> blocked.
    # A at J2: 40/c ~ 6.2 t.u. -> refractory until 9.2 t.u.
    # B2 at J2: 50/c ~ 7.7 t.u. -> within refractory window -> blocked.
    geom = {
        'a_to_j1': 35.0, 'a_to_j2': 40.0,
        'b1_to_j1': 45.0, 'b2_to_j2': 50.0,
        'j1_to_o1': 40.0, 'j2_to_o2': 40.0,
    }
    circ = build_shared_control_gate(**geom)
    truth = run_all_patterns(circ)

    out = {
        'circuit': 'shared-control selective gate (one control, two data paths)',
        'geometry': geom,
        'calibration': {
            'speed': C_SPEED,
            'tau_refract': TAU_REFRACT,
            'inhibition_window': W_INHIBIT,
        },
        'truth_table': truth,
    }

    path = os.path.join(FIG, 'rd_multi_gate_graph.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    print('Shared-control gate truth table (A B1 B2):')
    for label, res in truth.items():
        print(f"  {label}: O1={res['O1_fires']}, O2={res['O2_fires']}")
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
