"""
rd_router_graph.py -- 2-to-1 priority router / pulse arbiter.

A simple network built from the measured dark-spot operators:

    A -----
           \
            J ---- OUT
           /
    B -----

Rules:
- A alone or B alone: pulse passes to OUT.
- A and B arrive close together: the first one to reach J wins;
  the later pulse is blocked by junction refractoriness.
- This is a primitive ``router'' in the sense that two input streams
  compete for the single output throat.

Edge lengths are chosen so A and B reach the junction at comparable times
when fired together, letting inhibition decide the winner.

Outputs:
    Analysis/figures/rd_router_graph.json
"""

import json
import os

from rd_graph_sim import Circuit, Pulse, C_SPEED, TAU_REFRACT, W_INHIBIT, PULSE_AMP

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


def build_priority_router(a_to_j: float = 60.0,
                          b_to_j: float = 70.0,
                          j_to_out: float = 80.0) -> Circuit:
    """Two inputs merge into one output through junction J."""
    circ = Circuit(tau_refract=TAU_REFRACT, w_inhibit=W_INHIBIT)
    for n in ('A', 'B', 'J', 'OUT'):
        circ.add_node(n)

    circ.add_edge('A->J', 'A', 'J', a_to_j)
    circ.add_edge('B->J', 'B', 'J', b_to_j)
    circ.add_edge('J->OUT', 'J', 'OUT', j_to_out)

    # Both inputs route to the single output.
    circ.nodes['J'].routing = {
        'A->J': ['J->OUT'],
        'B->J': ['J->OUT'],
        'J->OUT': [],
    }
    return circ


def run_patterns(circ):
    patterns = {
        '00': (False, False),
        '01': (False, True),
        '10': (True, False),
        '11': (True, True),
    }
    results = {}
    for label, (fire_a, fire_b) in patterns.items():
        circ.reset()
        if fire_a:
            circ.schedule(0.0, 'A', 'J')
        if fire_b:
            circ.schedule(0.0, 'B', 'J')
        log = circ.run(t_max=200.0)
        out = [e for e in log if e['event'] == 'arrived' and e['dst'] == 'OUT']
        results[label] = {
            'out_count': len(out),
            'out_arrival': out[0]['t'] if out else None,
            'fired': bool(out),
        }
    return results


def main():
    circ = build_priority_router()
    truth = run_patterns(circ)

    out = {
        'circuit': '2-to-1 priority router / pulse arbiter',
        'geometry': {
            'a_to_j': 60.0,
            'b_to_j': 65.0,
            'j_to_out': 80.0,
        },
        'calibration': {
            'speed': C_SPEED,
            'tau_refract': TAU_REFRACT,
            'inhibition_window': W_INHIBIT,
        },
        'truth_table': truth,
    }

    path = os.path.join(FIG, 'rd_router_graph.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    print('2-to-1 priority router truth table:')
    for pat, res in truth.items():
        print(f"  {pat}: OUT count = {res['out_count']}, "
              f"arrival = {res['out_arrival']:.3f} t.u."
              if res['out_arrival'] else f"  {pat}: OUT count = {res['out_count']}")
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
