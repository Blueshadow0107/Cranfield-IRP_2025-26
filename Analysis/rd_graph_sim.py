"""
rd_graph_sim.py -- event-driven graph simulator for BZ pulse networks.

Builds a circuit from the measured dark-spot transfer functions and runs
input patterns without solving the full PDE.  Current target: collision-XOR
with a chamber at the merge.

Calibration (dark-spot Oregonator A, phi0=0.010):
    channel speed        c = 6.464 cells / t.u.
    refractory time      tau = 3.0 t.u.
    inhibition window    w_inh = 1.0 t.u.
    pulse amplitude      amp = 0.726

The dark-spot / projected-channel input method is the standard
photosensitive-BZ experimental protocol.  Collision-based XOR and other
logic gates have been demonstrated experimentally by Adamatzky & De Lacy
Costello (Phys Rev E 66, 046112, 2002) and De Lacy Costello & Adamatzky
(Chaos Solitons & Fractals 25, 535, 2005); projected light masks are the
usual way to define excitable channels in the Ru-catalysed reaction
(Stevens, arXiv:1204.5345, 2012).

Outputs:
    Analysis/figures/rd_graph_sim.json
"""

import json
import heapq
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Measured dark-spot transfer parameters (Oregonator A, phi0=0.010)
# ---------------------------------------------------------------------------
C_SPEED = 6.46411888154886      # cells / t.u., from rd_transfer_channel_darkspot
TAU_REFRACT = 3.0               # t.u., from rd_transfer_frequency_darkspot
W_INHIBIT = 1.0                 # t.u., from rd_transfer_logic_darkspot
PULSE_AMP = 0.7264304595123359  # free-medium far peak

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------
@dataclass
class Pulse:
    """A pulse travelling toward a node."""
    t_arrive: float
    src: str
    dst: str
    amp: float = PULSE_AMP

    def __lt__(self, other):
        return self.t_arrive < other.t_arrive


@dataclass
class Edge:
    """A throat / channel between two nodes."""
    name: str
    src: str
    dst: str
    length: float          # cells
    speed: float = C_SPEED
    refractory_until: float = 0.0

    def transit_time(self) -> float:
        return self.length / self.speed


@dataclass
class Node:
    """A pore / junction."""
    name: str
    refractory_until: float = 0.0
    # inhibition bookkeeping: pulses arriving here within W_INHIBIT annihilate
    recent_arrivals: List[Pulse] = field(default_factory=list)
    # optional routing table: incoming edge name -> list of outgoing edge names
    routing: Dict[str, List[str]] = field(default_factory=dict)


class Circuit:
    """Event-driven pulse network."""

    def __init__(self, tau_refract: float = TAU_REFRACT,
                 w_inhibit: float = W_INHIBIT):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.outgoing: Dict[str, List[Edge]] = {}
        self.tau_refract = tau_refract
        self.w_inhibit = w_inhibit
        self.events: List[Pulse] = []
        self.log: List[Dict] = []

    def add_node(self, name: str):
        self.nodes[name] = Node(name)
        self.outgoing[name] = []

    def add_edge(self, name: str, src: str, dst: str, length: float,
                 speed: float = C_SPEED):
        e = Edge(name, src, dst, length, speed)
        self.edges[name] = e
        self.outgoing[src].append(e)

    def schedule(self, t_start: float, src: str, dst: str,
                 amp: float = PULSE_AMP):
        """Inject a pulse at node `src` at time t_start, bound for dst."""
        e = self.edges.get(f"{src}->{dst}")
        if e is None:
            raise ValueError(f"no edge {src}->{dst}")
        heapq.heappush(self.events, Pulse(t_start + e.transit_time(),
                                          src, dst, amp))

    def reset(self):
        self.events = []
        self.log = []
        for n in self.nodes.values():
            n.refractory_until = 0.0
            n.recent_arrivals = []
        for e in self.edges.values():
            e.refractory_until = 0.0

    def _pending_collisions(self, dst: str, t: float) -> List[Pulse]:
        """Return events in the queue arriving at dst within W_INHIBIT of t."""
        return [
            q for q in self.events
            if q.dst == dst and abs(q.t_arrive - t) <= self.w_inhibit
        ]

    def _remove_events(self, to_remove: List[Pulse]):
        """Remove specific events from the priority queue."""
        removed = set(id(q) for q in to_remove)
        self.events = [q for q in self.events if id(q) not in removed]
        heapq.heapify(self.events)

    def run(self, t_max: float = 100.0) -> List[Dict]:
        """Process all events up to t_max."""
        while self.events:
            p = heapq.heappop(self.events)
            if p.t_arrive > t_max:
                continue

            node = self.nodes[p.dst]

            # 1) Drop if destination node is refractory.
            if p.t_arrive < node.refractory_until:
                self.log.append({
                    't': p.t_arrive,
                    'dst': p.dst,
                    'src': p.src,
                    'event': 'dropped_refractory_node',
                    'refractory_until': node.refractory_until,
                })
                continue

            # 2) Check for inhibition/collision within the inhibition window.
            #    Look at pending events as well as recent arrivals at this node.
            pending = self._pending_collisions(p.dst, p.t_arrive)
            recent = [
                q for q in node.recent_arrivals
                if abs(q.t_arrive - p.t_arrive) <= self.w_inhibit
            ]
            colliding = pending + recent
            if colliding:
                # Annihilation: mark node refractory and clear collision partner(s).
                node.refractory_until = p.t_arrive + self.tau_refract
                self._remove_events(pending)
                for q in recent:
                    if q in node.recent_arrivals:
                        node.recent_arrivals.remove(q)
                self.log.append({
                    't': p.t_arrive,
                    'dst': p.dst,
                    'src': p.src,
                    'event': 'annihilated',
                    'collided_with': [q.src for q in colliding],
                })
                continue

            # 3) Pulse survives: mark node refractory and forward to outputs.
            node.refractory_until = p.t_arrive + self.tau_refract
            self.log.append({
                't': p.t_arrive,
                'dst': p.dst,
                'src': p.src,
                'event': 'arrived',
                'amp': p.amp,
            })

            # Determine incoming edge name.
            incoming = f"{p.src}->{p.dst}"
            # Use routing table if present, otherwise forward to all outputs.
            allowed = node.routing.get(incoming)
            outputs = (
                [e for e in self.outgoing[p.dst] if e.name in allowed]
                if allowed is not None else self.outgoing[p.dst]
            )

            for e in outputs:
                if p.t_arrive < e.refractory_until:
                    self.log.append({
                        't': p.t_arrive,
                        'edge': e.name,
                        'event': 'dropped_refractory_edge',
                        'refractory_until': e.refractory_until,
                    })
                    continue
                e.refractory_until = p.t_arrive + self.tau_refract
                heapq.heappush(self.events, Pulse(
                    p.t_arrive + e.transit_time(),
                    p.dst, e.dst, p.amp))

            # Keep this arrival as a potential future collision partner.
            node.recent_arrivals.append(p)
            # Purge old arrivals outside the inhibition window.
            node.recent_arrivals = [
                q for q in node.recent_arrivals
                if p.t_arrive - q.t_arrive <= self.w_inhibit
            ]

        return self.log


# ---------------------------------------------------------------------------
# Collision-XOR with chamber
# ---------------------------------------------------------------------------
def build_xor_with_chamber(arm_length: float = 60.0,
                           chamber_radius: float = 10.0,
                           stem_length: float = 50.0) -> Circuit:
    """
    Two input arms merge into a circular chamber, then a single output stem.

    Nodes:
        A, B : input ports
        C    : chamber (collision site)
        OUT  : output probe location
    """
    circ = Circuit()
    for n in ('A', 'B', 'C', 'OUT'):
        circ.add_node(n)

    # Arm lengths include distance from input to chamber edge.
    circ.add_edge('A->C', 'A', 'C', arm_length)
    circ.add_edge('B->C', 'B', 'C', arm_length)
    # Chamber crossing: approximate as diameter (2*radius) at measured speed.
    circ.add_edge('C->OUT', 'C', 'OUT', 2.0 * chamber_radius + stem_length)

    return circ


def xor_truth_table(circ: Circuit, arm_length: float) -> Dict:
    """Run all four input patterns and report output arrival times."""
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
            circ.schedule(0.0, 'A', 'C')
        if fire_b:
            # Fire B from the same nominal time; real offsets can be swept.
            circ.schedule(0.0, 'B', 'C')

        log = circ.run(t_max=200.0)
        out_arrivals = [e for e in log
                        if e['event'] == 'arrived' and e['dst'] == 'OUT']

        results[label] = {
            'out_arrival': out_arrivals[0]['t'] if out_arrivals else None,
            'out_count': len(out_arrivals),
            'fired': bool(out_arrivals),
        }

    return results



# ---------------------------------------------------------------------------
# A AND (NOT B) inhibition gate
# ---------------------------------------------------------------------------
def build_andnot_gate(a_to_j: float = 62.0,
                      b_to_j: float = 49.0,
                      j_to_out: float = 144.0) -> Circuit:
    """
    T-junction inhibition gate.  A travels horizontally through the junction
    to the output; B enters from the top and has no output path.  B's arrival
    makes the junction refractory, blocking A.

    Edge lengths mirror the dark-spot T-junction transfer test
    (rd_transfer_logic_darkspot.py):
        A port centre x=34, junction x=96  -> a_to_j = 62 cells
        B port centre y=149, junction y=100 -> b_to_j = 49 cells
        junction x=96 to probe x=240        -> j_to_out = 144 cells
    """
    circ = Circuit()
    for n in ('A', 'B', 'J', 'OUT'):
        circ.add_node(n)

    circ.add_edge('A->J', 'A', 'J', a_to_j)
    circ.add_edge('B->J', 'B', 'J', b_to_j)
    circ.add_edge('J->OUT', 'J', 'OUT', j_to_out)
    # A passes through; B enters and blocks but does not exit through OUT.
    circ.nodes['J'].routing = {
        'A->J': ['J->OUT'],
        'B->J': [],
        'J->OUT': [],
    }
    return circ


def andnot_truth_table(circ: Circuit) -> Dict:
    """Run all four input patterns for A AND (NOT B)."""
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
        out_arrivals = [e for e in log
                        if e['event'] == 'arrived' and e['dst'] == 'OUT']

        results[label] = {
            'out_arrival': out_arrivals[0]['t'] if out_arrivals else None,
            'out_count': len(out_arrivals),
            'fired': bool(out_arrivals),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    arm_length = 60.0
    chamber_radius = 10.0
    stem_length = 50.0

    # ---- collision-XOR -----------------------------------------------------
    circ_xor = build_xor_with_chamber(arm_length, chamber_radius, stem_length)
    truth_xor = xor_truth_table(circ_xor, arm_length)

    sweep = {}
    for r in [5.0, 10.0, 15.0, 20.0]:
        c = build_xor_with_chamber(arm_length, r, stem_length)
        sweep[r] = xor_truth_table(c, arm_length)

    # ---- A AND (NOT B) -----------------------------------------------------
    circ_andnot = build_andnot_gate()
    truth_andnot = andnot_truth_table(circ_andnot)

    out = {
        'calibration': {
            'speed': C_SPEED,
            'tau_refract': TAU_REFRACT,
            'inhibition_window': W_INHIBIT,
            'pulse_amp': PULSE_AMP,
            'source': 'dark-spot transfer tests (Oregonator A, phi0=0.010)',
            'input_method_citation': (
                'Adamatzky A., De Lacy Costello B.P.J. "Experimental logical '
                'gates in a reaction-diffusion medium: The XOR gate and beyond", '
                'Phys Rev E 66, 046112 (2002); De Lacy Costello B., Adamatzky A. '
                '"Experimental implementation of collision-based gates in '
                'Belousov-Zhabotinsky medium", Chaos Solitons & Fractals 25, '
                '535-544 (2005); Stevens W.M. arXiv:1204.5345 (2012) on '
                'projected-channel light-sensitive BZ circuits.'
            ),
        },
        'collision_xor': {
            'geometry': {
                'arm_length': arm_length,
                'chamber_radius': chamber_radius,
                'stem_length': stem_length,
            },
            'truth_table': truth_xor,
            'chamber_radius_sweep': sweep,
            'note': ('PDE validation shows extended channel waves leak in the '
                     '11 case; the graph idealisation does not capture this.'),
        },
        'and_not_b': {
            'geometry': {
                'a_to_j': 62.0,
                'b_to_j': 49.0,
                'j_to_out': 144.0,
            },
            'truth_table': truth_andnot,
            'pde_reference': 'rd_transfer_logic_darkspot.json',
        },
    }

    path = os.path.join(FIG, 'rd_graph_sim.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    print('Collision-XOR truth table:')
    for pat, res in truth_xor.items():
        print(f"  {pat}: OUT fires = {res['fired']}, "
              f"arrival = {res['out_arrival']:.3f} t.u."
              if res['out_arrival'] else f"  {pat}: OUT fires = {res['fired']}")

    print('\nA AND (NOT B) truth table:')
    for pat, res in truth_andnot.items():
        print(f"  {pat}: OUT fires = {res['fired']}, "
              f"arrival = {res['out_arrival']:.3f} t.u."
              if res['out_arrival'] else f"  {pat}: OUT fires = {res['fired']}")

    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
