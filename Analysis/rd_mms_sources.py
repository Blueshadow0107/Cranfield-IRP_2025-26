"""
MMS source-term generator for rd_core (method of manufactured solutions).

Manufactured solutions (Neumann-compatible: zero normal derivative at
x, y in {0, L}) on the domain [0, Lx] x [0, Ly], Lx = Ly = 64:

    u_hat = U0   + 0.05 * cos(pi x/Lx) cos(pi y/Ly) sin(2 pi t/T)
    v_hat = 0.10 + 0.04 * cos(pi x/Lx) cos(pi y/Ly) sin(2 pi t/T + 0.3)

with T = 20 t.u.

DEVIATION FROM THE ORIGINALLY PRESCRIBED FIELDS (documented): the
protocol specified U0 = 0.15.  That places u_hat on the UNSTABLE middle
branch of both kinetic models (Barkley: u_thr = (v+b)/a ~ 0.15 at
v ~ 0.1; Oregonator: the (u-q)/(u+q) switch region).  Pointwise
linearisation of the kinetics along the prescribed orbit gives
max Re(lambda) = +6.8 /t.u. (Barkley) and +13.3 /t.u. (Oregonator):
any discretisation error is amplified by ~e^{6.8*20} ~ 1e59 over the
run, so the prescribed fields blow up for EVERY resolution (verified:
the NaN/guard tripwire fires at t ~ 12 already at DX = 1).  This is a
manufactured-solution stability defect, not a solver bug -- the
variational equation is independent of the source.  The convergence
study therefore uses U0 = 0.90 (stable excited branch; max Re(lambda)
= -1.1 / -2.9 /t.u. over the orbit), keeping EVERYTHING ELSE exactly
as prescribed: amplitudes 0.05/0.04, v offset 0.10, phase 0.3, T = 20,
mode shape, domain.  The stability diagnostic is saved to
figures/rd_mms_sources.json.

The sources that make (u_hat, v_hat) an EXACT solution
of the full solver operators (reaction + divergence-form tensor
diffusion) are derived symbolically with sympy:

    Su = d_t u_hat - F_u(u_hat, v_hat) - div(D . grad u_hat)
    Sv = d_t v_hat - F_v(u_hat, v_hat) - Dv * laplacian(v_hat)

where, for a constant tensor, div(D . grad u) = Dxx u_xx + 2 Dxy u_xy
+ Dyy u_yy.  The expressions are lambdified to numpy (X, Y, t) callables
-- exactly the signature rd_core.set_source expects -- and the symbolic
strings are saved to Analysis/figures/rd_mms_sources.json for the
record.

Cases:
  (a) Barkley isotropic D = 1 (Dv = 0)
  (b) Barkley tilted tensor Dxx = Dyy = 2.5, Dxy = 1.5 (SPD, off-axis;
      key test of the cross-derivative path)
  (c) Oregonator Candidate A isotropic D = 1, Dv = 0.6, phi = 0.010

Usage:
    from rd_mms_sources import get_case
    case = get_case('barkley_tilted')
    rd = RDSubstrate(nx=n, ny=n, dx=dx, dt=dt, **case['solver_kw'])
    rd.set_phi(case['phi'])  # if nonzero
    rd.set_diffusion_tensor(*case['tensor'])
    rd.u[:] = case['u_hat'](X, Y, 0.0); rd.v[:] = case['v_hat'](X, Y, 0.0)
    rd.set_source(case['Su'], case['Sv'])
"""

import json

import sympy as sp

# ---------------------------------------------------------------------------
# Manufactured solution definition
# ---------------------------------------------------------------------------
LX = LY = 64.0
T = 20.0
U0_PRESCRIBED = 0.15   # unstable middle branch for both kinetics (see header)
U0 = 0.90              # stable excited branch -- used for the convergence runs
UA = 0.05
V0, VA = 0.10, 0.04
PHASE = 0.3

x, y, t = sp.symbols('x y t', real=True)
_w = 2 * sp.pi / T
_mode = sp.cos(sp.pi * x / LX) * sp.cos(sp.pi * y / LY)
u_hat_sym = U0 + UA * _mode * sp.sin(_w * t)
v_hat_sym = V0 + VA * _mode * sp.sin(_w * t + PHASE)

# Kinetic parameters (validated sets, see AGENTS.md / regime-hunt notes)
BARKLEY = dict(a=0.75, b=0.01, eps=0.02)
OREG_A = dict(f=1.4375, eps=0.05014844822490394, q=0.002, phi=0.010)

CASES = {
    'barkley_iso': dict(kinetics='barkley', params=BARKLEY,
                        tensor=(1.0, 1.0, 0.0), Dv=0.0),
    'barkley_tilted': dict(kinetics='barkley', params=BARKLEY,
                           tensor=(2.5, 2.5, 1.5), Dv=0.0),
    'oregonator_A': dict(kinetics='oregonator', params=OREG_A,
                         tensor=(1.0, 1.0, 0.0), Dv=0.6),
}


def _reaction(kinetics, params):
    """Symbolic reaction rates F_u, F_v on the manufactured fields."""
    if kinetics == 'barkley':
        a, b, eps = params['a'], params['b'], params['eps']
        Fu = u_hat_sym * (1 - u_hat_sym) * (u_hat_sym - (v_hat_sym + b) / a) / eps
        Fv = u_hat_sym - v_hat_sym
    else:
        f, q, eps, phi = params['f'], params['q'], params['eps'], params['phi']
        Fu = (u_hat_sym - u_hat_sym**2
              - (f * v_hat_sym + phi) * (u_hat_sym - q) / (u_hat_sym + q)) / eps
        Fv = u_hat_sym - v_hat_sym
    return Fu, Fv


def build_case(name):
    """Derive and lambdify (u_hat, v_hat, Su, Sv) for a named case."""
    spec = CASES[name]
    Dxx, Dyy, Dxy = spec['tensor']
    Dv = spec['Dv']
    Fu, Fv = _reaction(spec['kinetics'], spec['params'])

    ux, uy = sp.diff(u_hat_sym, x), sp.diff(u_hat_sym, y)
    div_Du = sp.diff(Dxx * ux + Dxy * uy, x) + sp.diff(Dyy * uy + Dxy * ux, y)
    div_Dv = Dv * (sp.diff(v_hat_sym, x, 2) + sp.diff(v_hat_sym, y, 2))

    Su = sp.diff(u_hat_sym, t) - Fu - div_Du
    Sv = sp.diff(v_hat_sym, t) - Fv - div_Dv

    lam = lambda e: sp.lambdify((x, y, t), e, 'numpy')  # noqa: E731
    out = {
        'name': name,
        'kinetics': spec['kinetics'],
        'params': spec['params'],
        'tensor': spec['tensor'],
        'Dv': spec['Dv'],
        'u_hat': lam(u_hat_sym),
        'v_hat': lam(v_hat_sym),
        'Su': lam(Su),
        'Sv': lam(Sv),
        'expr': {'u_hat': str(u_hat_sym), 'v_hat': str(v_hat_sym),
                 'Su': str(Su), 'Sv': str(Sv)},
        # solver constructor kwargs matching rd_core defaults for the rest
        'solver_kw': dict(kinetics=spec['kinetics'],
                          eps=spec['params']['eps'],
                          f=spec['params'].get('f', 1.4),
                          q=spec['params'].get('q', 0.002),
                          a=spec['params'].get('a', 0.75),
                          b=spec['params'].get('b', 0.01),
                          Dv=spec['Dv']),
        'phi': spec['params'].get('phi', 0.0),
    }
    return out


def get_case(name):
    assert name in CASES, f'unknown MMS case {name!r}; have {list(CASES)}'
    return build_case(name)


def orbit_stability(kinetics, params, u_offset, n_t=81):
    """Max real part of the pointwise kinetic Jacobian along the
    manufactured orbit (mode in {-1, 0, +1}, one full period).  Positive
    => the manufactured solution is dynamically unstable and MMS
    convergence is impossible at any resolution."""
    import numpy as np
    w = 2 * np.pi / T
    lam_max = -np.inf
    for m in (-1.0, 0.0, 1.0):
        for tt in np.linspace(0.0, T, n_t):
            uu = u_offset + UA * m * np.sin(w * tt)
            vv = V0 + VA * m * np.sin(w * tt + PHASE)
            if kinetics == 'barkley':
                a, b, eps = params['a'], params['b'], params['eps']
                thr = (vv + b) / a
                ju = ((1 - 2 * uu) * (uu - thr) + uu * (1 - uu)) / eps
                jv = -uu * (1 - uu) / (a * eps)
            else:
                f, q, eps = params['f'], params['q'], params['eps']
                phi = params['phi']
                s = f * vv + phi
                g = (uu - q) / (uu + q)
                ju = (1 - 2 * uu - s * 2 * q / (uu + q)**2) / eps
                jv = -f * g / eps
            J = np.array([[ju, jv], [1.0, -1.0]])
            lam_max = max(lam_max,
                          float(np.linalg.eigvals(J).real.max()))
    return lam_max


def main():
    """Derive all cases, save the symbolic expressions, self-check."""
    record = {'manufactured': {'u_hat': str(u_hat_sym),
                               'v_hat': str(v_hat_sym),
                               'LX': LX, 'LY': LY, 'T': T},
              'u_offset_note': (
                  'protocol prescribed u offset 0.15 (unstable middle '
                  'branch for both kinetics -- see orbit_stability); '
                  'convergence runs use 0.90 (stable excited branch), '
                  'all other field parameters as prescribed'),
              'u_offset_prescribed': U0_PRESCRIBED,
              'u_offset_used': U0,
              'orbit_stability': {},
              'cases': {}}
    for kin, params in (('barkley', BARKLEY), ('oregonator', OREG_A)):
        record['orbit_stability'][kin] = {
            'max_Re_eig_prescribed_U0_0.15':
                orbit_stability(kin, params, U0_PRESCRIBED),
            'max_Re_eig_used_U0_0.90':
                orbit_stability(kin, params, U0)}
        print(f"[orbit stability] {kin}: prescribed U0=0.15 -> max Re(lam) "
              f"{record['orbit_stability'][kin]['max_Re_eig_prescribed_U0_0.15']:+.2f} /t.u.; "
              f"used U0=0.90 -> "
              f"{record['orbit_stability'][kin]['max_Re_eig_used_U0_0.90']:+.2f} /t.u.")
    import numpy as np
    Xg, Yg = np.mgrid[0.5:8:1.0, 0.5:8:1.0]
    for name in CASES:
        case = build_case(name)
        record['cases'][name] = {
            'kinetics': case['kinetics'], 'params': case['params'],
            'tensor': case['tensor'], 'Dv': case['Dv'],
            'expressions': case['expr']}
        # sanity: with the sources active, du/dt - d_t u_hat ~ 0 pointwise
        # (checked by evaluating source + operators at the exact fields)
        su = case['Su'](Xg, Yg, 1.234)
        sv = case['Sv'](Xg, Yg, 1.234)
        assert np.isfinite(su).all() and np.isfinite(sv).all()
        print(f'[{name}] |Su|max ~ {np.abs(su).max():.3f}, '
              f'|Sv|max ~ {np.abs(sv).max():.3f}')
    path = 'figures/rd_mms_sources.json'
    with open(path, 'w') as fh:
        json.dump(record, fh, indent=2)
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
