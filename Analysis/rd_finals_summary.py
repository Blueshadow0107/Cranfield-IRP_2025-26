"""
Consolidated finals summary -- definitive post-clamp-fix numbers for both
kinetics, with before/after comparison against the superseded runs.

Reads the _final2 JSONs (definitive):
    figures/rd_transfer_channel_final2.json      (Test 1, both kinetics)
    figures/rd_transfer_aniso_final2.json        (Test 4a, both kinetics)
    figures/rd_transfer_frequency_final2_*.json  (Test 2, both kinetics)
    figures/rd_transfer_logic_final2_*.json      (Test 3, both kinetics)

and the superseded reference JSONs (read-only):
    figures/rd_transfer_channel_final.json       (de-hacked, (0,0)-clamp)
    figures/rd_transfer_aniso_final.json         (de-hacked, barkley only)
    figures/rd_transfer_frequency.json           (OLD CLIPPED solver)
    figures/rd_transfer_frequency_oregonator.json (de-hacked, (0,0)-clamp)
    figures/rd_transfer_logic.json               (OLD CLIPPED solver)
    figures/rd_transfer_logic_oregonator.json    (de-hacked, (0,0)-clamp)

Movement flags: a number is flagged MOVED if its relative change exceeds
the convergence error bars -- spatial ~14% (speeds, sqrt(r) deviations)
or temporal ~1.5% (refractory period, following rates, inhibition
window).  For Barkley Tests 2/3 the reference numbers came from the OLD
CLIPPED solver (and a different Test-2 geometry), so movement there is
expected and physical, not a regression.  For the Oregonator the
reference is the post-de-hack (0,0)-clamp run, so any movement isolates
the clamp artifact.

Outputs: Analysis/figures/rd_finals_summary.{json,png}
"""

import json

import numpy as np
import matplotlib.pyplot as plt

SPATIAL_TOL = 0.14
TEMPORAL_TOL = 0.015

D = 'figures'


def load(p):
    with open(f'{D}/{p}') as fh:
        return json.load(fh)


def moved(new, old, tol):
    if old is None or new is None:
        return None
    if old == 0:
        return bool(new != 0)
    return abs(new - old) / abs(old) > tol


def entry(new, old, tol, note=''):
    rel = None if (old in (None, 0) or new is None) else (new - old) / old
    return {'new': new, 'old': old, 'rel_change': rel,
            'moved_beyond_tol': moved(new, old, tol), 'tol': tol,
            'note': note}


def main():
    ch2 = load('rd_transfer_channel_final2.json')
    an2 = load('rd_transfer_aniso_final2.json')
    fr2 = {k: load(f'rd_transfer_frequency_final2_{k}.json')
           for k in ('barkley', 'oregonator')}
    lo2 = {k: load(f'rd_transfer_logic_final2_{k}.json')
           for k in ('barkley', 'oregonator')}

    ch1 = load('rd_transfer_channel_final.json')
    an1 = load('rd_transfer_aniso_final.json')
    fr1 = {'barkley': load('rd_transfer_frequency.json'),
           'oregonator': load('rd_transfer_frequency_oregonator.json')}
    lo1 = {'barkley': load('rd_transfer_logic.json'),
           'oregonator': load('rd_transfer_logic_oregonator.json')}

    old_prov = {
        'barkley': {'freq': 'OLD CLIPPED solver, 256x256 W=32 geometry',
                    'logic': 'OLD CLIPPED solver'},
        'oregonator': {'freq': 'de-hacked, (0,0)-clamp artifact',
                       'logic': 'de-hacked, (0,0)-clamp artifact'},
    }

    summary = {'tolerances': {'spatial': SPATIAL_TOL,
                              'temporal': TEMPORAL_TOL},
               'kinetics': {}}
    for kin in ('barkley', 'oregonator'):
        f2, f1 = ch2[kin]['free_medium'], ch1[kin]['free_medium']
        eq2 = ch2[kin]['channel_speed_equality']
        eq1_ratio = None
        # old channel_final has no equality block; use W=60 sweep entry
        w60 = [r for r in ch1[kin]['sweep'] if r['W'] == 60]
        if w60 and w60[0]['speed_cells_per_tu']:
            eq1_ratio = (w60[0]['speed_cells_per_tu']
                         / f1['speed_cells_per_tu'])
        a2 = an2[kin]['cases']
        # old aniso reference exists for barkley only (aniso_final)
        a1 = an1['cases'] if kin == 'barkley' else {}
        s = {
            'label': ch2[kin]['kinetics_label'],
            'test1_channel': {
                'free_medium_speed_cells_per_tu': entry(
                    f2['speed_cells_per_tu'], f1['speed_cells_per_tu'],
                    SPATIAL_TOL,
                    'old = de-hacked channel_final with (0,0)-clamp'),
                'channel_speed_equality_ratio_W60': entry(
                    eq2['ratio'], eq1_ratio, SPATIAL_TOL,
                    'channel(W=60)/free speed; 1.0 = geometry-free'),
                'min_width_transmitted': {
                    'new': ch2[kin]['block_threshold']
                    ['min_width_transmitted'],
                    'old': ch1[kin]['block_threshold']
                    ['min_width_transmitted']},
                'control_quiet': ch2[kin]['control']['quiet'],
            },
            'test2_frequency': {
                'reference_provenance': old_prov[kin]['freq'],
                'refractory_period_tu': entry(
                    fr2[kin]['refractory_period_time'],
                    fr1[kin]['refractory_period_time'], TEMPORAL_TOL),
                'max_1to1_following_rate_per_tu': entry(
                    fr2[kin]['max_following_rate'],
                    fr1[kin]['max_following_rate'], TEMPORAL_TOL),
                'control_quiet': fr2[kin]['control_quiet'],
                'single_pulse_crossings': fr2[kin]['single_pulse_crossings'],
            },
            'test3_logic': {
                'reference_provenance': old_prov[kin]['logic'],
                'truth_table_window_peaks': lo2[kin]['truth_table'],
                'separation_ratio': entry(
                    lo2[kin]['separation_ratio'],
                    lo1[kin]['separation_ratio'], SPATIAL_TOL,
                    'false peak sits at the rest-state floor; ratio is '
                    'scale-free, treat movement qualitatively'),
                'inhibition_window_tu': entry(
                    lo2[kin]['inhibition_window_time'],
                    lo1[kin].get('inhibition_window_time',
                                 lo1[kin]['inhibition_window_steps'] * 0.05),
                    TEMPORAL_TOL),
                'control_quiet': lo2[kin]['control_quiet'],
                'single_pulse_crossings': lo2[kin]['single_pulse_crossings'],
            },
            'test4a_sqrt_r_deviation_pct': {
                r: entry(a2[r]['deviation_pct'],
                         a1[r]['deviation_pct'] if r in a1 else None,
                         SPATIAL_TOL,
                         'deviation of c_par/c_perp from sqrt(r); old '
                         '(barkley) from aniso_final' if r in a1 else
                         'no old oregonator reference')
                for r in ('1', '2', '4')},
        }
        summary['kinetics'][kin] = s

    # ---- console table -----------------------------------------------------
    for kin in ('barkley', 'oregonator'):
        s = summary['kinetics'][kin]
        print(f'\n=== {kin}: {s["label"]} ===')
        rows = [
            ('free speed (cells/t.u.)',
             s['test1_channel']['free_medium_speed_cells_per_tu']),
            ('channel/free speed (W=60)',
             s['test1_channel']['channel_speed_equality_ratio_W60']),
            ('refractory (t.u.)', s['test2_frequency']['refractory_period_tu']),
            ('max 1:1 rate (/t.u.)',
             s['test2_frequency']['max_1to1_following_rate_per_tu']),
            ('separation ratio', s['test3_logic']['separation_ratio']),
            ('inhibition window (t.u.)',
             s['test3_logic']['inhibition_window_tu']),
        ] + [(f'sqrt({r}) deviation (%)',
              s['test4a_sqrt_r_deviation_pct'][r]) for r in ('1', '2', '4')]
        print(f'  {"quantity":<30}{"final2":>12}{"old":>12}{"dRel":>9}  flag')
        for name, e in rows:
            old = e['old']
            rel = e['rel_change']
            flag = ('MOVED' if e['moved_beyond_tol'] else 'ok') \
                if e['moved_beyond_tol'] is not None else '-'
            print(f'  {name:<30}{e["new"]:>12.4g}'
                  f'{old if old is not None else float("nan"):>12.4g}'
                  f'{f"{rel:+.1%}" if rel is not None else "-":>9}  {flag}')
        tt = s['test3_logic']['truth_table_window_peaks']
        print(f'  truth table (window peaks): 00={tt["00"]:.3g} '
              f'10={tt["10"]:.3f} 01={tt["01"]:.3g} 11={tt["11"]:.3g}')

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(2)
    ax = axes[0]
    width = 0.35
    for j, (key, lab) in enumerate((('new', 'final2 (clamp-rest fix)'),
                                    ('old', 'old reference'))):
        vals = [summary['kinetics'][k]['test2_frequency']
                ['refractory_period_tu'][key] for k in
                ('barkley', 'oregonator')]
        ax.bar(x + j * width - width / 2, vals, width, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(['Barkley', 'Oregonator A'])
    ax.set_ylabel('refractory period (t.u.)')
    ax.set_title('Test 2: refractory period'); ax.legend(fontsize=8)

    ax = axes[1]
    tt_new = [summary['kinetics'][k]['test3_logic']
              ['truth_table_window_peaks'] for k in ('barkley', 'oregonator')]
    labels = ['00', '10', '01', '11']
    for i, kin in enumerate(('Barkley', 'Oregonator A')):
        ax.bar(np.arange(4) + i * width - width / 2,
               [tt_new[i][k] for k in labels], width, label=kin)
    ax.set_yscale('log'); ax.set_ylim(1e-4, 3)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels)
    ax.set_xlabel('inputs (A,B)'); ax.set_ylabel('window peak u')
    ax.set_title('Test 3: A AND (NOT B) truth table (final2)')
    ax.legend(fontsize=8)

    ax = axes[2]
    for kin in ('barkley', 'oregonator'):
        devs = [summary['kinetics'][kin]['test4a_sqrt_r_deviation_pct'][r]
                ['new'] for r in ('1', '2', '4')]
        ax.plot([1, 2, 4], devs, 'o-', label=f'{kin} final2')
    devs_old = [an1['cases'][r]['deviation_pct'] for r in ('1', '2', '4')]
    ax.plot([1, 2, 4], devs_old, 's--', color='gray',
            label='barkley old (aniso_final)')
    ax.axhline(14, color='r', ls=':', label='spatial tol 14%')
    ax.set_xlabel('r'); ax.set_ylabel('|c_par/c_perp - sqrt(r)| / sqrt(r) (%)')
    ax.set_title('Test 4a: sqrt(r) deviations'); ax.legend(fontsize=8)

    fig.suptitle('RD finals summary -- clamp-rest-fix definitive numbers '
                 'vs superseded references', fontsize=13)
    fig.tight_layout()
    figpath = f'{D}/rd_finals_summary.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'\nSaved {figpath}')

    jsonpath = f'{D}/rd_finals_summary.json'
    with open(jsonpath, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
