"""Schematic of the four transfer-test geometries (final2 scripts).

Drawn to scale from the constants in:
  rd_transfer_channel_final2.py   (256x256, W swept 1..60, port x[0,18],
                                   probe strips x=64/128/217)
  rd_transfer_frequency_final2.py (300x48, W=16, port x[0,18], probe x=240)
  rd_transfer_logic_final2.py     (T-junction, barkley layout 256x256 W=20,
                                   TJ=96, A port x[25,43), B port y[163,181),
                                   probe strip x=240)
  rd_transfer_aniso_final2.py     (256x256 free medium, seed disk r=25,
                                   uniform tensor, c(theta) rays)

Output: Analysis/figures/rd_transfer_geometries.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
from matplotlib.lines import Line2D
import numpy as np

WALL = '#9aa5b1'
MED = '#eef3f8'
PORT = 'k'
PROBE = 'tab:red'

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# --- 1. Channel ----------------------------------------------------------
ax = axes[0, 0]
NX = NY = 256
CY = NY // 2
W = 20  # example width (swept 1..60)
ax.add_patch(Rectangle((0, 0), NX, NY, fc=WALL, ec='none'))
ax.add_patch(Rectangle((0, CY - W // 2), NX, W, fc=MED, ec='none'))
ax.add_patch(Rectangle((0, CY - W // 2), 18, W, fc=PORT, ec='none'))
for x in (64, 128, 217):
    ax.plot([x, x], [CY - W // 2, CY + W // 2], color=PROBE, lw=2.5)
ax.annotate('port slab x[0,18]', (30, CY), color='k', ha='left', va='center',
            fontsize=8, fontweight='bold')
for x, lab in ((64, 'near'), (128, 'mid'), (217, 'far')):
    ax.annotate(lab, (x, CY + W // 2 + 8), color=PROBE, ha='center', fontsize=9)
ax.annotate('channel width W swept\n1,2,...,60 (W=20 shown)', (128, 30),
            ha='center', fontsize=9, style='italic')
ax.set_title('Test 1: channel wire (256x256)')
ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_aspect('equal')

# --- 2. Frequency --------------------------------------------------------
ax = axes[0, 1]
NX, NY, W = 300, 48, 16
CY = NY // 2
ax.add_patch(Rectangle((0, 0), NX, NY, fc=WALL, ec='none'))
ax.add_patch(Rectangle((0, CY - W // 2), NX, W, fc=MED, ec='none'))
ax.add_patch(Rectangle((0, CY - W // 2), 18, W, fc=PORT, ec='none'))
ax.plot([240, 240], [CY - W // 2, CY + W // 2], color=PROBE, lw=2.5)
ax.annotate('port: pulse train x[0,18]', (24, CY + 14), color='k', ha='left',
            fontsize=8, fontweight='bold')
ax.annotate('far probe strip x=240', (240, CY + 16), color=PROBE, ha='center',
            fontsize=9)
ax.annotate('W=16', (150, CY), ha='center', va='center', fontsize=9,
            style='italic')
ax.set_title('Test 2: frequency / refractory (300x48)')
ax.set_xlim(0, NX); ax.set_ylim(-14, NY + 14); ax.set_aspect('equal')

# --- 3. T-junction logic (barkley layout) ---------------------------------
ax = axes[1, 0]
NX = NY = 256
W, TJ = 20, 96
W2 = W // 2
CY = NY // 2
ax.add_patch(Rectangle((0, 0), NX, NY, fc=WALL, ec='none'))
ax.add_patch(Rectangle((0, CY - W2), NX, W, fc=MED, ec='none'))       # A->out
ax.add_patch(Rectangle((TJ - W2, CY), W, 200 - CY, fc=MED, ec='none'))  # B in
ax.add_patch(Rectangle((25, CY - W2), 18, W, fc=PORT, ec='none'))      # A port
ax.add_patch(Rectangle((TJ - W2, 163), W, 18, fc=PORT, ec='none'))     # B port
ax.plot([240, 240], [CY - W2, CY + W2], color=PROBE, lw=2.5)
ax.annotate('A port\nx[25,43)', (34, CY - 22), ha='center', fontsize=8,
            fontweight='bold')
ax.annotate('B port\ny[163,181]', (TJ + 34, 172), ha='center', fontsize=8,
            fontweight='bold')
ax.annotate('junction TJ=96', (TJ, CY + 14), ha='center', fontsize=8,
            style='italic')
ax.annotate('probe x=240', (240, CY - 24), color=PROBE, ha='center',
            fontsize=9)
ax.set_title('Test 3: T-junction logic (256x256, W=20)')
ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_aspect('equal')

# --- 4. Anisotropy ---------------------------------------------------------
ax = axes[1, 1]
NX = NY = 256
CX = CY = NX // 2
ax.add_patch(Rectangle((0, 0), NX, NY, fc=MED, ec='none'))
ax.add_patch(Circle((CX, CY), 25, fc=PORT, ec='none'))
ax.add_patch(Ellipse((CX, CY), 2 * 95, 2 * 95 / 2, angle=30, fill=False,
             ec='tab:blue', ls='--', lw=1.8))
for a in (0, 30, 60, 90):
    th = np.deg2rad(a)
    ax.arrow(CX, CY, 105 * np.cos(th), 105 * np.sin(th), head_width=4,
             color=PROBE, length_includes_head=True, lw=1.2)
ax.annotate('seed disk r=25\n(direct field seed)', (CX, CY - 42), ha='center',
            fontsize=8, fontweight='bold')
ax.annotate('elliptical front\n(r=4 shown, axes 30 deg)', (CX + 70, CY + 62),
            color='tab:blue', fontsize=8)
ax.annotate('c(theta) rays', (CX - 100, CY + 88), color=PROBE, fontsize=9)
ax.set_title('Test 4: anisotropy (256x256, uniform tensor field)')
ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_aspect('equal')

for ax in axes.flat:
    ax.set_xlabel('x (cells)', fontsize=8)
    ax.set_ylabel('y (cells)', fontsize=8)
    ax.tick_params(labelsize=7)

handles = [Rectangle((0, 0), 1, 1, fc=WALL, label='wall (no-flux)'),
           Rectangle((0, 0), 1, 1, fc=MED, ec='gray', label='free medium'),
           Rectangle((0, 0), 1, 1, fc=PORT, label='port / seed'),
           Line2D([], [], color=PROBE, lw=2, label='probe')]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
           frameon=False)
fig.suptitle('Transfer-test geometries (final2 protocols, u-clamp port era '
             '-- to be re-run with dark-spot phi sources)', fontsize=12)
plt.tight_layout(rect=(0, 0.04, 1, 0.97))
plt.savefig('figures/rd_transfer_geometries.png', dpi=150)
print('saved figures/rd_transfer_geometries.png')
