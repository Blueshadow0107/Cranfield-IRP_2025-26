"""Publication-quality Matplotlib style for the BZ reaction--diffusion thesis.

Usage:
    import rd_house_style as rhs
    rhs.setup()          # apply the style
    fig, ax = plt.subplots()
    ...
    rhs.save_both(fig, 'Analysis/figures/pub/my_figure')
"""
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Color palette: Okabe--Ito (colorblind-friendly)
# -----------------------------------------------------------------------------
COLORS = {
    'black':        '#000000',
    'orange':       '#E69F00',
    'sky_blue':     '#56B4E9',
    'bluish_green': '#009E73',
    'yellow':       '#F0E442',
    'blue':         '#0072B2',
    'vermillion':   '#D55E00',
    'purple':       '#CC79A7',
    'grey':         '#999999',
}

PALETTE = [
    COLORS['blue'],
    COLORS['vermillion'],
    COLORS['bluish_green'],
    COLORS['orange'],
    COLORS['purple'],
    COLORS['sky_blue'],
    COLORS['yellow'],
    COLORS['black'],
]

# -----------------------------------------------------------------------------
# Default dimensions (inches)
# -----------------------------------------------------------------------------
SINGLE_PANEL = (5.0, 4.0)
DOUBLE_PANEL = (9.0, 4.0)
TRIPLE_PANEL = (12.0, 4.0)
SQUARE_2X2 = (8.5, 7.5)
TALL_3x1 = (6.0, 8.0)

# -----------------------------------------------------------------------------
# rcParams
# -----------------------------------------------------------------------------
_RCPARAMS = {
    # fonts
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,

    # lines / markers
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'lines.markeredgewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,

    # colors
    'axes.prop_cycle': plt.cycler(color=PALETTE),
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',

    # legend
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#cccccc',
    'legend.fancybox': False,

    # figure
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.compression': 9,
}


def setup():
    """Apply the thesis house style to the current pyplot session."""
    plt.rcParams.update(_RCPARAMS)


def reset():
    """Restore Matplotlib defaults (useful when mixing styles)."""
    plt.rcdefaults()


def save_both(fig, path_without_ext, dpi=200, tight=True):
    """Save a figure as high-resolution PNG and vector PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path_without_ext : str or Path
        Output path without extension.  Parent directories are created.
    dpi : int, optional
        PNG resolution (PDF is vector, so dpi is ignored for it).
    tight : bool, optional
        Use tight bounding box.
    """
    path = Path(path_without_ext)
    path.parent.mkdir(parents=True, exist_ok=True)

    bbox = 'tight' if tight else None
    png_path = path.with_suffix('.png')
    pdf_path = path.with_suffix('.pdf')

    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox, pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches=bbox, pad_inches=0.02)
    return str(png_path), str(pdf_path)


def get_color(i):
    """Return the i-th color from the cyclic palette."""
    return PALETTE[i % len(PALETTE)]


if __name__ == '__main__':
    setup()
    fig, ax = plt.subplots(figsize=SINGLE_PANEL)
    for i, c in enumerate(PALETTE):
        ax.plot([0, 1], [i, i], color=c, marker='o', label=f'swatch {i}')
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_title('House style palette check')
    ax.legend(ncols=2, loc='upper right')
    save_both(fig, os.path.join(os.path.dirname(__file__),
                                'figures', 'pub', '_house_style_check'))
    print('Saved palette check to Analysis/figures/pub/_house_style_check.{png,pdf}')
