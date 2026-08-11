"""
Shared Matplotlib style for the audit-report figures.

Usage:
    import audit_style
    audit_style.setup()
    # ... make plots ...
    audit_style.save(fig, 'my_figure')
"""
import os
import matplotlib
import matplotlib.pyplot as plt

FIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Palette: colour-blind-friendly, print-friendly
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'grey': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf',
    'black': '#000000',
}

COLOR_CYCLE = [
    COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red'],
    COLORS['purple'], COLORS['brown'], COLORS['pink'], COLORS['grey']
]


def setup(font_size=10, line_width=1.2):
    """Apply the audit-report style to matplotlib."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'DejaVu Serif', 'serif'],
        'text.usetex': False,
        'font.size': font_size,
        'axes.titlesize': font_size + 1,
        'axes.labelsize': font_size,
        'legend.fontsize': font_size - 1,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'axes.prop_cycle': plt.cycler(color=COLOR_CYCLE),
        'axes.linewidth': line_width,
        'lines.linewidth': line_width,
        'lines.markersize': 5,
        'xtick.major.width': line_width,
        'ytick.major.width': line_width,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'image.cmap': 'viridis',
    })


def save(fig, name, pdf=False):
    """Save figure to the audit-report figures directory."""
    path_png = os.path.join(FIG_DIR, f'{name}.png')
    fig.savefig(path_png)
    if pdf:
        path_pdf = os.path.join(FIG_DIR, f'{name}.pdf')
        fig.savefig(path_pdf)
    print(f'[saved] {path_png}')
