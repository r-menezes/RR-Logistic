"""
@author: Rafael Menezes (github: r-menezes)
@date: 2025
@license: MIT License
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mm_to_inches = 0.03937008
sns.set_theme(style="ticks")
sns.set_palette("colorblind")
sns.set_context("paper")

plt.ioff()

# >> RC PARAMS <<

_dica = {
    # General figure properties
    'figure.figsize': (80*mm_to_inches, 64*mm_to_inches),    # width and height in inches (80-180 mm width)
    'figure.dpi': 600,               # high resolution for line art
    'savefig.dpi': 600,
    'savefig.format': 'pdf',         # vector graphics format
    'savefig.bbox': 'standard',      # avoid tight layout
    
    # Font properties
    'font.size': 10,                 # appropriate font size
    'font.family': 'sans-serif',     # default font family
    'font.sans-serif': ['CMU Sans Serif', 'DejaVu Sans', 'Arial', 'sans-serif'],
    'font.serif': ['CMU Serif', 'DejaVu Serif', 'Times New Roman', 'serif'],
    'text.usetex': False, 
    
    # Line properties
    'lines.linewidth': 1.5,          # thicker lines for better visibility after scaling
    'lines.markersize': 6,
    
    # Axes properties
    'axes.titlesize': 12,            # title size
    'axes.labelsize': 10,            # label size
    'axes.linewidth': 1,             # axis line width
    'axes.grid': True,               # enable grid
    'grid.alpha': 0.7,               # grid transparency
    
    # Tick properties
    'xtick.labelsize': 10,           # x tick label size
    'ytick.labelsize': 10,           # y tick label size
    'xtick.major.size': 5,           # major tick size
    'ytick.major.size': 5,           # major tick size
    'xtick.minor.size': 3,           # minor tick size
    'ytick.minor.size': 3,           # minor tick size
    'xtick.direction': 'in',         # ticks inside
    'ytick.direction': 'in',
    
    # Legend properties
    'legend.fontsize': 10,           # legend font size
    'legend.loc': 'best',            # legend location
    'legend.frameon': True,
    
    # Other properties
    'figure.autolayout': True,       # automatic layout adjustment
    'image.cmap': 'viridis',         # default colormap
}


_dicb = {
    'axes.grid': False,
    # spines
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    # ticks color
    'xtick.color': '[0,0,0,0.3]',
    'ytick.color': '[0,0,0,0.3]',
    # ticks width
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # ticks label color
    'xtick.labelcolor': '[0,0,0,1.]',
    'ytick.labelcolor': '[0,0,0,1.]',
}

plt.rcParams.update(_dica | _dicb)

# >> COLORS <<
from palettable.colorbrewer.diverging import RdBu_9 as pcmap

default_cmap = pcmap.mpl_colormap
default_norm = mpl.colors.LogNorm(vmin=1/10., vmax=10.0)

from palettable.cartocolors.qualitative import Antique_2 as qcmap
qcmap = qcmap.mpl_colors


# >> FUNCTIONS <<

## Labels

def magnitude_format(x, pos=None):
    from re import search

    if isinstance(x, str):
        x = f"{float(x):.2e}"
    else:
        x = f"{x:.2e}"

    match = search("(.*)e(.*)", x)
    
    if match is None:
        return ''
    else:
        value = float(match[1])
        exponent = int(match[2])
        
        if value > 3.16:
            exponent += 1
            return f"$10^{{{exponent:g}}}$"
        else:
            return f"$10^{{{exponent:g}}}$"

def my_savefig(fig, name):
    fig.savefig(f"{name}.png", dpi=600)
    plt.close();
    # fig.savefig(f"{name}.eps")
    # fig.savefig(f"../figs/{name}_small.png", dpi=150)
    # fig.savefig(f"../figs/{name}.pdf")
    # fig.savefig(f"../figs/{name}.svg")