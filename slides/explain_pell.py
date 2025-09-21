import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.path as mp
import matplotlib.patches as mpa
from matplotlib.collections import LineCollection

import scipy.optimize as so

from pell import PolarizationEllipse, colored_line, TAXIS

if __name__ == "__main__":
    dr   = np.deg2rad
    given = PolarizationEllipse ( dr(45.0), dr(10.), phase=dr(0.), mag=1. )

    pa    = dr(45.0)
    ap    = -dr(45.0)

    ONG   = "pngs/pell.png"

    fig   = plt.figure ('pell')

    ax    = fig.add_subplot ()
    ax.set_facecolor ( "#eeeeee" )

    norm  = mc.Normalize ( 0., 360 )
    sc    = plt.cm.ScalarMappable ( norm, 'gist_rainbow' )
    cax = fig.colorbar ( sc, ax=ax, orientation='vertical', label='Phase / deg' )

    colored_line ( given.ex, given.ey, TAXIS, ax, cmap='gist_rainbow', linewidth=4, zorder=10 )

    __x   = np.linspace ( -0.74, 0.74, 128 )
    ax.plot ( __x, np.tan(pa) * __x, ls='--', c='k' )
    __x   = np.linspace ( -0.169, 0.169, 128 )
    ax.plot ( __x, np.tan(ap) * __x, ls=':', c='r' )

    ax.spines[['left','bottom']].set_position('center')
    ax.spines[['right','top']].set_visible(False)
    ax.set_xlim (-1.1, 1.1)
    ax.set_ylim (-1.1, 1.1)
    # ax.axis('off')
    ax.set_aspect('equal')
    ##
    ax.set_xticks ([-1.0,  1.0], labels=["", ""])
    ax.set_yticks ([-1.0,  1.0], labels=["",""])
    ##
    ax.tick_params (axis='x', top=False, which='both')
    ax.tick_params (axis='y', right=False, which='both')
    ax.tick_params (axis='x', which='minor', bottom=False)
    ax.tick_params (axis='y', which='minor', left=False)
    ax.tick_params (axis='both', direction='inout')
    
    ax.scatter ( 0.7, -0.7, marker=r'$\circlearrowleft$', c='k', s=100 )

    # plt.show ()
    fig.savefig(ONG, dpi=300, bbox_inches='tight', facecolor='#eeeeee')
