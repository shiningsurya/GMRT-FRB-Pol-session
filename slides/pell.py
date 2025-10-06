"""
polarization ellipse

determining psi,delta from alpha,beta is so sensitive.
numerically it is difficult to find a direct solution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.path as mp
import matplotlib.patches as mpa
from matplotlib.collections import LineCollection

import scipy.optimize as so

N     = 2048
TAXIS = np.linspace ( 0., 2.0*np.pi, N ) 

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

class PolarizationEllipse:
    def __init__ (self, alpha, beta, phase=0., mag=1.):
        """
        Eq. 1.9

        alpha, beta, phase in radians
        """
        cb, sb  = np.cos ( beta ), np.sin ( beta )
        ca, sa  = np.cos ( alpha ), np.sin ( alpha )

        ct  = np.cos ( TAXIS + phase )
        st  = np.sin ( TAXIS + phase )

        self.ex  = mag * ( ca*cb*ct - sa*sb*st )
        self.ey  = mag * ( sa*cb*ct + ca*sb*st )

        rd  = np.rad2deg
        self.txt = f" alpha={rd(alpha):.3f} beta={rd(beta):.3f} phase={rd(phase):.3f} mag={mag:1f}"

        self.alpha = alpha
        self.beta  = beta
        self.mag   = mag

    def __str__ (self,):
        return self.txt

    def decompose (self, alpha1, beta1):
        """
        oppositely polarized decomposition

        g1  = alpha - alpha1
        g2  = alpha - alpha2 = alpha - alpha1 + pi/2
        alpha2 = alpha1 - pi/2
        """
        g1  = self.alpha - alpha1

        ### opposite filter
        alpha2 = alpha1 - (0.5*np.pi)
        beta2  = -beta1

        z1  = np.cos(g1)*np.cos(self.beta-beta1) + 1.0j*np.sin(g1)*np.sin(self.beta+beta1)
        z2  =-np.sin(g1)*np.cos(self.beta+beta1) + 1.0j*np.cos(g1)*np.sin(self.beta-beta1)

        z1 *= self.mag
        z2 *= self.mag
        ######################
        p1  = PolarizationEllipse ( alpha1, beta1, phase=np.angle(z1), mag=np.abs(z1) )
        p2  = PolarizationEllipse ( alpha2, beta2, phase=np.angle(z2), mag=np.abs(z2) )

        return p1, p2


if __name__ == "__main__":
    dr   = np.deg2rad
    given = PolarizationEllipse ( dr(60.0), dr(0.), phase=dr(0.), mag=1. )

    # ONG   = "pngs/base.png"
    # comp1, comp2 = given.decompose ( dr(60.), dr(20.) )
    # ONG   = "pngs/linear.png"
    # comp1, comp2 = given.decompose ( dr(0.), dr(0.) )
    # ONG   = "pngs/circular.png"
    # comp1, comp2 = given.decompose ( dr(0.), dr(45.) )
    ONG   = "pngs/fr0.png"
    comp1, comp2 = given.decompose ( dr(0.), dr(45.) )

    print ( given, comp1, comp2, sep='\n' )

    fig  = plt.figure ('decomp')

    axg, ax1, ax2 = fig.subplots ( 1, 3, sharex=True, sharey=True)

    norm  = mc.Normalize ( 0., 360 )
    sc    = plt.cm.ScalarMappable ( norm, 'gist_rainbow' )

    cax = fig.colorbar ( sc, ax=[axg, ax1, ax2], orientation='horizontal', label='Phase / deg' )
    # cax = fig.colorbar ( sc, ax=[ax2], orientation='vertical', label='Phase / deg' )

    colored_line ( given.ex, given.ey, TAXIS, axg, cmap='gist_rainbow', linewidth=4, zorder=10 )

    colored_line ( comp1.ex, comp1.ey, TAXIS, ax1, cmap='gist_rainbow', linewidth=4, zorder=10 )

    colored_line ( comp2.ex, comp2.ey, TAXIS, ax2, cmap='gist_rainbow', linewidth=4, zorder=10 )

    ## put phase start
    # ax1.scatter ( comp1.ex[0], comp1.ey[0], marker='D', c='k', zorder=100 )
    # ax2.scatter ( comp2.ex[0], comp2.ey[0], marker='D', c='k', zorder=100 )

    for ax, ee in zip ([axg, ax1, ax2],[given, comp1, comp2]):
        ax.spines[['left','bottom']].set_position('center')
        ax.spines[['right','top']].set_visible(False)
        ax.set_xlim (-1.1, 1.1)
        ax.set_ylim (-1.1, 1.1)
        # ax.axis('off')
        ax.set_aspect('equal')
        ax.set_facecolor ( "#eeeeee" )

        ##
        ax.set_xticks ([-1.0,  1.0], labels=["", ""])
        ax.set_yticks ([-1.0,  1.0], labels=["",""])
        ##
        ax.tick_params (axis='x', top=False, which='both')
        ax.tick_params (axis='y', right=False, which='both')
        ax.tick_params (axis='x', which='minor', bottom=False)
        ax.tick_params (axis='y', which='minor', left=False)
        ax.tick_params (axis='both', direction='inout')

    axg.scatter ( 0.7, -0.7, marker=r'$\circlearrowleft$', c='k', s=100 )
    ax1.scatter ( -0.7, -0.7, marker=r'$\circlearrowleft$', c='k', s=100 )
    ax2.scatter ( -0.7, -0.7, marker=r'$\circlearrowright$', c='k', s=100 )
    
    axg.set_title('Given')

    ax1.set_title("First\ncomponent")

    ax2.set_title("Second\ncomponent")

    axg.text (1.3, 0., "=", ha='center', va='center', weight='bold', fontsize='x-large', color='blue')
    ax1.text (1.3, 0., "+", ha='center', va='center', weight='bold', fontsize='x-large', color='blue')

    # plt.show ()
    fig.savefig(ONG, dpi=300, bbox_inches='tight', facecolor='#eeeeee')



