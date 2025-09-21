"""

show QUV individual


"""
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('../thesisplot.mplstyle')
import matplotlib.colors as mc
import matplotlib.path as mp
import matplotlib.patches as mpa
from matplotlib.collections import LineCollection

import warnings

from pell import PolarizationEllipse as PEllipse

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

if __name__ == "__main__":

    """
    beta is phase difference


    for pure-V
        sin(2beta)
        axial ratio is +- 45 deg
        which happens when
        sin^2(2psi)sin^2(delta) = 1

        delta = +- 90 deg
        psi   = +- 45 deg

    for pure-Q
        cos(2alpha) cos(2beta) = +- 1
        (alpha,beta) = (0,0), (0,pi), (pi,0), (pi,pi)
        (alpha,beta) = (0,0), (pi,0)
        --
        axial ratio is 0 deg or 90 deg can only be zero
        happens when 
        sin^2(2psi)sin^2(delta) = 0
        delta =  0  or psi=0 or +-pi/2
        alpha = psi or  alpha=0 

    for pure-U
        sin(2alpha) cos(2beta)
        sin(2alpha) = +-1 ==> 2alpha = +- pi/2
        alpha = +- pi/4



    """
    PI1  = float ( np.deg2rad ( 180 ) )
    PI0  = float ( np.deg2rad ( 0  ) )
    PI2  = float ( np.deg2rad ( 90 ) )
    PI4  = float ( np.deg2rad ( 45 ) )

    pels = dict()
    ## PEllipse ( alpha, beta )
    pels['pQ']   = PEllipse ( 0, 0.0 )
    pels['mQ']   = PEllipse ( PI2, 0.0 )

    pels['pU']   = PEllipse ( PI4, 0.0 )
    pels['mU']   = PEllipse ( -PI4, 0.0 )
    
    pels['pV']   = PEllipse ( PI4, PI4 )
    pels['mV']   = PEllipse ( PI4, -PI4 )

    ##################################################################
    norm  = mc.Normalize ( 0., 360 )
    sc    = plt.cm.ScalarMappable ( norm, 'gist_rainbow' )

    fig   = plt.figure ('drawell', figsize=(8,8), layout='constrained')

    axes  = fig.subplot_mosaic ( [['pQ', 'pU', 'pV'], ['mQ', 'mU','mV'], ['cb','cb','cb']], sharex=False, sharey=False, height_ratios=[0.48, 0.48, 0.04])
    cbax  = fig.colorbar ( sc, cax=axes['cb'], orientation='horizontal', label='Phase / deg' )

    _lx   = np.linspace ( -1.0, 1.0, 2048 )


    for k in ['pQ', 'pU', 'pV', 'mQ', 'mU', 'mV']:
        ax = axes[k]
        kp = pels[k]
        ##

        # ax.scatter ( kp.ex, kp.ey, marker='s', c=col, s= )
        # ax.plot ( kp.ex, kp.ey )
        colored_line ( kp.ex, kp.ey, TAXIS, ax, cmap='gist_rainbow', linewidth=10, zorder=10 )

        # ax.scatter ( kp.ex[0], kp.ey[0], marker='D', c='k', zorder=100)

        # ax.plot ( _lx, np.tan(kp.pa) * _lx, ls=':', c='k', zorder=200 )
        ax.set_facecolor ( "#eeeeee" )

        ax.set_xlim (-1., 1.)
        ax.set_ylim (-1., 1.)
        # ax.set_aspect('equal')
        ax.spines[['left','bottom']].set_position('center')
        ax.spines[['right','top']].set_visible(False)
        ax.set_xticks ([-1.0,  1.0], labels=["", ""])
        ax.set_yticks ([-1.0,  1.0], labels=["",""])
        ax.tick_params (axis='x', top=False, which='both')
        ax.tick_params (axis='y', right=False, which='both')
        ax.tick_params (axis='x', which='minor', bottom=False)
        ax.tick_params (axis='y', which='minor', left=False)
        ax.tick_params (axis='both', direction='inout')

        # ax.axis ('off')
        ax.set_aspect('equal')

        ##
        # step = 32
        # for _i in range(0, 2048-step, step):
            # __a = mpa.FancyArrowPatch ( posA=(kp.ex[_i], kp.ey[_i]), posB=(kp.ex[_i+step], kp.ex[_i+step]) )
            # ax.scatter ( kp.ex[_i], kp.ey[_i], marker='s', c='b')

            # ax.add_artist ( __a )

    ## right for _neg
    # axes['pi0_pos'].scatter ( 0.75, -0.50, marker=r'$\circlearrowleft$', c='k', s=400 )
    # axes['pi1_pos'].scatter ( 0.75, -0.50, marker=r'$\circlearrowleft$', c='k', s=400 )
    # axes['pi0_neg'].scatter ( 0.75, -0.50, marker=r'$\circlearrowright$', c='k', s=400 )
    # axes['pi1_neg'].scatter ( 0.75, -0.50, marker=r'$\circlearrowright$', c='k', s=400 )

    # txtprop = dict(fontsize='large', weight='semibold')
    txtprop = dict()
    axes['pQ'].text ( -0.75, 1.00, "Q > 0", ha='center', va='top', **txtprop)
    axes['mQ'].text ( -0.75, 1.00, "Q < 0", ha='center', va='top', **txtprop)
    axes['pU'].text ( -0.75, 1.00, "U > 0", ha='center', va='top', **txtprop)
    axes['mU'].text ( -0.75, 1.00, "U < 0", ha='center', va='top', **txtprop)
    axes['pV'].text ( -0.75, 1.00, "V > 0", ha='center', va='top', **txtprop)
    axes['mV'].text ( -0.75, 1.00, "V < 0", ha='center', va='top', **txtprop)

    axes['mV'].scatter ( 0., 0., marker=r'$\circlearrowright$', c='k', s=400 )
    axes['pV'].scatter ( 0., 0., marker=r'$\circlearrowleft$', c='k', s=400 )

    # plt.show ()
    fig.savefig('pngs/quv.png', facecolor='#eeeeee')





