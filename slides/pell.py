"""
polarization ellipse

determining psi,delta from alpha,beta is so sensitive.
numerically it is difficult to find a direct solution
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as mpa

import scipy.optimize as so

N     = 2048
TAXIS = np.linspace ( 0., 2.0*np.pi, N ) 

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
    pell = PolarizationEllipse ( dr(45.0), dr(0.), phase=dr(0.), mag=1. )

    ###
    p1, p2 = pell.decompose ( dr(0.), dr(45.) )

    print ( pell, p1, p2, sep='\n' )


    fig  = plt.figure ('pell')
    ax   = fig.add_subplot ()


    ax.scatter ( pell.ex, pell.ey, marker='.', c='k' )
    ax.scatter ( p1.ex, p1.ey, marker='.', c='b' )
    ax.scatter ( p2.ex, p2.ey, marker='.', c='r' )
    # ax.scatter ( p1.ex+p2.ex, p1.ey+p2.ey, marker='.', c='g' )

    ax.set_aspect('auto')

    ax.set_xlim (-1., 1.)
    ax.set_ylim (-1., 1.)

    plt.show ()



