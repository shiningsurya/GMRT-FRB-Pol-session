"""
Visualizes the pacv file
"""
import os
import sys
import json

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.colors as mc

import astropy.io.fits as aif

def get_args ():
    import argparse as agp
    ap   = agp.ArgumentParser ('vis_pacv', description='Visualizes pacv solution', epilog='Part of GMRT/FRB')
    add  = ap.add_argument
    add ('pacv', help='PACV solution FITS file')
    add ('-o','--opng', help='Save PNG file', default=None, dest='save_png')
    return ap.parse_args ()

def read_sol (f):
    """reads pacv and returns frequencies and parameters and their names"""
    cal   = aif.open (f)
    names = [fi.name for fi in cal]
    # print (f"received = {names}")
    # get SOLUTION TABLE
    cpar  = cal[2]
    # extract
    cdata = cpar.data['DATA'].reshape ((cpar.header['NCHAN'],cpar.header['NCPAR']))
    cerr  = cpar.data['DATAERR'].reshape ((cpar.header['NCHAN'],cpar.header['NCPAR']))
    freq  = cpar.data['DAT_FREQ'][0]
    wts   = np.array (cpar.data['DAT_WTS'][0], dtype=bool)
    names = [cpar.header[f"PAR_{i:04d}"] for i in range(cpar.header['NCPAR'])]
    return freq[wts], cdata[wts], cerr[wts], names

if __name__ == "__main__":
    args          = get_args ()

    fig              = plt.figure ("vis_pacv")
    axgg, axdg, axdp = fig.subplots ( 3, 1, sharex=True, )

    #######################################
    fw, sw, swerr, names = read_sol ( args.pacv )
    abs_gain      = sw[...,0]
    diff_gain     = sw[...,1]
    abs_gain_err  = swerr[...,0]
    diff_gain_err = swerr[...,1]
    ## got to unwrap the diff_phase
    # diff_phase    = np.unwrap (sw[...,2], period=np.pi)
    # diff_phase_err= np.unwrap (swerr[...,2], period=np.pi)
    diff_phase    = sw[...,2]
    diff_phase_err= swerr[...,2]
    fw_mhz        = fw * 1E6 
    #######################################

    axgg.errorbar ( fw, abs_gain, yerr=abs_gain_err, marker='.', c='b', capsize=5, ls='' )
    axdg.errorbar ( fw, diff_gain, yerr=diff_gain_err, marker='.', c='b', capsize=5, ls='' )
    axdp.errorbar ( fw, diff_phase, yerr=diff_phase_err, marker='.', c='b', capsize=5, ls='' )

    axgg.set_ylabel ('GAIN')
    axdg.set_ylabel ('DGAIN')
    axdp.set_ylabel ('DPHASE / rad')

    axdp.set_xlabel ('Frequency / MHz')

    fig.suptitle ( os.path.basename(args.pacv) )

    if args.save_png:
        fig.savefig (args.save_png, dpi=300, bbox_inches='tight')
    else:
        plt.show ()

