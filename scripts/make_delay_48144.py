"""
prepares crosshand delay correction
using J0139
specifically made for 48_144 dataset where they did not 
observe neither a quasar nor a noise diode.
"""
import os
import json

import numpy as np
import pandas as pd
from io import StringIO

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs

import astropy.time  as at
import astropy.units as au
import astropy.coordinates as asc

from astropy.io import fits

################################
#RAD,DECD         = dict(),dict()
#RAD['3C138']     = 79.5687917
#DECD['3C138']    = 16.5907806
#RAD['R3']        = 29.50312583
#DECD['R3']       = 65.71675422
#RAD['3C48']      = 24.4220417
#DECD['3C48']     = 33.1597417
datxt = r"""
3C138 79.5687917 16.5907806
3C48  24.4220417 33.1597417
R3    29.50312583 65.71675422
0217+738 34.882245 73.9229
"""
dadf  = pd.read_csv (StringIO(datxt), names=['source','radeg','decdeg'], sep='\\s+').set_index('source')

C                = 299792458.0 # m/s

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("make_delay", description="Makes a pacv calibration solution in circular basis", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('ar_file', help=":prepare_48144.py:")
    add ('--delays_grid', help='Delays grid (min:max:steps)', dest='delays_grid', default="-100:100:8192")
    add ('-O', '--outdir', help='Output directory', default="./", dest='odir')
    return agp.parse_args ()

if __name__ == "__main__":
    args = get_args ()
    ###################################################
    ### delay grid
    ### input validation
    ###################################################
    __dgrid     = args.delays_grid.split(':')
    try:
        delays_grid = np.linspace ( float ( __dgrid[0] ), float ( __dgrid[1] ), int ( __dgrid[2] ) )
    except: 
        raise RuntimeError (f"Delays grid not understood, input={args.delays_grid}")

    ###################################################
    ### prepare files/filenames
    ###################################################
    AR_FILE     = args.ar_file
    base,_      = os.path.splitext ( os.path.basename ( AR_FILE ) )
    pfile       = os.path.join ( args.odir, base + ".pcal.png" )
    zfile       = os.path.join ( args.odir, base + ".pcal.npz" )
    ofile       = base + ".pcal.pacv"
    outfile     = os.path.join ( args.odir, ofile  )
    RET         = dict()
    ###################################################
    ### read calibrator file
    ###################################################
    ## read
    pkg         = np.load ( AR_FILE )
    freqs_mhz   = pkg['freqs']
    freqs_ghz   = freqs_mhz * 1E-3
    wav2        = pkg['lam2']
    qq          = pkg['Q']
    uu          = pkg['U']
    ll          = qq - 1.0j*uu
    RET['freqs']=freqs_mhz
    RET['wav2'] = wav2
    print (f"**************************")
    print (f" THERE IS A SIGN FLIP IN STOKES-U SOMEWHERE")
    print (f" I AM HAVING TO DO IT TO GET GOOD XDELAY")
    print (f"**************************")
    pa          = 0.5 * np.angle(ll)
    #### source RM
    ## J0139+5814 psrcat RM
    srcrm       = -94.13 
    #### correct for both
    rm_corr     = srcrm
    #### logging
    print (f" Correcting RM     = {rm_corr:.3f}")
    pa_corr     = (rm_corr * wav2)
    ##################################################
    ### math
    ###################################################
    ######### PA correction
    data_pa     = 0.5 * np.angle ( ll * np.exp(-2.0j*pa_corr) )
    ##### delay_grid
    dmags       = np.zeros_like ( delays_grid )
    for i, idelay in enumerate ( delays_grid ):
        dmags [ i ]  = np.abs ( np.mean ( np.exp ( 2.0j * ( data_pa - ( 0.5 * idelay * np.pi * freqs_ghz ) ) ) ) )
    delay_ns    = delays_grid [ np.argmax ( dmags ) ]
    ### estimate bias
    biaser      = np.exp ( 2.0j * ( data_pa - ( 0.5 * delay_ns * np.pi * freqs_ghz ) ) )
    bias_rad    = np.angle ( np.sum ( biaser ) )
    model_pa    = 0.5 * np.angle ( np.exp ( 1.0j * ( ( delay_ns * np.pi * freqs_ghz ) + ( bias_rad ) ) ) )
    dphase      = 0.5 * np.angle ( np.exp ( -1.0j * ( ( delay_ns * np.pi * freqs_ghz ) + ( bias_rad ) ) ) )
    error_pa    = np.arctan ( np.tan ( data_pa - model_pa ) )
    ##################################################
    st        = f" estimated cable delay = {delay_ns:.3f} ns | bias_rad = {bias_rad:.3f} rad"
    RET['delay'] = delay_ns
    RET['bias_rad'] = bias_rad
    RET['res']   = error_pa
    RET['model'] = model_pa
    RET['dphase'] = dphase
    np.savez ( zfile, **RET )
    print ( st )
    ### make diagnostic plot
    ###################################################
    fig       = plt.figure ('pacv_circ')
    """
    dphase
    error
    """
    gs        = mgs.GridSpec ( 3, 1, figure=fig )
    axdd      = fig.add_subplot ( gs[0] )
    axpa      = fig.add_subplot ( gs[1] )
    axep      = fig.add_subplot ( gs[2], sharex=axpa )
    ################
    axdd.plot ( delays_grid, dmags, c='b', marker='.' )
    axdd.axvline ( delay_ns, ls=':', c='k' )
    axdd.set_xlabel ('Delays / ns')
    axdd.set_ylabel ('Magnitude')

    axpa.errorbar ( freqs_mhz, data_pa, c='k', marker='.', capsize=5, ls='' )
    axpa.plot ( freqs_mhz, model_pa, c='b')

    axep.scatter ( freqs_mhz, error_pa, c='k', marker='.' )
    axep.set_xlabel ('Freq / MHz')
    axep.set_ylabel ('Error / rad')
    axpa.set_ylabel ('PHI / rad')
    _ylow, _yhigh = axep.get_ylim ()
    __yy  = max ( abs ( _ylow ), abs ( _yhigh ) )
    axep.set_ylim ( -__yy, __yy )
    axep.axhline ( 0., ls=':', c='green' )

    fig.suptitle ( base+'\n'+st )

    fig.savefig ( pfile, bbox_inches='tight', dpi=300 )
    # plt.show ()


