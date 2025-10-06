"""
Makes a pacv file to be used in calibration
with polarization expressed in circular basis

extension of my_pacv

- designed for circular feeds
- corrects for ionospheric RM contribution in deriving calibration solution.
- corrects for parallactic angle and position angle of the source

"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs

import astropy.time  as at
import astropy.units as au
import astropy.coordinates as asc

from astropy.io import fits

try:
    import psrchive
except ImportError:
    print (" psrchive-python is required for this script...")
    print (" Please ensure it is installed")
    import sys
    sys.exit (0)

################################
RAD,DECD         = dict(),dict()
RAD['3C138']     = 79.5687917
DECD['3C138']    = 16.5907806
RAD['R3']        = 29.50312583
DECD['R3']       = 65.71675422
RAD['3C48']      = 24.4220417
DECD['3C48']     = 33.1597417

C                = 299792458.0 # m/s

def read_pkg ( ar_file ):
    """
    returns dict 
    taken from `make_pkg`
    """
    ff  = psrchive.Archive_load ( ar_file )
    ff.convert_state ('Stokes')
    ff.remove_baseline ()
    ff.dedisperse ()
    ###
    basis = ff.get_basis()
    nbin  = ff.get_nbin()
    nchan = ff.get_nchan()
    dur   = ff.get_first_Integration().get_duration()
    fcen  = ff.get_centre_frequency ()
    fbw   = ff.get_bandwidth ()
    freqs = fcen + np.linspace (-0.5 * fbw, 0.5 * fbw, nchan, endpoint=True)
    fchan = fbw / nchan
    ## center frequency is already centered
    # freqs += fchan
    tsamp = dur / nbin
    ###
    data  = ff.get_data ()
    #### making data and wts compatible
    ww = np.array (ff.get_weights ().squeeze(), dtype=bool)
    wts   = np.ones (data.shape, dtype=bool)
    wts[:,:,ww,:] = False
    mata  = np.ma.array (data, mask=wts, fill_value=np.nan)
    ###
    start_time   = ff.start_time ().in_days ()
    end_time     = ff.end_time ().in_days ()
    mid_time     = 0.5 * ( start_time + end_time )
    ###
    src          = ff.get_source ()
    ##########################################
    pkg  = dict(
       data=data, wts=wts, freqs=freqs,
       bandwidth=fbw, center_freq=fcen, nchan=nchan, nbin=nbin,
       mjd=start_time, src=src, duration=dur,
       basis=basis
    )
    return pkg

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("make_pacv_circ", description="Makes a pacv calibration solution in circular basis", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('ar_file', help="calibrator archive file")
    add ('-z','--zap', help='Zap the channels (comma-separated, start:stop)', dest='zap', default='')
    add ('--on', help='ON region in bins (comma-separated, start:stop)', dest='on_region',)
    add ('--off', help='OFF region in bins (comma-separated, start:stop)', dest='off_region',)
    add ('-O', '--outdir', help='Output directory', default="./", dest='odir')
    add ('-v','--verbose', action='store_true', dest='v')
    return agp.parse_args ()

def mask_maker ( mask_string, mask_size ):
    """
    takes initial mask and updates in place
    """
    masks   = np.zeros ( mask_size, dtype=bool )
    for ss in mask_string.split(','):
        if len (ss) == 0:
            continue
        start, stop = ss.split(':')
        lz  = slice ( int(start), int(stop) )
        masks[lz] = True
    return masks

if __name__ == "__main__":
    args = get_args ()
    ###################################################
    ### prepare files/filenames
    ###################################################
    AR_FILE     = args.ar_file
    ###################################################
    ### read calibrator file
    ###################################################
    ## read
    pkg         = read_pkg ( AR_FILE )
    freqs_mhz   = pkg['freqs']
    mata        = np.ma.MaskedArray ( pkg['data'], mask=pkg['wts'] )[0]
    mjd         = pkg['mjd']
    source      = pkg['src']
    nchan       = freqs_mhz.shape[0]
    fbw         = freqs_mhz[1] - freqs_mhz[0]
    bandwidth   = pkg['bandwidth']
    fcen        = pkg['center_freq']
    #### freqs
    freqs_ghz   = freqs_mhz * 1E-3 
    wav2        = np.power ( 299.792458 / freqs_mhz, 2.0 )
    ### (Stokes, freq, bin)
    ### make masks
    onmask      = mask_maker ( args.on_region, mata.shape[2] )
    ofmask      = mask_maker ( args.off_region, mata.shape[2] )
    fqmask      = mask_maker ( args.zap, mata.shape[1] )
    #### ON - OFF
    oo          = mata[...,onmask].mean(-1) - mata[...,ofmask].mean(-1)
    ooerr       = mata[...,ofmask].std(-1)
    fqmask     |= oo.mask[0]
    fqmask     |= oo.mask[1]
    fqmask     |= oo.mask[2]
    fqmask     |= oo.mask[3]
    #####################################
    ## oo == ( Stokes, freq )
    ## if oo[0,ifreq] <= ooerr[0] : flag it
    lz          = oo[0] <= 0 
    if np.any (lz):
        print (f" ON-OFF Stokes-I is below 0, this should not be, flagging it.")
        fqmask |= lz
    
    ## apply the mask to oo
    oo.mask[...,:] |= fqmask
    ooerr.mask[...,:] |= fqmask
    ## compute Stokes-I frequency averaged time series
    pp       = mata[0].mean(0)
    ###################################################
    ii,qq,uu,vv = oo
    data_pa     = 0.5 * np.arctan2 ( uu, qq )
    ###################################################
    fig       = plt.figure ('show_noise')

    dx, ax    = fig.subplots ( 2, 1, sharex=True, height_ratios=[0.2,0.8] )

    dx.set_facecolor ( "#eeeeee" )
    ax.set_facecolor ( "#eeeeee" )

    ax.plot ( freqs_mhz, ii, c='k', label='I' )
    ax.plot ( freqs_mhz, vv, c='g', label='V' )
    ax.plot ( freqs_mhz, qq, c='b', label='Q' )
    ax.plot ( freqs_mhz, uu, c='r', label='U' )
    ax.legend (loc='best')

    dx.scatter ( freqs_mhz, data_pa, marker='.', c='k' )

    ax.set_xlabel ('Freq / MHz')
    ax.set_ylabel ('Stokes')
    dx.set_ylabel ('Cross phase / rad')


    # fig.savefig ( pfile, bbox_inches='tight', dpi=300 )
    # plt.show ()
    fig.savefig("slides/pngs/visnoise.png", dpi=300, bbox_inches='tight', facecolor='#eeeeee')
