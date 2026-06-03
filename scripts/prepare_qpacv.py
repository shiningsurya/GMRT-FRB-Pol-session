"""
this is only because of singularity issue.

dspsr_cuda is the standard for psrchive/dspsr

but that is too old

this script prepares input for the :make_pacv_circ_q.py:
"""
from __future__ import print_function

import os
import json

import numpy as np
#import pandas as pd
from io import StringIO

try:
    import psrchive
except ImportError:
    print (" psrchive-python is required for this script...")
    print (" Please ensure it is installed")
    import sys
    sys.exit (0)

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
#dadf  = pd.read_csv (StringIO(datxt), names=['source','radeg','decdeg'], sep='\s+').set_index('source')
RAD,DECD         = dict(),dict()
toks   = datxt.split()
for k in range(0, len(toks), 3):
    src = toks [ k ]
    ra  = toks [ k + 1]
    dec = toks [ k + 2]
    RAD[src] = float ( ra )
    DECD[src] = float ( dec )

C                = 299792458.0 # m/s

def read_pkg ( ar_file, keep_baseline=False ):
    """
    returns dict 
    taken from `make_pkg`
    """
    ff  = psrchive.Archive_load ( ar_file )
    ff.convert_state ('Coherence')
    if not keep_baseline: ff.remove_baseline ()
    ff.dedisperse ()
    ###
    ff.tscrunch()
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
    ###
    srccoord     = ff.get_coordinates()
    srcra        = srccoord.ra().getDegrees()
    srcdec       = srccoord.dec().getDegrees()
    ##########################################
    pkg  = dict(
       data=data, wts=wts, freqs=freqs,
       bandwidth=fbw, center_freq=fcen, nchan=nchan, nbin=nbin,
       mjd=start_time, src=src, duration=dur,
       basis=basis,
       ra  = srcra, dec=srcdec
    )
    return pkg

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("prepare_qpacv", description="Prepares input for make_pacv_circ_q.", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('-q', '--qsr', help="quasar folded archive", required=True, dest='qsr')
    add ('-p', '--psr', help="pulsar folded archive", required=True, dest='psr')
    add ('--on', help='ON region in bins (comma-separated, start:stop) for pulsar', dest='on_region', required=True)
    add ('--off', help='OFF region in bins (comma-separated, start:stop) for pulsar and quasar', dest='off_region', required=True)
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
    OUT  = dict()
    ###################################################
    ### prepare files/filenames
    ### outfilename primarily from psr
    ###################################################
    base,_      = os.path.splitext ( os.path.basename ( args.psr ) )
    zfile       = os.path.join ( args.odir, base + ".prepare.qpacv.npz" )
    pfile       = os.path.join ( args.odir, base + ".prepare.qpacv.png" )
    ofile       = base + ".pcal.pacv"
    outfile   = os.path.join ( args.odir, ofile  )
    OUT['pfile']   = pfile
    OUT['ofile']   = pfile
    OUT['outfile'] = outfile
    OUT['base']    = base
    ###################################################
    ### read calibrator file
    ###################################################
    ## read quasar
    qsr_pkg     = read_pkg ( args.qsr, keep_baseline=False )
    qsr_mata    = np.ma.MaskedArray ( qsr_pkg['data'], mask=qsr_pkg['wts'] )[0]
    qsr_freqs   = qsr_pkg['freqs']
    ## read pulsar
    psr_pkg     = read_pkg ( args.psr, keep_baseline=False )
    freqs_mhz   = psr_pkg['freqs']
    psr_mata    = np.ma.MaskedArray ( psr_pkg['data'], mask=psr_pkg['wts'] )[0]
    mjd         = psr_pkg['mjd']
    psr         = psr_pkg['src']
    psr_ra      = psr_pkg['ra']
    psr_dec     = psr_pkg['dec']
    nchan       = freqs_mhz.shape[0]
    fbw         = freqs_mhz[1] - freqs_mhz[0]
    bandwidth   = psr_pkg['bandwidth']
    fcen        = psr_pkg['center_freq']
    #### freqs
    freqs_ghz   = freqs_mhz * 1E-3 
    wav2        = np.power ( 299.792458 / freqs_mhz, 2.0 )
    ### sanity check if frequency axis is the same
    if not np.all ( np.isclose ( qsr_freqs, freqs_mhz ) ):
        raise RuntimeError (" quasar frequency axis is not identical to pulsar frequency axis\n"
                            " should not be the case")
    ### (coherence, freq, bin)
    ### make masks
    onmask      = mask_maker ( args.on_region, psr_mata.shape[2] )
    ofmask      = mask_maker ( args.off_region, psr_mata.shape[2] )
    #### ON - OFF
    qsr_pp      = np.mean ( qsr_mata[0] + qsr_mata[1], 0 )
    qsr_mask    = qsr_pp >= (0.60 * qsr_pp.max())
    qsr_oo      = qsr_mata[...,qsr_mask].mean(-1) - qsr_mata[...,~qsr_mask].mean(-1)
    psr_oo      = psr_mata[...,onmask].mean(-1) - psr_mata[...,ofmask].mean(-1)
    psr_pp      = np.mean ( psr_mata[0] + psr_mata[1], 0 )
    #### errors only from pulsar
    #### the std axis=0 because numpy transposes for some reason
    err_aa      = qsr_mata[0,...,~qsr_mask].std(0)
    err_bb      = qsr_mata[1,...,~qsr_mask].std(0)
    err_cr      = psr_mata[2,...,ofmask].std(0)
    err_ci      = psr_mata[3,...,ofmask].std(0)
    ####
    ## needed only for diagnostics
    #OUT['qsr_mata']   = qsr_mata.filled(np.nan)
    #OUT['qsr_mask']   = qsr_mask.filled(np.nan)
    #####
    #### diagnostics
    OUT['psr_onmask'] = onmask
    OUT['psr_ofmask'] = onmask
    OUT['qsr']        = qsr_oo.filled(np.nan)
    OUT['psr']        = psr_oo.filled(np.nan)
    OUT['psr_pp']     = psr_pp.filled(np.nan)
    OUT['err_aa']     = err_aa.filled(np.nan)
    OUT['err_bb']     = err_bb.filled(np.nan)
    OUT['err_cr']     = err_cr.filled(np.nan)
    OUT['err_ci']     = err_ci.filled(np.nan)
    OUT['freqs_mhz']  = freqs_mhz
    OUT['wav2']       = wav2
    OUT['mjd']        = mjd
    OUT['psr_name']   = psr
    OUT['psr_ra']     = psr_ra
    OUT['psr_dec']    = psr_dec
    OUT['bandwidth']  = bandwidth
    OUT['center_freq']= fcen
    np.savez (zfile, **OUT)
