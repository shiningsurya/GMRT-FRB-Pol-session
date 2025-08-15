"""
reads a psrchive file and saves relevant data into npz file
"""
import os
import json
import pickle as pkl

try:
    import psrchive
except ImportError:
    print (" psrchive-python is required for this script...")
    print (" Please ensure it is installed")
    import sys
    sys.exit (0)

import numpy as np

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('make_pkg', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-O','--outdir', help='Output directory', dest='odir', default="./")
    add ('-j','--json', default=None, help="JSON file containing tstart,tstop,fstart,fstop", dest='json')
    add ('-n','--no-json', action='store_true', help='Make pkg without json', dest='nojson')
    add ('-DD',action='store_true', help='Do not de-disperse', dest='DD')
    add ('-RR', action='store_true', help='Do not remove baseline', dest='RR')
    add ('file', help="archive file")
    return ag.parse_args ()


if __name__ == "__main__":
    args    = get_args ()
    bn      = os.path.basename ( args.file )
    if args.odir:
        outfile = os.path.join ( args.odir, bn + ".pkg.npz" )
    else:
        outfile =  args.file + ".pkg.npz"
    ################################
    ## json logic
    ran   = dict()
    if args.json:
        with open (args.json, 'rb') as f:
            ran = json.load (f)
    elif not args.nojson:
        with open (args.file+".json", 'r') as f:
            ran = json.load (f)
    ################################
    ff  = psrchive.Archive_load ( args.file )
    ff.convert_state ('Stokes')
    if not args.RR:
        ff.remove_baseline ()
    if not args.DD:
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
    with open (outfile, 'wb') as f:
       np.savez (f, **dict(
           data=data, wts=wts, freqs=freqs,
           bandwidth=fbw, center_freq=fcen,
           mjd=start_time, src=src, duration=dur,
           basis=basis
       ))
