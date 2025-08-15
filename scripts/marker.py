
import os
import sys
import json

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.widgets import RangeSlider, RectangleSelector

import astropy.io.fits as aif

def block_reduce (x, fac, func=np.mean):
    ''' for time/frequency downsampling  ''' 
    xs  = x.shape
    rxs = ()
    mxs = ()
    ii  = 1
    for i, f in zip (xs, fac):
        rxs += (int(i//f), f)
        mxs += (ii,)
        ii  += 2
    # oxs = (int(xs[0]//fac[0]), int(xs[1]//fac[1]))
    # dx  = x.reshape (rxs).mean (mxs)
    dx  = func (x.reshape (rxs), axis=mxs)
    return dx

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('marker', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('file', help="archive file")
    add ('-t','--tscrunch', help='Time scrunch', default=4, type=int, dest='ts')
    add ('-f','--fscrunch', help='Freq scrunch', default=16, type=int, dest='fs')
    return ag.parse_args ()

def dd_process (idd):
    """preprocesses filterbank"""
    nch, nbin = idd.shape
    ### remove per channel mean/std.dev
    odd    = np.float32 (idd)
    odd    -= np.mean(odd, 1).reshape ((nch, 1))
    sshape = np.std (odd,1).reshape ((nch, 1))
    odd    = np.divide (odd, sshape, out=np.zeros_like (odd), where=sshape != 0.)
    return odd

def read_ar (f):
    """reads the AR file without using psrchive """
    f       = aif.open (f, ignore_missing_simple=True)
    #### get SUBINT table
    names   = [fi.name for fi in f]
    idx     = None
    try:
        idx = names.index ('SUBINT')
    except:
        raise RuntimeError (" SUBINT table not found")
    #### get nchan, npol
    fh      = f[idx].header
    fd      = f[idx].data
    nchan   = fh['NCHAN']
    npol    = fh['NPOL']
    tbin    = fh['TBIN'] # second
    fbw     = fh['CHAN_BW'] # MHz
    #### get scales, offsets, weights and data
    scl     = fd['DAT_SCL'].reshape ((1, npol, nchan, 1))
    offs    = fd['DAT_OFFS'].reshape ((1, npol, nchan, 1))
    wts     = fd['DAT_WTS']
    mask    = wts[0] == 0.
    #### get stokes I filterbank and apply mask
    dd      = (scl * fd['DATA']) + offs
    dd[...,mask,:] = np.nan
    #### #### coherence products still
    idd     = dd[0,0] + dd[0,1]
    return dd_process ( idd ), tbin, fbw

def tester (f):
    """
    dummy test
    """
    return np.random.randn(512, 1024), 327.68E-6, 0.09765625

if __name__ == "__main__":
    ##
    args    = get_args ()
    FAC     = (args.fs, args.ts)
    ##
    dd, tbin_s, fbw_mhz    = read_ar ( args.file )
    # dd      = tester ( args.file )
    nchan, nsamp = dd.shape
    dd      = block_reduce ( dd, FAC, func=np.nanmean )
    ##
    pp      = np.nanmean ( dd, 0 )
    ff      = np.nanmean ( dd, 1 )
    times   = np.linspace (0, nsamp, dd.shape[1])
    freqs   = np.linspace (0, nchan, dd.shape[0])
    #############################
    outfile = args.file + ".json"
    ####
    fig        = plt.figure ('marker')

    # fig.canvas.mpl_connect ('key_press_event', on_press)

    fbx      = plt.subplot2grid ( (5,5), (1,0), rowspan=4, colspan=4, fig=fig )
    ppx      = plt.subplot2grid ( (5,5), (0,0), rowspan=1, colspan=4, fig=fig )
    sx       = plt.subplot2grid ( (5,5), (1,4), rowspan=4, colspan=1, fig=fig )

    slider = RangeSlider(sx, "Contrast", np.nanmin ( dd ), np.nanmax ( dd ), orientation='vertical', valfmt="%.2f")

    fb   = fbx.imshow ( dd, aspect='auto', interpolation='none', cmap='plasma', origin='lower', extent=[0, nsamp, 0, nchan] )
    ppx.step ( times, pp, where='mid', lw=1, color='blue' )
    ppx.sharex ( fbx )

    pp_start = ppx.axvline(0, color='k', lw=3)
    pp_stop  = ppx.axvline(nsamp, color='k', lw=3)

    fbx.set_xlim (-45, nsamp+45)
    fbx.set_ylim (-45, nchan+45)

    saverange = dict(tstart=0, tstop=0, fstart=0, fstop=0)

    def range_updater(val):
        """
        range_slider updater
        """
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        # Update the image's colormap
        fb.norm.vmin = val[0]
        fb.norm.vmax = val[1]

        # Update the position of the vertical lines
        # pp_start.set_xdata([val[0], val[0]])
        # pp_stop.set_xdata([val[1], val[1]])

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    def select_callback(eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        # print(f"The buttons you used were: {eclick.button} {erelease.button}")

        pp_start.set_xdata([x1, x1])
        pp_stop.set_xdata([x2, x2])

        fig.canvas.draw_idle()

        ############################
        ## adjust y1, y2
        y1     = max ( 0, y1 )
        y2     = min ( nchan, y2 )
        saverange['tstart'] = int ( np.floor ( x1 ) )
        saverange['tstop']  = int ( np.ceil  ( x2 ) )
        saverange['fstart'] = int ( np.floor ( y1 ) )
        saverange['fstop']  = int ( np.ceil  ( y2 ) )
        ## measured parameters
        saverange['width_ms'] = 1E3 * tbin_s * abs(x2 - x1)
        saverange['bw_mhz']   = fbw_mhz * abs(y2 - y1)
        # print ( f" ranges time=({saverange['tstart']:d},{saverange['tstop']:d}) freq=({saverange['fstart']:d},{saverange['fstop']:d})" )

    def save_as_json (event):
        if event.key == 'm' or event.key == 'M':
            with open (outfile, "w") as f:
                json.dump (saverange, f, indent=2)
            sys.exit (0)

    rectsel = RectangleSelector ( 
        fbx, select_callback, useblit=True, button=[1,3], interactive=True,
        minspanx = 8, minspany = 8, spancoords = 'data', 
        props = dict(facecolor='black', edgecolor='none', alpha=0.35, fill=True)
    )


    fbx.set_xlabel ('Time / unit')
    fbx.set_ylabel ('Freq / unit')

    ppx.xaxis.tick_top ()
    ppx.xaxis.set_label_position ('top')
    ppx.set_yticklabels ([])

    slider.on_changed(range_updater)
    ppx.set_title ( "Mark the ON region and press m to save..." )

    fig.canvas.mpl_connect('key_press_event', save_as_json)

    plt.show()


