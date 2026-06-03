"""
2D
"""
from __future__ import print_function
import os
import sys
import json

from itertools import cycle
from collections import namedtuple

import numpy as np

try:
    import psrchive
except ImportError:
    print (" psrchive-python is required for this script...")
    print (" Please ensure it is installed")
    import sys
    sys.exit (0)

# from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.colors as mc

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

def write_csv ( cet, fname ):
    """
    need to remove pandas dependency
    """
    kstr, vstr = "",""
    for k,v in cet.items():
        kstr += str(k) + ","
        vstr += str(v) + ","
    with open ( fname, 'w' ) as _f:
        _f.write ( kstr[:-1] + "\n" + vstr[:-1] + "\n" )

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
       basis=basis,
    )
    return pkg

def read_prepare_2d ( 
        pkg,
        on_region,
        of_region,
        fscrunch,
        v=False
    ):
    """
    pkg : dictionary of `read_pkg`
    fscrunch: int 
    v: bool verbose flag
    returns
    freq_list, IQUV, errors(IQUV)
    """
    ##
    # pkg     = np.load ( pkg_file )

    ## read meta
    Nch     = int ( pkg['nchan'] / fscrunch )
    Nbin    = pkg['nbin']

    freqs     = np.array ( pkg['freqs'] )
    freq_list = np.array ( pkg['freqs'] )

    on_mask = np.zeros ( pkg['nbin'], dtype=bool )
    of_mask = np.zeros ( pkg['nbin'], dtype=bool )
    ff_mask = np.zeros ( pkg['nchan'], dtype=bool )

    ons     = slice ( on_region['tstart'], on_region['tstop'] )
    on_mask[ons]   = True

    ## 20230314 : everything that is not ON is OFF
    if off_region is None:
        of_mask[:] = True
        of_mask[on_region['tstart']:on_region['tstop']]   = False
    else:
        ofs            = slice ( off_region['tstart'], off_region['tstop'] )
        of_mask[ofs]   = True

    ofs     = slice ( on_region['fstart'], on_region['fstop'] )
    ff_mask[ofs]   = True
    wid     = on_region['tstop'] - on_region['tstart']

    if fscrunch > 1:
        print (" Frequency downsampling by {fs:d}\t {nch0:d} --> {nch1:d}".format (fs=fscrunch, nch0=pkg['nchan'], nch1=Nch))
        ff_mask     = np.array ( block_reduce ( ff_mask, (fscrunch,), func=np.mean ), dtype=bool )
        freqs       = np.array ( block_reduce (pkg['freqs'], (fscrunch,), func=np.mean ))
        freq_list   = np.array ( block_reduce (pkg['freqs'], (fscrunch,), func=np.mean ))

    # read data
    mata    = np.ma.MaskedArray ( pkg['data'][0], mask=pkg['wts'][0], fill_value=np.nan )
    # data    = block_reduce (  pkg['data'][0], (1, fscrunch, 1), func=np.mean )
    # wts     = np.ones (pkg['data'].shape, dtype=bool)
    # ww      = np.array (pkg['wts'], dtype=bool)
    # wts[:,:,ww,:] = False
    # ww      = block_reduce (  wts[0] ,  (1, fscrunch, 1), func=np.mean )

    # mata    = np.ma.array (data, mask=ww, fill_value=np.nan)
    # mata    = data
    mata    = mata.filled ( np.nan )
    mata    = block_reduce ( mata, (1, fscrunch, 1), func=np.nanmean )

    nsamp   = mata.shape[2]
    # mask    = ww[0].sum (1) == 0.0
    ff_mask = ff_mask

    # axes
    tsamp   = float (pkg['duration']) / float ( nsamp )
    times   = np.linspace ( 0., float(pkg['duration']), nsamp )
    times   *= 1E3
    # print ( times, tsamp, pkg['duration'], nsamp )
    # freqs     = np.linspace (-0.5*pkg['bandwidth'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']
    # freq_list = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']

    times     -= np.median (times[ons])
    btimes    = times[ons]

    freq_lo   = freq_list.min ()
    freq_hi   = freq_list.max ()

    ## Stokes ON pulse
    I_on    = np.array ( mata[0,ff_mask][...,on_mask] )
    Q_on    = np.array ( mata[1,ff_mask][...,on_mask] )
    U_on    = np.array ( mata[2,ff_mask][...,on_mask] )
    V_on    = np.array ( mata[3,ff_mask][...,on_mask] )

    ## Stokes OFF pulse
    I_off   = np.array ( mata[0,ff_mask][...,of_mask] )
    Q_off   = np.array ( mata[1,ff_mask][...,of_mask] )
    U_off   = np.array ( mata[2,ff_mask][...,of_mask] )
    V_off   = np.array ( mata[3,ff_mask][...,of_mask] )

    ## freq_list
    freq_list = freq_list [ ff_mask ]

    ## per channel std-dev
    I_std   = np.std ( I_off, 1 )
    Q_std   = np.std ( Q_off, 1 )
    U_std   = np.std ( U_off, 1 )
    V_std   = np.std ( V_off, 1 )

    ## Sum over ON pulse
    I_sum_on  = np.sum ( I_on, 1 )
    ## Choose high S/N, avoid channels with non-positive I
    omask     = np.zeros (I_sum_on.shape[0], dtype=bool)
    I_std_mask= np.std ( I_on, 1 )
    I_off_mean= np.mean (I_off, 1)
    for i,ii in enumerate (I_sum_on):
        if ( ii > 1.66 * I_std_mask[i] ) and ( ii > I_off_mean[i] ):
            omask[i] = True
    ## since i am manually selecting the subband

    I  = I_on [ omask ] -  np.mean (I_off [ omask ], 1)[:,np.newaxis]
    Q  = Q_on [ omask ] -  np.mean (Q_off [ omask ], 1)[:,np.newaxis]
    U  = U_on [ omask ] -  np.mean (U_off [ omask ], 1)[:,np.newaxis]
    V  = V_on [ omask ] -  np.mean (V_off [ omask ], 1)[:,np.newaxis]

    ## sum over time
    # I      = I.sum (1)
    # Q      = Q.sum (1)
    # U      = U.sum (1)
    # V      = V.sum (1)

    # nON       = np.sqrt ( ons.stop - ons.start )
    # if v:
        # print (" Number of ON samples = {on:d}".format(on=ons.stop - ons.start))

    # 20230313 : use whole pulse region to compute the standard deviation
    # 20230313 : and multiply with sqrt ( width )

    I_err     = I_std [ omask ]
    Q_err     = Q_std [ omask ]
    U_err     = U_std [ omask ]
    V_err     = V_std [ omask ]
    freq_list = freq_list [ omask ]

    return freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err
# from skimage.measure import block_reduce

def pa_meanstd ( pas, shiftpa=0.5*np.pi ):
    """
    wrap or not
    """
    sap  = np.arctan ( np.tan ( pas + shiftpa ) )
    ##
    mstd = pas.std ()
    nstd = sap.std ()
    if mstd < nstd:
        return np.mean ( pas ), mstd
    else:
        return np.arctan ( np.tan ( np.mean ( sap ) - shiftpa ) ), nstd

class RMPABootstrap:
    """
    bootstrapping RM and PA
    """
    def __init__ (self, w2, paw2, rm_grid):
        """
        could be masked but then masked are removed
        """
        __unmasked = np.logical_not ( paw2.mask )
        self.w2    = w2 [ __unmasked ]
        self.pa    = paw2 [ __unmasked ]
        ###
        self.rm_grid = rm_grid.copy ()

    def statistic (self, w2, pas):
        """
        ML estimate of RM and PA corresponding to the RM
        """
        ret     = [ np.abs ( np.sum ( np.exp ( 2.0j * ( pas - ( irm * w2 ) ) ) ) ) for irm in self.rm_grid ]
        rm_stat = self.rm_grid [ np.argmax ( ret ) ]
        pa_stat = 0.5 * np.angle ( np.sum ( np.exp ( 2.0j * ( pas - ( rm_stat * w2 ) ) ) ) )
        return rm_stat, pa_stat

    def __call__ (self, n_resamples=999, f_trial=0.85, confidence_level=0.95):
        """
        run the bootstrap

        follows
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
        but considers the wavelength2 as well.

        does alternative='two-sided' and method='basic'

        f_trial is fraction of samples per trial
        """
        ##
        ## ML estimate
        rm_ml, pa_ml = self.statistic ( self.w2, self.pa )

        nsamples     = self.w2.size
        ntrial       = int ( f_trial * nsamples )

        # rng          = np.random.default_rng()
        rng          = np.random

        re_stat      = np.zeros ( n_resamples, dtype=np.float32 )
        pe_stat      = np.zeros ( n_resamples, dtype=np.float32 )

        # for i_resample in tqdm ( range(n_resamples), desc='Bootstrap', unit='bt' ):
        for i_resample in range ( n_resamples ):
            print ("   Bootstrap: {i_resample:03d} / {n_resamples:03d} ... ".format(i_resample=i_resample, n_resamples=n_resamples), end='\r')
            # __i    = rng.choice ( nsamples, size=ntrial, replace=True, shuffle=False )
            __i    = rng.choice ( nsamples, size=ntrial, replace=True, )
            ## slice
            t_w2   = self.w2 [ __i ]
            t_pa   = self.pa [ __i ]
            ##
            re_stat [i_resample], pe_stat[i_resample]  = self.statistic ( t_w2, t_pa )

        alpha      = 0.5 * ( 1.0 - confidence_level )

        ## basic
        rm_low     = (2.0*rm_ml) - np.percentile ( re_stat, ( 1.0 - alpha ) * 100.0 )
        rm_high    = (2.0*rm_ml) - np.percentile ( re_stat, alpha * 100.0 )

        pa_low     = (2.0*pa_ml) - np.percentile ( pe_stat, ( 1.0 - alpha ) * 100.0 )
        pa_high    = (2.0*pa_ml) - np.percentile ( pe_stat, alpha * 100.0 )

        __pa_mean, __pa_std = pa_meanstd ( pe_stat )

        return dict(
            rm=rm_ml, rm_low=rm_low, rm_high=rm_high, rm_se=np.std(re_stat, ddof=1),
            pa=pa_ml, pa_low=pa_low, pa_high=pa_high, pa_se=__pa_std,
            rm_mean=np.mean(re_stat), pa_mean=__pa_mean
        ), re_stat, pe_stat

class PASpec:
    """
    PA spectrum man

    2D RM fitting

    Phi = 0.5 * arctan ( U / Q )

    Phi, Phierr = 

    RM Lambda^2 + Psi

    layout = (frequency,time)

    RMTF is disabled here
    """
    def __init__ (self, wave2, q, u, qerr, uerr, ierr, mpoints=128):
        """
        wave2: array
        stokesi: array

        data arrays are  (frequency, time)
        error arrays are (frequency,)
        """
        self.nchan, self.nsamp = q.shape
        self.q        = q
        self.u        = u
        ###
        self.l        = q + 1.0j*u
        self.l2       = np.ma.MaskedArray ( (q**2) + (u**2), mask=np.zeros_like(u, dtype=bool) )
        ### Everett, Weisburg mask
        # self.mask     = np.sqrt( self.l2 ) / ierr.reshape((-1,1)) < 1.57
        self.mask     = np.zeros_like (u, dtype=bool )
        ###
        self.lt       = self.l.mean (0)
        self.lf       = self.l.mean (1)
        # self.fmask    = np.sqrt( self.lf**2 ) / ierr < 1.57
        self.fmask     = np.zeros_like ( self.lf, dtype=bool )
        ###
        self.w2       = wave2.copy()
        self.w2d      = np.ma.MaskedArray ( np.repeat (wave2.reshape((self.nchan,1)), self.nsamp, axis=1) , mask=self.mask )
        self.pa       = np.ma.MaskedArray ( 0.5 * np.arctan2 ( u, q ), mask=self.mask )
        self.paerr    = 0.5 * np.sqrt( (q*uerr.reshape((-1,1)))**2 + (qerr.reshape((-1,1))*u)**2 ) / self.l2
        ###
        self.w2min    = wave2.min()
        self.w2max    = wave2.max()
        self.mw2      = np.linspace ( self.w2min, self.w2max, mpoints, endpoint=True )

    def subtract_patime (self, patime):
        """
        subtracts pa(time) and averages
        """
        self.qu_sub    = self.l * np.exp ( -2.0j * ( patime.reshape((1, self.nsamp)) ) )
        self.pa_sub    = np.ma.MaskedArray ( 0.5 * np.angle ( self.qu_sub.mean(1) ), mask=self.fmask )

    def rmtf  ( self, rms, peak_rm ):
        """
        RM transfer function
        this method does not use any w2

        rms is (nrms,)
        peak_rm is float

        the expectation is that this function will only be called once with peak_rm as the 
        final RM (averaged or something).
        """
        nrms = rms.size

        ret  = np.zeros ((nrms,), dtype=np.complex64)

        for irm in range ( nrms ):
            """
            one convention needs an additional sign flip later.
            other convention already incorporates that.

            i am not sure where this sign flip is coming from.
            """
            ## one convention
            # _rm  = rms[irm] - peak_rm
            ## other
            _rm  = peak_rm - rms[irm]
            ret[irm] = np.sum ( np.exp ( 2.0j * (  _rm * self.w2  ) ) )
        
        mag  = np.abs ( ret )
        pa   = 0.5 * np.angle ( ret )

        # wpa  = np.unwrap ( pa, period=np.pi )
        wpa  = np.unwrap ( pa, discont=0.5*np.pi )

        slope, _ = np.polyfit ( rms, wpa, 1 )

        return {'rmtf_mag':mag, 'rmtf_pa':pa, 'rmtf_pa_slope':slope, 'peak_rm':peak_rm}

    def rm_spectrum (self, rms):
        """
        rm spectra?
        return magnitude?
        """
        nrms = rms.size

        ret  = np.zeros ((nrms,self.nsamp), dtype=np.complex64)

        for irm in range ( nrms ):
            _rm  = rms[irm]
            ret[irm] = np.sum ( np.exp ( 2.0j * ( self.pa - ( _rm * self.w2d ) ) ), axis=0 )

        # return np.abs ( ret )
        return np.abs ( ret ), 0.5 * np.angle ( ret )

    def bootstrap_rmpa (self, rms, n_resamples=999, f_trial=0.90, confidence_level=0.95):
        """
        estimate RM error using bootstrap
        """
        boot   = RMPABootstrap ( self.w2, self.pa_sub, rms )
        res, rm_boot, pa_boot    = boot (n_resamples=n_resamples, f_trial=f_trial, confidence_level=confidence_level)
        return res, rm_boot, pa_boot

    def pa_time (self, rm):
        """
        subtracts pa(time) and averages
        """
        mqu       = self.l * np.exp ( -2.0j * ( self.w2d * rm ) )
        mpa       = 0.5 * np.angle ( mqu.mean(0) )
        return mpa

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

C      = 299.792458 # 1E6 * m / s 

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('prepare_onerm', epilog='Part of GMRT/FRB polarization pipeline')
    add  = ag.add_argument
    add ('ar_file', help="burst archive file")
    add ('-f','--fscrunch', default=4, type=int, help='Frequency downsample', dest='fs')
    add ('-j','--json', help='JSON file', default=None, dest='json')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='./', dest='odir')
    ##
    return ag.parse_args ()

if __name__ == "__main__":
    args    = get_args ()
    ####################################
    bn      = os.path.basename ( args.ar_file )
    bnf     = split_extension ( bn )
    odir    = args.odir
    ####################################
    on_region  = dict()
    if args.json:
        with open (args.json, 'rb') as f:
            on_region.update ( json.load (f) )
    else:
        with open (args.ar_file+".json", 'r') as f:
            on_region.update ( json.load (f) )

    off_region = None
    ####################################
    pkg       = read_pkg ( args.ar_file )
    data_freq_low    = np.min ( pkg['freqs'] )
    data_freq_high   = np.max ( pkg['freqs'] )
    ####################################
    freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err = read_prepare_2d (
        pkg,
        on_region,
        off_region,
        args.fs,
        args.v
    )

    ####################
    ## do time averaging
    ## need to sum  c.f. :read_prepare_tscrunch:
    I               = np.sum ( I, axis=1 ).reshape((-1,1))
    Q               = np.sum ( Q, axis=1 ).reshape((-1,1))
    U               = np.sum ( U, axis=1 ).reshape((-1,1))
    V               = np.sum ( V, axis=1 ).reshape((-1,1))
    ####################

    PA              = 0.5 * np.arctan2 ( U, Q )
    freqs           = np.repeat ( freq_list.reshape((-1,1)), I.shape[1],  axis=1, )
    w2              = np.power ( C / freqs, 2 )


    RET             = dict ()
    RET['filename'] = bn
    RET['fs']       = args.fs

    RET['freqs']    = freqs
    RET['lam2']     = w2
    RET['I']        = I
    RET['Q']        = Q
    RET['U']        = U
    RET['V']        = V
    RET['PA']       = PA
    
    RET['I_err']    = I_err.reshape((-1,1))
    RET['Q_err']    = Q_err.reshape((-1,1))
    RET['U_err']    = U_err.reshape((-1,1))
    RET['V_err']    = V_err.reshape((-1,1))

    for k,v in RET.items():
        if np.ma.isMaskedArray ( v ): RET[k] = np.array ( v.filled(np.nan) )
    np.savez ( os.path.join ( args.odir, bn + "_prepare.npz"), **RET)




