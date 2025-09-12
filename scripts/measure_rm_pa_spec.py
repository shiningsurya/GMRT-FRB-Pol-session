"""


"""
from __future__ import print_function
import os
import sys
import json

import numpy as np
import pandas as pd


try:
    import psrchive
except ImportError:
    print (" psrchive-python is required for this script...")
    print (" Please ensure it is installed")
    import sys
    sys.exit (0)

# from tqdm import tqdm

import matplotlib
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

def read_prepare_tscrunch ( 
        pkg,
        on_region,
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
    of_mask = np.ones ( pkg['nbin'], dtype=bool )
    ff_mask = np.zeros ( pkg['nchan'], dtype=bool )

    ## 20230314 : everything that is not ON is OFF
    ons     = slice ( on_region['tstart'], on_region['tstop'] )
    on_mask[ons]   = True
    of_mask[on_region['tstart']:on_region['tstop']]   = False

    ofs     = slice ( on_region['fstart'], on_region['fstop'] )
    ff_mask[ofs]   = True
    wid     = on_region['tstop'] - on_region['tstart']

    if fscrunch > 1:
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
    mata    = block_reduce ( mata, (1, fscrunch, 1), func=np.mean )

    if fscrunch > 1:
        print (" Frequency downsampling by {fs:d}\t {nch0:d} --> {nch1:d}".format (fs=fscrunch, nch0=pkg['nchan'], nch1=Nch))

    # mata    = np.ma.array (data, mask=ww, fill_value=np.nan)
    # mata    = data
    mata    = mata.filled ( np.nan )

    nsamp   = mata.shape[2]
    # mask    = ww[0].sum (1) == 0.0
    ff_mask = ff_mask

    # axes
    tsamp   = float (pkg['duration']) / float ( nsamp )
    times   = np.linspace ( 0., float(pkg['duration']), nsamp )
    times   *= 1E3
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
    I      = I.sum (1)
    Q      = Q.sum (1)
    U      = U.sum (1)
    V      = V.sum (1)

    nON       = np.sqrt ( ons.stop - ons.start )
    if v:
        print (" Number of ON samples = {on:d}".format(on=ons.stop - ons.start))

    # 20230313 : use whole pulse region to compute the standard deviation
    # 20230313 : and multiply with sqrt ( width )

    I_err     = nON * I_std [ omask ]
    Q_err     = nON * Q_std [ omask ]
    U_err     = nON * U_std [ omask ]
    V_err     = nON * V_std [ omask ]
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

        rng          = np.random.default_rng()

        re_stat      = np.zeros ( n_resamples, dtype=np.float32 )
        pe_stat      = np.zeros ( n_resamples, dtype=np.float32 )

        # for i_resample in tqdm ( range(n_resamples), desc='Bootstrap', unit='bt' ):
        for i_resample in range ( n_resamples ):
            print ("   Bootstrap: {i_resample:03d} / {n_resamples:03d} ... ".format(i_resample=i_resample, n_resamples=n_resamples), end='\r')
            __i    = rng.choice ( nsamples, size=ntrial, replace=True, shuffle=False )
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

    1D RM fitting

    Phi = 0.5 * arctan ( U / Q )

    Phi, Phierr = 

    RM Lambda^2 + Psi
    """
    def __init__ (self, wave2, q, u, qerr, uerr, ierr, mpoints=128):
        """
        wave2: array
        stokesi: array

        data arrays are (frequency, )
        error arrays are (frequency,)
        """
        ###
        self.l2       = np.ma.MaskedArray ( (q**2) + (u**2), mask=np.zeros_like(u, dtype=bool) )
        ### Everett, Weisburg mask
        mask          = np.sqrt( self.l2 ) / ierr < 1.57
        ###
        self.w2       = np.ma.MaskedArray ( wave2.copy(), mask=mask )
        self.w2size   = wave2.size
        self.pa       = np.ma.MaskedArray ( 0.5 * np.arctan2 ( u, q ), mask=mask )
        self.paerr    = 0.5 * np.sqrt( (q*uerr)**2 + (qerr*u)**2 ) / self.l2
        ###
        self.w2min    = wave2.min()
        self.w2max    = wave2.max()
        self.mw2      = np.linspace ( self.w2min, self.w2max, mpoints, endpoint=True )

    def rmtf  ( self, rms, peak_rm ):
        """
        RM transfer function
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

        wpa  = np.unwrap ( pa, period=np.pi )

        slope, _ = np.polyfit ( rms, wpa, 1 )

        return {'rmtf_mag':mag, 'rmtf_pa':pa, 'rmtf_pa_slope':slope}

    def rm_spectrum (self, rms):
        """
        rm spectra?
        return magnitude?
        """
        nrms = rms.size

        ret  = np.zeros ((nrms,), dtype=np.complex64)

        for irm in range ( nrms ):
            _rm  = rms[irm]
            ret[irm] = np.sum ( np.exp ( 2.0j * ( self.pa - ( _rm * self.w2 ) ) ) )

        # return np.abs ( ret )
        return np.abs ( ret ), 0.5 * np.angle ( ret )

    def bootstrap_rmpa (self, rms, n_resamples=999, f_trial=0.90, confidence_level=0.95):
        """
        estimate RM error using bootstrap
        """
        boot   = RMPABootstrap ( self.w2, self.pa, rms )
        res, rm_boot, pa_boot    = boot (n_resamples=n_resamples, f_trial=f_trial, confidence_level=confidence_level)
        return res, rm_boot, pa_boot

    def pa_noise (self, rm_estimate, residual_power):
        """
        theoretically PA noise from Characteristic function of gaussian random distribution is 
        exp(-2sigma^2) at peak
        """
        max_mag  = np.sum ( np.exp ( 2.0j * ( self.pa - ( rm_estimate * self.w2 ) ) ) )
        max_mag  = np.abs ( max_mag ) / residual_power
        pa_sigma = np.sqrt ( -0.5 * np.log ( max_mag ) )
        return {'unbiased_paerr':pa_sigma}

    def estimate_pa0 (self, rm):
        """
        inverse variance weighted average

        PA error is QUADRATURE sum of PA RMS and PA weighted error
        """
        pa_freq = np.arctan ( np.tan ( self.pa - (rm * self.w2) ) )
        pa_err  = self.paerr

        pa_w    = np.power ( pa_err, -2.0 )

        pa_mean = np.sum ( pa_w * pa_freq ) / np.sum ( pa_w )
        pa_mean_err  = np.power ( np.sum ( pa_w ), -0.5 )
        pa_mean_rms  = np.sqrt ( np.mean ( np.power ( pa_freq - pa_mean, 2.0 ) ) )
        pa_std       = np.sqrt ( pa_mean_rms**2 + pa_mean_err**2 )

        pa_smean= np.mean ( pa_freq )

        return dict(pa_freq=pa_freq, pa_mean=pa_mean, pa_err=pa_std, pa_mean_simple=pa_smean)

    def model (self, rm, pa0, w2=None):
        """
        pa model
        """
        if w2 is None:
            w2 = self.w2
        return np.arctan ( np.tan ( pa0 + ( rm * w2 ) ) )

    def residual_pa (self, rm, pa0):
        """
        pa model
        """
        rpa = np.arctan ( np.tan ( self.pa - pa0 - ( rm * self.w2 ) ) )
        rpa_power = np.abs ( np.sum ( np.exp ( 2.0j * rpa ) ) )
        return rpa_power, rpa

    def chi2_reduced ( self, rm, pa0 ):
        """ chi2 reduced """
        model = self.model ( rm, pa0 )
        ye    = np.power ( self.paerr, 2 )
        # chi2  = np.sum ( np.power ( ( model - self.pa ), 2 ) / ye )
        chi2  = np.sum ( np.power ( ( model - self.pa ), 2 ))
        dof   = self.w2.size - 2
        return chi2 / dof

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

C      = 299.792458 # 1E6 * m / s 

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('rm_spec', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('ar_file', help="burst archive file")
    add ('-f','--fscrunch', default=4, type=int, help='Frequency downsample', dest='fs')
    add ('-n','--ntrials', default=999, type=int, help='Number of bootstrap trials', dest='ntrials')
    add ( '-r', '--rmgrid', help='RM grid (min:max:steps)', default='-200:-10:2048', dest='rmgrid' )
    add ( '--ton', help='Burst time window in bins (start:stop)', required=True, dest='ton' )
    add ( '--fon', help='Burst frequency window in bins (start:stop)', required=True, dest='fon' )
    # add ('--rmlow', help='Minimum RM in grid', dest='rmlow', default=-200, type=float)
    # add ('--rmhigh', help='Maximum RM in grid', dest='rmhigh',default=-10, type=float)
    # add ('--rmstep', help='Steps in RM grid', dest='rmgrid', default=2048, type=int)
    # add ('--freqlow', help='Lowest frequency in MHz', dest='flow', default=550, type=float)
    # add ('--freqhigh', help='Highest frequency in MHz', dest='fhigh',default=750, type=float)
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
    try:
        _tstart, _tstop = args.ton.split(':')
        _fstart, _fstop = args.fon.split(':')
        on_region['tstart']    = int ( _tstart )
        on_region['tstop']     = int ( _tstop )
        on_region['fstart']    = int ( _fstart )
        on_region['fstop']     = int ( _fstop )
    except:
        raise RuntimeError(" ON region not understood ton={ton} fon={fon}".format(ton=args.ton, fon=args.fon))
    ####################################
    try:
        rmlow, rmhigh, rmsteps = args.rmgrid.split(':')
        rmlow  = float ( rmlow )
        rmhigh = float ( rmhigh )
        rmsteps= int ( rmsteps )
    except:
        raise RuntimeError (" RM grid not understood, {rmgrid}".format(rmgrid=args.rmgrid))
    if args.v:
        print (" RM Grid = {rmlow:.3f} ... {rmhigh:.3f} with {rmsteps:d} steps".format(rmlow=rmlow, rmhigh=rmhigh, rmsteps=rmsteps))
    rm_grid   = np.linspace ( rmlow, rmhigh, rmsteps, endpoint=True )
    ####################################
    pkg       = read_pkg ( args.ar_file )
    data_freq_low    = np.min ( pkg['freqs'] )
    data_freq_high   = np.max ( pkg['freqs'] )
    ####################################
    freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err = read_prepare_tscrunch (
        pkg,
        on_region,
        args.fs,
        args.v
    )

    ## compute lambdas
    lam2      = np.power ( C / freq_list, 2 )

    RET     = dict ()
    CET     = dict ()
    RET['filename'] = bn
    RET['lam2'] = lam2
    RET['fs']   = args.fs
    CET['filename'] = bn
    CET['fs']   = args.fs

    if args.v:
        print (" Calling PASpec fitting ... ")

    ### do the actual call
    paspec    = PASpec ( lam2, Q, U, Q_err, U_err, I_err )

    ################################
    ### compute magnitude spectrum
    rmspec, pagridspec  = paspec.rm_spectrum ( rm_grid ) 

    peak_rm   = rm_grid [ np.argmax ( rmspec ) ]

    ### impulse RMTF
    rmtf      = paspec.rmtf ( rm_grid, peak_rm )

    ### fit rm 
    fitrm, rm_boot, pa_boot   = paspec.bootstrap_rmpa ( rm_grid, n_resamples=args.ntrials )

    ### get model
    model     = paspec.model ( fitrm['rm'], fitrm['pa'] )
    m_model   = paspec.model ( fitrm['rm'], fitrm['pa'], paspec.mw2 )
    # rpa0      = np.arctan ( np.tan ( paspec.pa - model ) )
    rpa_power, rpa     = paspec.residual_pa ( fitrm['rm'], fitrm['pa'] )

    ### unbiased PA noise
    # unbiased_paerr   = paspec.pa_noise ( fitrm['rm'], rpa_power )

    ### compute reduced CHI2
    rchi2     = paspec.chi2_reduced ( fitrm['rm'], fitrm['pa'])

    # ut    = f"RM-ML={fitrm['rm']:.3f}+-{fitrm['rm_se']:.3f}\nPA0={np.rad2deg(fitrm['pa']):.3f}+-{np.rad2deg(fitrm['pa_se']):.3f}\nrCHI2={rchi2:.3f}"
    ut    = "RM-ML={rm:.3f}+-{rmse:.3f}\nPA0={pa:.3f}+-{pase:.3f}\nrCHI2={rchi2:.3f}".format ( 
        rm = fitrm['rm'], rmse = fitrm['rm_se'],
        pa = np.rad2deg(fitrm['pa']), pase = np.rad2deg(fitrm['pa_se']),
        rchi2 = rchi2
    )

    if args.v:
        print ( ut )
        print (" done")

    RET.update ( fitrm )
    RET['w2']     = paspec.w2
    RET['pa']     = paspec.pa
    RET['paerr']  = paspec.paerr
    RET['res_pa'] = rpa
    RET['rmgrid'] = rm_grid
    RET['rmspec'] = rmspec
    RET['paspec'] = pagridspec
    RET.update ( rmtf )
    CET.update ( fitrm )
    CET['rmtf_pa_slope'] = rmtf['rmtf_pa_slope']
    RET['boot_rm'] = rm_boot
    RET['boot_pa'] = pa_boot
    ###########################################################
    cf   = pd.DataFrame ( CET, index=[0] )

    ###########################################################
    fig = plt.figure ('paspec', figsize=(9,5))

    gs  = mgs.GridSpec ( 3, 3, figure=fig )

    axpa = fig.add_subplot ( gs[1,:] )
    axrs = fig.add_subplot ( gs[2,:], sharex=axpa )

    axgg = fig.add_subplot ( gs[0, 2] )
    axph = fig.add_subplot ( gs[0, 1] )
    axrh = fig.add_subplot ( gs[0, 0] )

    axpa.errorbar ( paspec.w2, np.rad2deg( paspec.pa ), yerr=np.rad2deg( paspec.paerr ), marker='.', c='k', capsize=5, ls='' )
    axpa.plot ( paspec.mw2, np.rad2deg( m_model ), c='b' )

    axrs.plot ( paspec.w2, np.rad2deg( rpa ), marker='.', c='b' )
    axrs.axhline (0., ls=':', c='k', alpha=0.4 )

    # axgg.scatter ( rm_grid, rmspec, marker='.', c='k' )
    # axgg.plot ( rm_grid, rmspec_model, c='b' )
    axgg.scatter ( rm_grid, rmspec / rpa_power, marker='.', c='k' )
    # axgg.plot ( rm_grid, rmspec_model / rpa_power, c='b' )
    axgg.axvline ( fitrm['rm'], ls='--', c='b' )

    axgg.set_xlabel ('RM / rad m$^{-2}$')
    axgg.set_ylabel ('mag')
    axpa.set_ylabel ('PA / deg')
    axrs.set_ylabel ('res-PA / deg')
    axrs.set_xlabel ('Wavelength$^{2}$ / m$^{2}$')

    to_freq = lambda wav : (C / wav**0.5)
    from_freq = lambda freq: (C / freq)**2
    faxpa= axpa.secondary_xaxis ('top', functions=(to_freq, from_freq))
    faxpa.set_xlabel('Freq / MHz')

    axpa.set_xlim ( from_freq(data_freq_high), from_freq(data_freq_low) )
    axpa.set_ylim ( -90., 90. )
    axrs.set_ylim (-30, 30)

    axrh.hist ( rm_boot, bins='auto', density=True, color='blue' )
    axrh.axvline ( fitrm['rm_mean'], ls=':', c='k', alpha=0.75 )
    axph.hist ( np.rad2deg(pa_boot), bins='auto', density=True, color='blue' )
    axph.axvline ( np.rad2deg(fitrm['pa_mean']), ls=':', c='k', alpha=0.75 )

    # axrh.set_xlabel (f"RM = {fitrm['rm_mean']:.3f}")
    # axph.set_xlabel (f"PA = {np.rad2deg(fitrm['pa_mean']):.3f}")
    axrh.set_xlabel ("RM = {rm:.3f}".format(rm=fitrm['rm_mean']))
    axph.set_xlabel ("PA = {pa:.3f}".format(pa=np.rad2deg(fitrm['pa_mean'])))

    for _ax in [axgg, axph, axrh]:
        _ax.xaxis.tick_top ()
        _ax.xaxis.set_label_position('top')
        _ax.yaxis.tick_right ()
        _ax.yaxis.set_label_position('right')

    ###########################################################
    # plt.show ()
    fig.savefig ( os.path.join ( args.odir, bn + ".png" ), dpi=300, bbox_inches='tight' )
    cf.to_csv ( os.path.join ( args.odir, bn + "_spec.csv" ), index=False )
    np.savez ( os.path.join ( args.odir, bn + "_spec.npz"), **RET)



