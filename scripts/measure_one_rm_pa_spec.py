"""


"""
from __future__ import print_function
import os
import sys
import json

import numpy as np

# from tqdm import tqdm

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.colors as mc

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
        self.w2    = w2 
        self.pa    = paw2 
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

    1D RM fitting

    Phi = 0.5 * arctan ( U / Q )

    Phi, Phierr = 

    RM Lambda^2 + Psi
    """
    def __init__ (self, wave2, pa, mpoints=128):
        """
        wave2: array
        stokesi: array

        data arrays are (frequency, )
        error arrays are (frequency,)
        """
        ###
        self.w2       = wave2.copy()
        self.w2size   = wave2.size
        self.pa       = pa.copy ()
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

        # wpa  = np.unwrap ( pa, period=np.pi )
        wpa  = np.unwrap ( pa, discont=0.5*np.pi )

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
        # ye    = np.power ( self.paerr, 2 )
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
    ag   = agp.ArgumentParser ('rm_spec', epilog='Part of GMRT/FRB polarization pipeline')
    add  = ag.add_argument
    add ('onerm', help="onerm npz output of :prepare_onerm:", nargs='+')
    add ('-k','--snr', default=3, type=int, help='std cutoff', dest='std_cutoff')
    add ('-b','--bn', required=True, help='onerm file name, usually MJD', dest='bn')
    add ('-n','--ntrials', default=999, type=int, help='Number of bootstrap trials', dest='ntrials')
    add ( '-r', '--rmgrid', help='RM grid (min:max:steps)', default='-200:-10:4096', dest='rmgrid' )
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='./', dest='odir')
    ##
    return ag.parse_args ()

if __name__ == "__main__":
    args    = get_args ()
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
    wave2     = []
    pa        = []
    for onerm in args.onerm:
        with np.load ( onerm ) as frm:
            mask     = np.sqrt ( frm['Q']**2 + frm['U']**2 ) >= (args.std_cutoff * frm['I_err'])
            pa.append    (frm['PA'][mask])
            wave2.append (frm['lam2'][mask])
    pa        = np.concatenate ( pa )
    wave2     = np.concatenate ( wave2 )
    ####################################
    data_freq_low    = 550.
    data_freq_high   = 750.
    ####################################
    ### do the actual call
    paspec    = PASpec ( wave2, pa )
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

    RET  = dict()
    CET     = dict ()
    RET.update ( fitrm )
    CET.update ( fitrm )
    RET['w2']     = paspec.w2
    RET['pa']     = paspec.pa
    #RET['paerr']  = paspec.paerr
    RET['res_pa'] = rpa
    RET['rmgrid'] = rm_grid
    RET['rmspec'] = rmspec
    RET['paspec'] = pagridspec
    RET.update ( rmtf )
    RET['boot_rm'] = rm_boot
    RET['boot_pa'] = pa_boot
    CET['rmtf_pa_slope'] = rmtf['rmtf_pa_slope']
    ###########################################################
    fig = plt.figure ('paspec', figsize=(9,5))
    """
    pa(lambda2)
    error_pa(lambda2)
    """

    axpa, axrs = fig.subplots ( 2, 1, sharex=True )


    axpa.scatter ( paspec.w2, np.rad2deg( paspec.pa ), marker='.', c='k', )
    axpa.plot ( paspec.mw2, np.rad2deg( m_model ), c='b' )

    axrs.scatter ( paspec.w2, np.rad2deg( rpa ), marker='.', c='b' )
    axrs.axhline (0., ls=':', c='k', alpha=0.4 )

    # axgg.scatter ( rm_grid, rmspec, marker='.', c='k' )
    # axgg.plot ( rm_grid, rmspec_model, c='b' )
    # axgg.plot ( rm_grid, rmspec_model / rpa_power, c='b' )

    axpa.set_ylabel ('PA / deg')
    axrs.set_ylabel ('res-PA / deg')
    axrs.set_xlabel ('Wavelength$^{2}$ / m$^{2}$')

    to_freq = lambda wav : (C / wav**0.5)
    from_freq = lambda freq: (C / freq)**2
    # faxpa= axpa.secondary_xaxis ('top', functions=(to_freq, from_freq))
    # faxpa.set_xlabel('Freq / MHz')

    axpa.set_xlim ( from_freq(data_freq_high), from_freq(data_freq_low) )
    axpa.set_ylim ( -90., 90. )
    #axrs.set_ylim (-30, 30)
    axrs.set_ylim (-90, 90)

    ###########################################################
    #plt.show ()
    fig.suptitle ( ut )
    fig.savefig ( os.path.join ( args.odir, args.bn + ".png" ), dpi=300, bbox_inches='tight' )
    write_csv ( CET, os.path.join ( args.odir, args.bn + "_measureonerm.csv" ) )
    # cf.to_csv ( os.path.join ( args.odir, bn + "_spec.csv" ), index=False )
    for k,v in RET.items():
        if np.ma.isMaskedArray ( v ): RET[k] = np.array ( v.filled(np.nan) )
    np.savez ( os.path.join ( args.odir, args.bn + "_measureonerm.npz"), **RET)




