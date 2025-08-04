"""
Makes a pacv file to be used in calibration
with polarization expressed in circular basis

extension of my_pacv

- designed for circular feeds
- corrects for ionospheric RM contribution in deriving calibration solution.
- corrects for parallactic angle and position angle of the source

ignore the ionospheric RM contribution
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

################################
RAD,DECD         = dict(),dict()
RAD['3C138']     = 79.5687917
DECD['3C138']    = 16.5907806
RAD['R3']        = 29.50312583
DECD['R3']       = 65.71675422
RAD['3C48']      = 24.4220417
DECD['3C48']     = 33.1597417

C                = 299792458.0 # m/s

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("make_pacv_circ", description="Makes a pacv calibration solution in circular basis", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('npz_file', help='npz file (output of make_pkg.py)',)
    add ('-z','--zap', help='Zap the channels (comma-separated, start:stop)', dest='zap', default='')
    add ('--on', help='ON region in bins (comma-separated, start:stop)', dest='on_region',)
    add ('--off', help='OFF region in bins (comma-separated, start:stop)', dest='off_region',)
    add ('-O', '--outdir', help='Output directory', default="./", dest='odir')
    add ('-v','--verbose', action='store_true', dest='v')
    # add ('-a','--par-angle', help='Parallactic angle in degrees', dest='pangle', default=None, type=float)
    # add ('-i','--ionos-rm', help='Ionospheric RM', dest='ionosrm', default=None, type=float)
    add ('-n','--noise-diode', help='Noise diode', action='store_true', dest='noise_diode')
    add ('--delays_grid', help='Delays grid (min:max:steps)', dest='delays_grid', default="0:600:2048")
    return agp.parse_args ()

class BasePacvInfo(object):
    """
    One class containing info which goes into 
    pacv calibration solution file
    """
    def __init__(self, mjd):
        self.file_date     = self.__format_date__  (at.Time.now().isot)
        self.observer      = "LGM"
        self.proj_id       = "GMRT-FRB"
        self.obs_date      = ""

        #### freq info
        self.freqs         = None
        self.fcenter       = 0.0
        self.bw            = 0.0
        self.nchan         = 0
        self.chan_bw       = 0.0

        #### source info
        self.sc            = None
        self.src_name      = ""
        self.ra_str        = "00:00:00"
        self.dec_str       = "+00:00:00"

        #### beam info
        self.bmaj_deg      = 0.0
        self.bmin_deg      = 0.0
        self.bpa_deg       = 0.0

        #### data info
        self.scan_len      = 0
        self.tsamp         = 0.0
        self.nsuboffs      = 0.0
        self.nsblk         = 0

        ## taken from observatories.dat from tempo2
        ## 1656342.30    5797947.77      2073243.16       GMRT                gmrt
        self.telescope     = "GMRT"
        self.ant_x         = 1656342.30
        self.ant_y         = 5797947.77
        self.ant_z         = 2073243.16
        self.el            = asc.EarthLocation.from_geocentric (self.ant_x, self.ant_y, self.ant_z, unit="m")
        self.longitude     = self.el.lon.degree

        self.npoln         = 4
        self.poln_order    = "AABBCRCI"

        ## mjd
        ### 
        self.start_time   = at.Time (mjd, format='mjd')
        self.stt_imjd     = int(mjd)
        stt_smjd          = (mjd - self.stt_imjd) * 24 * 3600
        self.stt_smjd     = int (stt_smjd)
        self.stt_offs     = stt_smjd - self.stt_smjd
        self.obs_date     = self.__format_date__ (self.start_time.isot)
        ### LST
        self.stt_lst      = self.__get_lst__ (mjd, self.longitude)

    def fill_freq_info(self, nchans, bandwidth, freqs):
        """ uGMRT gives nchans, bandwidth and either flow or fhigh 
            All frequency units in MHz

            psrfits requires centre frequency of each channel

        """
        self.bw           = bandwidth
        self.nchan        = nchans
        self.chan_bw      = bandwidth / nchans
        self.freqs        = freqs
        self.fcenter      = self.freqs[nchans//2]

    def fill_source_info(self, src_name, rad, decd):
        """ loads src_name, RA/DEC string """
        self.sc       = asc.SkyCoord(rad, decd, unit="deg")
        ra_hms        = self.sc.ra.hms
        dec_dms       = self.sc.dec.dms
        self.src_name = src_name
        self.ra_str   = f"{int(ra_hms[0]):02d}:{np.abs(int(ra_hms[1])):02d}:{np.abs(ra_hms[2]):07.4f}"
        self.dec_str  = f"{int(dec_dms[0]):02d}:{np.abs(int(dec_dms[1])):02d}:{np.abs(dec_dms[2]):07.4f}"

    def get_parallactic_angle (self, tobs, noise_diode=False):
        """ compute parallactic angle 

        from astroplan:Observer class
        """
        if noise_diode:
            return 0.0
        ############################
        lst   = tobs.sidereal_time ( 'mean', longitude=self.el.lon, model=None )
        h     = (lst - self.sc.ra).radian
        q     = np.arctan2 ( 
                np.sin ( h ), 
                np.tan ( self.el.lat.radian ) * np.cos ( self.sc.dec.radian ) - 
                np.sin ( self.sc.dec.radian ) * np.cos ( h )
        )
        ############################
        return q

    def get_position_angle (self, noise_diode=False):
        """ compute position angle of the source
            
            Perley Butler here
            in band-4, we do not expect pos-angle to change

           -32 degrees

           This Q/U sign flip
           There is a sign flip in V

           also returns the reference frequency

           all this thanks uGMRT Band-4 imaging polcal
        """
        if noise_diode:
            ## probably noise diode scan
            return np.deg2rad ( 0 ), self.freqs[0]
        elif self.src_name == "3C138":
            return np.deg2rad ( -32 ), self.freqs[0]
        else:
            raise RuntimeError ("Source not identified src=",self.src_name)

    def get_rotation_measure (self, noise_diode=False):
        """ 
        Does 3C138 have RM?

        Tabara&Inoue say -2.1
        https://ui.adsabs.harvard.edu/abs/1980A%26AS...39..379T/abstract

        VLBI studies say RM is equivalent to zero
        https://ui.adsabs.harvard.edu/abs/1995A%26A...299..671D/abstract
        https://ui.adsabs.harvard.edu/abs/1997A&A...325..493C

        VLA just says take RM to be zero.
            
        """
        if noise_diode:
            return 0.0
        elif self.src_name == "3C138":
            return -2.1
        else:
            raise RuntimeError ("Source not identified src=",self.src_name)

    def fill_beam_info(self, beam_size):
        """ currently only support circular beams """
        self.bmaj_deg  = beam_size / 3600.0
        self.bmin_deg  = beam_size / 3600.0
        self.bpa_deg   = 0.0

    def __get_lst__ (self, mjd, longitude):
        ## magic numbers
        gfac0    = 6.697374558
        gfac1    = 0.06570982441908
        gfac2    = 1.00273790935
        gfac3    = 0.000026
        mjd0     = 51544.5  # MJD at 2000 Jan 01 12h
        ##
        H        = (mjd - int(mjd)) * 24  # Hours since previous 0h
        D        = mjd - mjd0  # Days since MJD0
        D0       = int(mjd) - mjd0  # Days between MJD0 and prev 0h
        T        = D / 36525.0  # Number of centuries since MJD0
        ##
        gmst     = gfac0 + gfac1 * D0 + gfac2 * H + gfac3 * T ** 2.0
        lst      = ((gmst + longitude / 15.0) % 24.0) * 3600.0
        return lst

    def __format_date__ (self, date_str):
        # Strip out the decimal seconds
        out_str = date_str.split(".")[0]
        return out_str

    def fill_primary_header(self, feed, hand, chan_dm=0.0, scan_len=0):
        """
        Writes the primary HDU

        Need to check:
        XXX
            - FD_SANG, FD_XYPH, BE_PHASE, BE_DCC
            - beam info: BMAJ, BMIN, BPA
            - if CALIBRATION
        """
        if feed not in ['CIRC','LIN']:
            raise RuntimeError (f"Feed={feed} not understood")
        if hand not in [+1, -1]:
            raise RuntimeError (f"Hand={hand} not understood")
        # XXX need to check
        p_hdr = fits.Header()
        p_hdr["HDRVER"] = (
            "6.2             ",
            "Header version                               ",
        )
        p_hdr["FITSTYPE"] = ("PSRFITS", "FITS definition for pulsar data files        ")
        p_hdr["DATE"] = (
            self.file_date,
            "File creation date (YYYY-MM-DDThh:mm:ss UTC) ",
        )
        p_hdr["OBSERVER"] = (
            self.observer,
            "Observer name(s)                             ",
        )
        p_hdr["PROJID"] = (
            self.proj_id,
            "Project name                                 ",
        )
        p_hdr["TELESCOP"] = (
            self.telescope,
            "Telescope name                               ",
        )
        p_hdr["ANT_X"] = (self.ant_x, "[m] Antenna ITRF X-coordinate (D)            ")
        p_hdr["ANT_Y"] = (self.ant_y, "[m] Antenna ITRF Y-coordinate (D)            ")
        p_hdr["ANT_Z"] = (self.ant_z, "[m] Antenna ITRF Z-coordinate (D)            ")
        p_hdr["FRONTEND"] = (
            "GWB",
            "Rx and feed ID                               ",
        )
        p_hdr["NRCVR"] = (2, "Number of receiver polarisation channels     ")
        p_hdr["FD_POLN"] = (feed, "LIN or CIRC                                  ")
        p_hdr["FD_HAND"] = (hand, "+/- 1. +1 is LIN:A=X,B=Y, CIRC:A=L,B=R (I)   ")

        ### XXX
        """
            WvS+?? psrchive+polcal paper says FD_SANG for circular feeds should be 0deg
            FD_HAND=+1 for circular feeds
        """
        p_hdr["FD_SANG"] = (0.0, "[deg] FA of E vect for equal sigma in A&B (E)  ")
        p_hdr["FD_XYPH"] = (0.0, "[deg] Phase of A^* B for injected cal (E)    ")

        p_hdr["BACKEND"]  = ("uGMRT", "Backend ID                                   ")
        p_hdr["BECONFIG"] = ("N/A", "Backend configuration file name              ")
        ### XXX
        ## BE_PHASE affects StokesV so check
        ## XXX all usb's so it should be +ive???
        p_hdr["BE_PHASE"] = (+1, "0/+1/-1 BE cross-phase:0 unknown,+/-1 std/rev")
        ## in some uGMRT bands, the top subband is taken and in some the lower subband is
        p_hdr["BE_DCC"]   = (0, "0/1 BE downconversion conjugation corrected  ")

        p_hdr["BE_DELAY"] = (0.0, "[s] Backend propn delay from digitiser input ")
        p_hdr["TCYCLE"]   = (0.0, "[s] On-line cycle time (D)                   ")

        ### PSR mode
        p_hdr["OBS_MODE"] = ("PCM", "(PSR, CAL, SEARCH)                           ")
        p_hdr["DATE-OBS"] = (
            self.obs_date,
            "Date of observation (YYYY-MM-DDThh:mm:ss UTC)",
        )

        #### freq info
        p_hdr["OBSFREQ"] = (
            self.fcenter,
            "[MHz] Centre frequency for observation       ",
        )
        p_hdr["OBSBW"] = (self.bw, "[MHz] Bandwidth for observation              ")
        p_hdr["OBSNCHAN"] = (
            self.nchan,
            "Number of frequency channels (original)      ",
        )
        p_hdr["CHAN_DM"] = ("*", "DM used to de-disperse each channel (pc/cm^3)")

        ### beam info
        p_hdr["BMAJ"] = (self.bmaj_deg, "[deg] Beam major axis length                 ")
        p_hdr["BMIN"] = (self.bmin_deg, "[deg] Beam minor axis length                 ")
        p_hdr["BPA"]  = (self.bpa_deg, "[deg] Beam position angle                    ")

        ## source info
        p_hdr["SRC_NAME"] = (
            self.src_name,
            "Source or scan ID                            ",
        )
        p_hdr["COORD_MD"] = ("J2000", "Coordinate mode (J2000, GAL, ECLIP, etc.)    ")
        p_hdr["EQUINOX"]  = (2000.0, "Equinox of coords (e.g. 2000.0)              ")
        p_hdr["RA"]       = (self.ra_str, "Right ascension (hh:mm:ss.ssss)              ")
        p_hdr["DEC"]      = (self.dec_str, "Declination (-dd:mm:ss.sss)                  ")
        p_hdr["STT_CRD1"] = (
            self.ra_str,
            "Start coord 1 (hh:mm:ss.sss or ddd.ddd)      ",
        )
        p_hdr["STT_CRD2"] = (
            self.dec_str,
            "Start coord 2 (-dd:mm:ss.sss or -dd.ddd)     ",
        )
        p_hdr["TRK_MODE"] = ("TRACK", "Track mode (TRACK, SCANGC, SCANLAT)          ")
        p_hdr["STP_CRD1"] = (
            self.ra_str,
            "Stop coord 1 (hh:mm:ss.sss or ddd.ddd)       ",
        )
        p_hdr["STP_CRD2"] = (
            self.dec_str,
            "Stop coord 2 (-dd:mm:ss.sss or -dd.ddd)      ",
        )
        p_hdr["SCANLEN"] = (
            scan_len,
            "[s] Requested scan length (E)                ",
        )
        ### it is FA for uGMRT
        ### CPA is super cool
        p_hdr["FD_MODE"] = ("FA", "Feed track mode - FA, CPA, SPA, TPA          ")
        p_hdr["FA_REQ"]  = (0.0, "[deg] Feed/Posn angle requested (E)          ")
        
        ### calibration 
        p_hdr["CAL_MODE"] = ("N/A", "Cal mode (OFF, SYNC, EXT1, EXT2)             ")
        p_hdr["CAL_FREQ"] = (-1.0,  "[Hz] Cal modulation frequency (E)            ")
        p_hdr["CAL_DCYC"] = (1.0,   "Cal duty cycle (E)                           ")
        p_hdr["CAL_PHS"]  = (-1.0,  "Cal phase (wrt start time) (E)               ")

        ### dates
        p_hdr["STT_IMJD"] = (
            self.stt_imjd,
            "Start MJD (UTC days) (J - long integer)      ",
        )
        p_hdr["STT_SMJD"] = (
            self.stt_smjd,
            "[s] Start time (sec past UTC 00h) (J)        ",
        )
        p_hdr["STT_OFFS"] = (
            self.stt_offs,
            "[s] Start time offset (D)                    ",
        )
        p_hdr["STT_LST"] = (
            self.stt_lst,
            "[s] Start LST (D)                            ",
        )
        return p_hdr

    def fill_history_table (self):
        """
        its an empty header
        its a  dummy table
        """
        t_hdr = fits.Header()
        poc_columns = [ 
            fits.Column(name="DATE_PRO",  format="24A", array=[[self.file_date]]),
            fits.Column(name="PROC_CMD",  format="256A", array=[["UNKNOWN"]]),
            fits.Column(name="SCALE",     format="8A", array=[["FluxDen"]]),
            fits.Column(name="POL_TYPE",  format="8A", array=[["AABBCRCI"]]),

            fits.Column(name="NSUB",      format="1J",  array=[[0]]),
            fits.Column(name="NPOL",      format="1I",  array=[[4]]),
            fits.Column(name="NBIN",      format="1I",  array=[[512]]),
            fits.Column(name="NBIN_PRD",  format="1I",  array=[[512]]),
            fits.Column(name="TBIN",      format="1D",  unit="s", array=[[self.tsamp]]),
            fits.Column(name="CTR_FREQ",  format="1D",  unit="MHz", array=[[self.fcenter]]),
            fits.Column(name="NCHAN",      format="1J",  array=[[self.nchan]]),
            fits.Column(name="CHAN_BW",    format="1D",  unit="MHz", array=[[self.chan_bw]]),
            fits.Column(name="REF_FREQ",  format="1D",  unit="MHz", array=[[self.fcenter]]),

            fits.Column(name="DM",        format="1D",  unit="", array=[[0.]]),
            fits.Column(name="RM",        format="1D",  unit="", array=[[0.]]),

            fits.Column(name="PR_CORR",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="FD_CORR",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="BE_CORR",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="RM_CORR",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="DEDISP",    format="1I",  unit="", array=[[0]]),

            fits.Column(name="DDS_MTHD",    format="32A",  unit="", array=[["UNSET"]]),
            fits.Column(name="SC_MTHD",     format="32A",  unit="", array=[["NONE"]]),
            fits.Column(name="CAL_MTHD",    format="32A",  unit="", array=[["NONE"]]),
            fits.Column(name="RFI_MTHD",    format="32A",  unit="", array=[["NONE"]]),

            fits.Column(name="CAL_FILE",    format="256A",  unit="", array=[["NONE"]]),

            fits.Column(name="RM_MODEL",    format="32A",  unit="", array=[["NONE"]]),
            fits.Column(name="AUX_RM_C",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="DM_MODEL",    format="32A",  unit="", array=[["NONE"]]),
            fits.Column(name="AUX_DM_C",   format="1I",  unit="", array=[[0]]),
        ]

        # Add the columns to the table
        polyco_hdu = fits.BinTableHDU(
            fits.FITS_rec.from_columns(poc_columns), name="history", header=t_hdr
        )
        return polyco_hdu

    def fill_solution_header(self, delay_pi, bias):
        """
        Put solution

        delay_pi and bias are the calibration solution

        SINGLE cross coupling method
        """
        t_hdr = fits.Header()
        t_hdr['CAL_MTHD']    = ("single", "Cross coupling method")
        t_hdr['NCPAR']    = (3, "Number of coupling parameters")
        t_hdr['NCOVAR']   = (0, "Number of parameter covariances")
        t_hdr['NCHAN']    = (self.nchan, "Nr of channels in Feed coupling data")
        t_hdr['EPOCH']    = (self.start_time.mjd, "[MJD] Epoch of calibration obs")

        t_hdr["PAR_0000"]     = ("G", "scalar gain")
        t_hdr["PAR_0001"]     = ("gamma", "differential gain (hyperbolic radians)")
        t_hdr["PAR_0002"]     = ("phi", "differential phase (radians)")

        ## record keeping parameter
        t_hdr["DELAY_PI"]     = (delay_pi, "DELAY_PI term in the mypacv generation")
        t_hdr["BIAS"]         = (bias, "BIAS term in the mypacv generation")
        return t_hdr

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
    FILE_PKG    = args.npz_file
    base,_      = os.path.splitext ( os.path.basename ( FILE_PKG ) )
    pfile       = os.path.join ( args.odir, base + ".pcal.png" )
    ofile       = base + ".pcal.pacv"
    outfile   = os.path.join ( args.odir, ofile  )
    ###################################################
    ### read calibrator file
    ###################################################
    ## read
    with open ( FILE_PKG, 'rb' ) as __f:
        pkg         = np.load ( __f )
        freqs_mhz   = pkg['freqs']
        mata        = np.ma.MaskedArray ( pkg['data'], mask=pkg['wts'] )[0]
        mjd         = pkg['mjd']
        source      = pkg['src'].item().decode('utf-8')
        nchan       = freqs_mhz.shape[0]
        fbw         = freqs_mhz[1] - freqs_mhz[0]
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
    ### prepare pacv file
    ###################################################
    feed      = 'CIRC'
    hand      = 1 # fixed to one because feed is circular
    tobs      = at.Time ( mjd, format='mjd' )
    dt        = tobs.strftime ("%Y%m%d")
    #imjd       = int ( mjd )
    pinfo     = BasePacvInfo ( mjd )
    pinfo.fill_freq_info ( nchan, fbw, freqs_mhz )
    ##
    pinfo.fill_source_info ( source, RAD[source], DECD[source] )
    pinfo.fill_beam_info ( 0. )
    #### parallactic angle
    #### position angle
    pal_angle = pinfo.get_parallactic_angle ( tobs, noise_diode = args.noise_diode )
    pos_angle, pal_freq   = pinfo.get_position_angle ( noise_diode = args.noise_diode )
    #### source RM
    srcrm       = pinfo.get_rotation_measure ( noise_diode = args.noise_diode ) 
    #### correct for both
    angle_corr  = pal_angle + pos_angle
    rm_corr     = srcrm
    #### logging
    print (f" Parallactic angle = {np.rad2deg(pal_angle):.3f}")
    print (f" Position angle    = {np.rad2deg(pos_angle):.3f}")
    print (f" Correction angle  = {np.rad2deg(angle_corr):.3f}")
    print (f" Source RM         = {srcrm:.3f}")
    pa_corr     = angle_corr + (rm_corr * wav2)
    ##################################################
    ### math
    ###################################################
    ii,qq,uu,vv = oo
    data_pa     = 0.5 * np.arctan2 ( uu, qq )
    ######### PA correction
    data_pa     = np.arctan ( np.tan ( data_pa - pa_corr ) )
    ######### PA correction
    aa          = 0.5 * ( ii + vv )
    bb          = 0.5 * ( ii - vv )
    #### error propagation
    ierr,qerr,uerr,verr = ooerr
    _aberr      = np.sqrt ( ierr**2 + verr**2 )
    dab         = (_aberr/aa) + (_aberr/bb)
    ### gain dgain
    gain        = np.sqrt ( 2.0 * np.sqrt ( aa * bb ) )
    gainerr     = 0.25 * gain * dab

    dgain       = 0.25 * np.log ( aa / bb )
    dgainerr    = 0.25 * dab 
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
    ### dphase error is standard deviation of error_pa
    dphaseerr   = np.zeros_like ( dphase ) + np.std ( error_pa )
    ##################################################
    st        = f" estimated cable delay = {delay_ns:.3f} ns | bias_rad = {bias_rad:.3f} rad"
    print ( st )
    ### make diagnostic plot
    ###################################################
    fig       = plt.figure ('pacv_circ')
    gs        = mgs.GridSpec ( 3, 2, figure=fig )
    axpp      = fig.add_subplot ( gs[0,0] )
    axdd      = fig.add_subplot ( gs[0,1] )
    axpa      = fig.add_subplot ( gs[1,:] )
    axep      = fig.add_subplot ( gs[2,:], sharex=axpa )
    ################
    axdd.plot ( delays_grid, dmags, c='b', marker='.' )
    axdd.axvline ( delay_ns, ls=':', c='k' )
    axdd.set_xlabel ('Delays / ns')
    axdd.set_ylabel ('Magnitude')

    __bins    = np.arange ( mata.shape[2] )
    axpp.plot ( __bins, pp, c='k' )
    _ylow, _yhigh = axpp.get_ylim ()
    axpp.vlines ( np.where(onmask)[0], color='g', alpha=0.20, ymin=_ylow, ymax=_yhigh  )
    axpp.vlines ( np.where(ofmask)[0], color='r', alpha=0.20, ymin=_ylow, ymax=_yhigh  )
    axpp.set_xlabel ('Bin')
    axpp.set_ylabel ('I')

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
    ##################################################
    ### write into pacv
    ###################################################
    npar      = 3
    dat_freq  = np.array (freqs_mhz, dtype=np.float32).reshape ((1, nchan))
    ## need the logical not for dat_wts
    mask      = np.logical_not ( fqmask )
    dat_wts   = np.array (mask, dtype=np.float32).reshape ((1, nchan))
    data      = np.zeros ((1, nchan, npar), dtype=np.float32)
    dataerr   = np.zeros ((1, nchan, npar), dtype=np.float32)
    #####
    data[0, ..., 0] = gain
    data[0, ..., 1] = dgain
    data[0, ..., 2] = dphase
    dataerr[0, ..., 0]  = gainerr
    dataerr[0, ..., 1]  = dgainerr
    dataerr[0, ..., 2]  = dphaseerr
    ##################################################
    ### pacv fits header
    ###################################################
    primary_header  = pinfo.fill_primary_header ( feed, hand )
    primary_hdu     = fits.PrimaryHDU (header=primary_header)
    history_hdu     = pinfo.fill_history_table ()

    calsol_header   = pinfo.fill_solution_header ( delay_ns, bias_rad )
    ## XXX note that there is no ``hin'' operation happening here
    calsol_columns  = [
        fits.Column(name="DAT_FREQ", format=f"{nchan:d}D", unit="MHz", array=dat_freq),
        fits.Column(name="DAT_WTS",  format=f"{nchan:d}E", array=dat_wts),
        fits.Column(name="DATA",     format=f"{nchan*npar:d}E", array=data),
        fits.Column(name="DATAERR",  format=f"{nchan*npar:d}E", array=dataerr),
    ]
    calsol_hdu      = fits.BinTableHDU(
        fits.FITS_rec.from_columns(calsol_columns), name="feedpar", header=calsol_header
    )
    #################################
    fits_data       = fits.HDUList ()
    fits_data.append ( primary_hdu )
    fits_data.append ( history_hdu )
    fits_data.append ( calsol_hdu )
    #################################
    fits_data.writeto(outfile, overwrite=True)



