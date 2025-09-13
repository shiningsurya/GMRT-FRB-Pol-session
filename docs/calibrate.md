
# Calibration

We are given two calibrator archives.
```
data/cals/3C138_bm1_pa_550_200_32_29jan2021.raw.calonoff.ar.T
data/cals/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar
```

We use the script `make_pacv_circ.py` to derive a calibration solution from noise diode archive and polarized quasar archive.
This script reads in `psrchive-archive` file and outputs `pacv` file and a `png` plot file which shows diagnostics.
`pacv` file is the `psrchive` format to store calibration solution.
We will see the `png` file to examine the calibration solution generation.
We visualize `pacv` file either using `pacv` or `vis_pacv.py`.


`make_pacv_circ.py` has the following help:
```
usage: make_pacv_circ [-h] [-z ZAP] [--on ON_REGION] [--off OFF_REGION] [-O ODIR] [-v] [-n] [--delays_grid DELAYS_GRID]
                      [--ionospheric_rm IONOSRM]
                      ar_file

Makes a pacv calibration solution in circular basis

positional arguments:
  ar_file               calibrator archive file

options:
  -h, --help            show this help message and exit
  -z ZAP, --zap ZAP     Zap the channels (comma-separated, start:stop)
  --on ON_REGION        ON region in bins (comma-separated, start:stop)
  --off OFF_REGION      OFF region in bins (comma-separated, start:stop)
  -O ODIR, --outdir ODIR
                        Output directory
  -v, --verbose
  -n, --noise-diode     Noise diode
  --delays_grid DELAYS_GRID
                        Delays grid (min:max:steps)
  --ionospheric_rm IONOSRM
                        Ionospheric RM compute using spinifex

GMRT-FRB polarization pipeline
```

## Measure ON and OFF regions

We need to give ON and OFF regions as input to the script. To locate them, we make use of another script `marker.py`.

```
usage: marker [-h] [-t TS] [-f FS] file

positional arguments:
  file                  archive file

options:
  -h, --help            show this help message and exit
  -t TS, --tscrunch TS  Time scrunch
  -f FS, --fscrunch FS  Freq scrunch

Part of GMRT/FRB polarization pipeline
```
Time scrunch and frequency scrunch control averaging in time or frequency axis. It does not affect us so much at this point, so we can leave it as default.

We start with identifying ON and OFF regions of `data/cals/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar`.
We run:
```
python scripts/marker.py data/cals/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar
```
. We see dynamic spectra (in bottom panel) and frequency averaged profile (in top panel).
Play around with the Constrast slider bar to see what it does.
There are tools for zooming-in-out (magnifying glass) and panning (moving around, four-arrows-thing) in the bottom left panel, that you can also use.

At this point, we are interested in measuring ON and OFF windows. We look at the Time axis. 
As blue region corresponds to low in the time profile (top panel), that is OFF region.
The yellow region is where the time profile is high, so that is ON region.
By eye, we can identify
```
ON region  = 150 : 350
OFF region = 0:100, 400:500 
```

_Would you agree with this?_
Note that there is no one answer here. You could have chosen `140:340` as ON region as well. 
It does not matter. All that matters is, what we consider as ON is ON. 
You could very well choose all the region where the time profile is high as ON region, and everything that is not ON as OFF and it would also be OK.

Now do the same for the other calibrator archive
```
data/cals/3C138_bm1_pa_550_200_32_29jan2021.raw.calonoff.ar.T
```
What is the ON and OFF regions for the above archive?
Identify the regions before proceeding ahead.

## Generating a calibration solution

`Delays grid` is the array of trial cross hand delays that is tested against the calibrator data.
It is in the units of nanoseconds.
The default is from 0 to 100 ns with 2048 samples of resolution.
This range and resolution is sufficient for all calibration purposes. So it is not needed to change it. 
You can change it but it would not change our calibration solution.

#### Noise diode

Since noise diode is inside the receiver, there will not be any ionospheric RM contribution.
Moreover, as the help suggests, we will need to pass `-n` option.

Therefore the command to generate calibration solution from noise diode archive would be
```
python scripts/make_pacv_circ.py -O scratch data/cals/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar --on 150:350 --off 0:100,400:500 -n 
```

We put the calibration products in `scratch`. From the above command, we will have a `pacv` file which stores our calibration solution and which we will use to calibrate our bursts.
We will also have a `png` file which looks like this:

<img src="https://github.com/shiningsurya/GMRT-FRB-Pol-session/blob/15b6ad8cefd5ce27719cfc953932a18f4d54a181/reference/cals/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar.pkg.pcal.png">

The top-left panel shows the frequency average time profile. The red and the green regions are the OFF and ON regions respectively. 
The top-right panel shows magnitude (how correct is the trial cross hand delay) against trial cross hand delays (Delays grid). The vertical dotted black line is at the chosen cross hand delay.
The middle panel shows PHI ($\phi$, in radians) as a function of frequency with data as black points and the linear model with blue line.
The bottom panel shows difference of data and linear model also as a function of frequency. It is clear that error is of the order of 0.1 rad or around 6 deg.

**Notice how the error has structures in it. It does not seem Gaussian. There seems to be locally correlated. This has implications when we are measuring RM.**

#### Polarized quasar

Polarized quasar is an astronomical source. Therefore, there will be some RM contribution, not just from Inter-Stellar-Medium, but also from ionosphere. 

Estimating ionospheric RM contribution is done using `get_ionos_rm.py`, which uses [`spinifex`](https://git.astron.nl/RD/spinifex) package. 
But in order for the `spinifex` package to run, you would need an account on `cddis.nasa.gov` so that you access ionospheric Earth data which complicates the whole process. So, instead, the ionospheric RM contribution is simply provided here. 

`3C138` is shown to have zero RM ([Table 4 of Perley and Bulter, 2013](https://ui.adsabs.harvard.edu/abs/2013ApJS..206...16P/abstract)). But there is one really old paper which says it has an RM of $-2.1$ rad m$^{-2}$ ([Tabara and Inoue (1980)](https://ui.adsabs.harvard.edu/abs/1980A%26AS...39..379T/abstract)). 
We are at a bit of low frequency that it affects us, to err on side of caution, we use it. 

| RM cause | RM |
|----------|----|
| Ionospheric RM contribution | 0.379 | 
| Intrinsic RM | -2.1 |

So, our command to generate calibration solution from `3C138` would look like this:
```
python scripts/make_pacv_circ.py -O scratch data/cals/3C138_bm1_pa_550_200_32_29jan2021.raw.calonoff.ar.T  --ionospheric_rm 0.379
```

**It is naturally missing OFF and ON regions as input and, you are expected to identify and fill them here.**

_if you get stuck, and only if you get stuck, check out_ `reference/cmds_make_pacv` _for the commands._



## Verification

At this point, we will only verify that your cross hand delay and bias measurements agree with my measurements:

| Calibrator archive | Cross hand delay in ns | Bias in radians |
|--------------------|------|-----|
|`data/cals/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar` | 36.852 | -1.790 |
|`data/cals/3C138_bm1_pa_550_200_32_29jan2021.raw.calonoff.ar.T` | 31.574 | 2.756 |

In reality, to verify your calibration solutions, you would need to self calibrate, that is, apply the calibration solution to the same calibrator data from which it is derived and look for anomalies. Or apply the calibration solution on known source (such as a pulsar) and compare with literature measurements.
All of which is beyond the scope of this tutorial.


## Visualize pacv

You can visualize the generated `pacv` files with either `psrchive` command `pacv` or script `vis_pacv.py`.

```
pacv -D 1/xw <generated-pacv-file>
```
or 
```
python scripts/vis_pacv.py <generated-pacv-file>
```

Both show `GAIN, DGAIN` and `DPHASE` as a function of frequency. The three parameters together is used to construct the Mueller matrix. Applying the inverse of the Mueller matrix to observed Stokes parameters is called calibration.

Here is a trick question:
**If the delay is positive, why is the slope of the DPHASE negative?**

## Applying calibration

Suppose the generated `pacv` file is 
```
scratch/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar.pkg.pcal.pacv
```

12 November 2022 is MJD 59894.

We apply the calibration solution to the two bursts from MJD 59894 using `psrchive` command `pac`.
It has a long list of options which you can use by running `pac -h` but we are only interested in the following options:
```
A program for calibrating Pulsar::Archives
Usage: pac [options] filenames
  -q             Quiet mode
  -v             Verbose mode
  -V             Very verbose mode

Calibrator options:
  -A filename    Use the calibrator in filename, as output by pcm/pacv
  -P             Calibrate polarisation only (not flux)

Input/Output options:
  -e ext         Extension added to output filenames (default .calib)
  -O path        Path to which output files are written

See http://psrchive.sourceforge.net/manuals/pac for more details
```

We pass our calibration solution using `-A` option. We will put our calibrated archives in `scratch` by using `-O` option. Moreover, we will give them `.calP` extension by passing `-e cal` option. 
Note that it says `filenames`, that means, we can give as many of calibrated archives as we want as arguments to `pac` and it will calibrate all of them. 
Also, we are only interested in polarization calibration as this point, so we will pass `-P` option.

But in case of MJD 59894, we only have two so our command will look like this:
```
pac -P -O scratch -A scratch/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar.pkg.pcal.pacv data/bursts/59894.7963734623_sn100.87_lof750_R3.ar data/bursts/59894.8480059734_sn49.24_lof750_R3.ar
```

This performs calibration of the two MJD 59894 bursts and puts the calibrated archives in `scratch` folder.

Now, try to calibrate the three MJD 59243 bursts.
_if you get stuck, and only if you get stuck, check out_ `reference/cmds_pac` _for the commands._


## If you have calibrated all the bursts, it marks the ends of this chapter. Next is measurement.
