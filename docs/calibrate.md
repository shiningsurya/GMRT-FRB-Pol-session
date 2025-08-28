
# Calibration

We derive calibration solution from noise diode scan and polarized quasar scan.
We first examine the calibrator archives and then convert `psrchive-PSRFITS` format calibrator archives into `npz` file.
This conversion is required because `psrchive` performs various data processing in the background that is a pain to implement outside of `psrchive`.

We make use `make_pkg.py` script to do so. It uses `psrchive-python`. It has the following help.
```
usage: make_pkg [-h] [-O ODIR] [-j JSON] [-n] [-DD] [-RR] file

positional arguments:
  file                  archive file

optional arguments:
  -h, --help            show this help message and exit
  -O ODIR, --outdir ODIR
                        Output directory
  -j JSON, --json JSON  JSON file containing tstart,tstop,fstart,fstop
  -n, --no-json         Make pkg without json
  -DD                   Do not de-disperse
  -RR                   Do not de-baseline

Part of GMRT/FRB
```

At this point, we do not need to give a `JSON` file (it will be necessary when measuring RM). 
Calibrators do not have any dispersion measure and we do not want to keep baseline so our incantation would be like
```
python scripts/make_pkg.py -O scratch/ <calibrator-archive>
```

Replace `<calibrator-archive>` with any of the calibrator archives from `data/cals`.


Having created `npz` file, we now generate a calibration solution using `make_pacv_circ.py`.
This script reads in `npz` file and outputs `pacv` file and a `png` plot file which shows diagnostics.
We can visualize `pacv` file either using `pacv` or `vis_pacv.py`.

`make_csv_circ.py` has the following help:
```
Makes a pacv calibration solution in circular basis

positional arguments:
  npz_file              npz file (output of make_pkg.py)

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

```
psrchive> pacv <pacv-file>

$> python scripts/vis_pacv.py <pacv-file>
```

A typical png file looks like 

<img src="references/cals/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar.pkg.pcal.png">


Having calibrated bursts, we start the procedure of measuring RMs.
To begin with, we use `marker.py` to measure ON region of the burst.
Then we use `make_pkg.py` to create a `numpy-zip` file.
`make_pkg.py` uses `psrchive-python` to read the archive and saves as `numpy` array.
It also uses output of `marker.py` to store the ON region within the `numpy-zip` file.
Lastly, `measure_rm_pa_spec.py` reads the `numpy-zip` file and fits RM and PA (at infinite frequency).

