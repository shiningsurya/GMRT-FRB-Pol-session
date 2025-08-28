
Welcome to the hands-on session of the GMRT Polarimetry. In this session, we will polarization calibrate and measure various polarization observables of bursts of repeating Fast Radio Burst 20180916B detected using upgraded Giant Metrewave Radio Telescope (uGMRT) in Band 4 (550 to 750 MHz).

We have five bursts, three detected on MJD 59243 and two on MJD 59894. MJD 59243 bursts are calibrated using a polarized quasar (3C138) and MJD 59894 bursts using a noise diode scan. In either case, the calibration procedure is the same, but as we will see, calibration done using polarized quasar requires some additional inputs.

Post calibration, we will measure the Rotation Measure (RM), Positon Angle (PA), and linear polarization fraction (Lp).

## Before we get started

### Tutorial 

This hands-on session assumes a bit of polarization knowledge that is presented in the [google-slides](link-here).
The presentation is the tutorial and it is expected that you have gone through it before attempting this hands-on exercise.

### Requirements

We will make use of `psrchive` commands and also use `psrchive-python` within `Python` scripts. 
Additionally, we require the following `Python` packages.
The basic scientific python packages:
- `numpy`, `matplotlib`, `pandas`, `pickle` , `astropy`
- [`spinifex`](https://git.astron.nl/RD/spinifex) which we only need for calculating Ionospheric RM contribution. So it is not strictly needed for this tutorial.


### Data

All the data required for this hands-on session is provided in `data`.
It has two calibrator archives:
```
data/cals/3C138_bm1_pa_550_200_32_29jan2021.raw.calonoff.ar.T
data/cals/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar
```
and five FRB 20180916B de-dispersed burst archives detected by uGMRT in Band 4 (550-750 MHz):
```
data/bursts/59243.4552563413_sn77.27_lof750_R3.ar
data/bursts/59243.4823292439_sn121.00_lof750_R3.ar
data/bursts/59243.5481613923_sn97.90_lof750_R3.ar
data/bursts/59894.7963734623_sn100.87_lof750_R3.ar
data/bursts/59894.8480059734_sn49.24_lof750_R3.ar
```

## Let's go ...

### Solutions

`reference` folder (and its sub-folders) contain all the commands run, calibration solutions, calibrated bursts, and measurements that is provided as reference. 
If at any time you get stuck, please feel free to look for hints (or the specific command) in this folder. 

### scratch

It is highly recommended to create a new folder where all the intermediary and diagnostic plots can be saved. 
It is named `scratch` within this hands-on session, but you can choose to call it anything.
But then make sure to refer to this directory when running scripts or commands.

### Caveats

This tutorial conveniently skips calibration procedure using an unpolarized quasar and pulsar, as it would complicate the tutorial further.

We noticed that `pac` when working on `PSRFITS` format archives performs incorrect parallactic angle correction.
So, in order to measure accurate PAs using `psrchive`, one must keep the format of the bursts in `TIMER` format. 

## Glossary of commands and script

| Action | What it does |
|--------|--------------|
| `pac`  | This is a `psrchive` command which is used to polarization calibrate archives. |
| `pazi`  | This is a `psrchive` command which is used to interactively zap frequency channels with RFI. |
| `pam`  | This is a `psrchive` command which can do all the need Archive Manipulations. We use it to do Faraday Rotation correction. |
| `make_pacv_circ.py` | This is a `Python` script to generate a calibration solution from either a polarized quasar or noise diode scan for data in Circular basis. | 
| `measure_rm_pa_spec.py` | `Python` script to measure RM and PA (at infinite frequency) | 
| `make_pkg.py` | _might get obsolete soon_  Converts `psrchive-archive` format into `numpy` ready for science format. | 

