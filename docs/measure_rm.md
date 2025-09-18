
# Measure RM

Having calibrated bursts, we start the procedure of measuring RMs.
Firstly, we will zap any bad frequency channels which are full of RFI using `pazi`. 
Then, we identify the burst region of the burst using `marker.py`.
After which, we will measure RM and PA (at infinite frequency) using `measure_rm_pa_spec.py`.
Finally, we use `pam` to perform the Faraday rotation correction to the calibrated archive to obtain RM corrected calibrated archives.

## Frequency zapping

Calibration scales every frequency channel because of `GAIN` term in the calibration model.
Therefore, it is advisable to do a round of frequency zapping where we identify and disable any bad frequency channels, as now all frequency channels of the same scale.
We do this by using `pazi`.

You really cannot do anything wrong while frequency zapping. Just make sure not to zap the channels with actual burst. 
In this tutorial, bursts provided are high S/N and free of RFI, so you might even consider skipping this step.

And for those brave souls, who would not skip, run `pazi` like this:

```
pazi <archive-file>
```

And then hit `h` to get help. This tutorial would not provide further help. It is not so hard. Ask the teacher to explain once. 
It is something that will take forever to explain through text but is quite easy to show. 

## Burst region

It is understood that all further analysis is done with either `pazi` archive or non `pazi` archive if you skipped the frequency zapping step.

Now, we measure the ON region of the burst. As an example, we work with 
```
data/bursts/59894.7963734623_sn100.87_lof750_R3.ar
```
Which post calibration would be something like, 
```
scratch/59894.7963734623_sn100.87_lof750_R3.calP
```
assuming your work directory is `scratch` and the `-e cal` was passed to `pac`.

We make use `marker.py` script to identify the burst region.

```
python scripts/marker.py <calibrated-archive>
```

**Where is the burst region?**

Use the zoom (magnifying glass, bottom left panel) to zoom over the burst region.
Using the mouse, click and drag to draw a rectangle over the burst region.
Use the controls over the edges to adjust the rectangle so that it fits as best as possible to the burst region.
You can control the constrant with slider given on the right. 
The two thick black vertical lines shows the temporal extent of the rectangle. 
Once you are happy with your selection, press `m` to save the region as a `JSON` file.

Do this exercise for all the bursts. 
If you want to cheat, `reference/jsons` has all the jsons already. 

**but i really recommend you try this exercise because this is real astronomy.**

## Measuring RM and PA (at infinite frequency)

We measure RM and PA at infinite frequency using `measure_rm_pa_spec.py`. It has the following help:
```
usage: rm_spec [-h] [-f FS] [-n NTRIALS] [-r RMGRID] --ton TON --fon FON [-v] [-O ODIR] ar_file

positional arguments:
  ar_file               burst archive file

options:
  -h, --help            show this help message and exit
  -f FS, --fscrunch FS  Frequency downsample
  -n NTRIALS, --ntrials NTRIALS
                        Number of bootstrap trials
  -r RMGRID, --rmgrid RMGRID
                        RM grid (min:max:steps)
  --ton TON             Burst time window in bins (start:stop)
  --fon FON             Burst frequency window in bins (start:stop)
  -v, --verbose         Verbose
  -O ODIR, --outdir ODIR
                        Output directory

Part of GMRT/FRB polarization pipeline
```

This script measures RM by identifying the peak in magnitude over RM grid.
To provide reasonable error measurements and avoid finite RM grid resolution effects, it does bootstrap. That is, it re-samples with replacement `NTRIALS` (default is 999) times, and for each time, measures RM and PA (at infinite frequency). Then, it computes average and standard deviations of both.

`--ton` and `--fon` define the burst region. `-r` passes the RM grid. Since the RM can be negative, and since negative numbers begin with `-` which interferes with arguments, we pass RM grid as `-r=min:max:steps`, which will be obvious when we look at an example.


Suppose we are working with `scratch/59894.7963734623_sn100.87_lof750_R3.calP`, whose burst region is 
```
{
  "tstart": 246,
  "tstop": 266,
  "fstart": 48,
  "fstop": 2048,
  "width_ms": 6.272255968816038,
  "bw_mhz": 195.23215846696846
}
```

Then, RM measurement command would be 
```
python scripts/measure_rm_pa_spec.py -f 4 -n 999 -r="-160:-30:2048" -O scratch --ton 246:266 --fon 48:2048 scratch/59894.7963734623_sn100.87_lof750_R3.calP
```

It would output the following files
```
59894.7963734623_sn100.87_lof750_R3.calP.pkg.npz.png
59894.7963734623_sn100.87_lof750_R3.calP.pkg.npz_spec.csv
59894.7963734623_sn100.87_lof750_R3.calP.pkg.npz_spec.npz
```

`csv` is a simple table which tabulates all the measurements. The bootstrapped RM and PA are the `rm_mean` and `pa_mean` columns. PA given here is given in radians.
`npz` file stores the entire data which helps me keeping track of the measurement process. Feel free to explore it using `numpy`.
`png` file is the diagnostic plot we examine to test our measurement.

<img src="https://github.com/shiningsurya/GMRT-FRB-Pol-session/blob/60388c25d07ab68a99157b63e369ff5e9fba5b1f/reference/rms/59894.7963734623_sn100.87_lof750_R3.Pz.pkg.npz.png">

The top leftmost panel shows the bootstrapped histogram of the RM. The vertical black dotted line is the average. 
Similar plot for PA (at infinite frequency) in degrees is shown in top middle panel.
The top right panel shows the magnitude versus RM grid. The vertical blue dashed line is the peak RM.
The middle panel shows PA in degrees against wavelength squared (in meter squared). The black dots with error bars is from data. The blue line is the model.
The bottom panel shows the difference between the data and model in degrees. 
The top x-axis in middle and bottom panel shows corresponding frequency in MHz.

Similar command can be constructed for other calibrated bursts, and polarization observables can be measured. 

_if you get stuck, and only if you get stuck, check out_ `reference/cmds_measure_rm` _for the commands._


## Verification

If you have measured RMs and PAs, you can either your measurements against this.
You can also look through `reference/rms` for the diagnostic plots.

| Burst | RM (rad per meter squared) | PA (at infinite frequency, degrees) |
|---|---|---|
| `59243.4552563413_sn77.27_lof750_R3.ar` | -116.964 | 33.669 |
| `59243.4823292439_sn121.00_lof750_R3.ar` | -116.423 | 31.933 |
| `59243.5481613923_sn97.90_lof750_R3.ar` |  -116.634 | 45.414 |
| `59894.7963734623_sn100.87_lof750_R3.ar` | -62.114 | -10.562 |
| `59894.8480059734_sn49.24_lof750_R3.ar` | -61.726 | -13.016 | 


## Faraday rotation correction

We perform RM correction using `pam`.
It also has a really long list of options, but we only focus on the following:
```
A program for manipulating Pulsar::Archives
Usage: pam [options] filenames
  -q               Quiet mode
  -v               Verbose mode
  -V               Very verbose mode
  -e extension     Write new files with this extension
  -u path          Write files to this location

The following options take floating point arguments
  -d DM            Alter the header dispersion measure
  -R RM            Correct ISM Faraday rotation (wrt centre freq)
  --RM RM          Install a new RM (but don't correct)

See http://psrchive.sourceforge.net/manuals/pam for more details
```
We will pass our measured RM using `-R` option.
Note that `pam` only corrects with respect to centre frequency. That means, whatever PA we measure will also be referenced with respect to centre frequency.

Working again with `scratch/59894.7963734623_sn100.87_lof750_R3.calP`, whose RM we measure to be -62.114 rad per meter squared, we can perform Faraday correction with the following command:
```
pam -R -62.114 -e PR scratch/59894.7963734623_sn100.87_lof750_R3.calP
```

We give a new extension `PR` to denote Polarization calibration and Faraday rotation corrected burst archives. 

Run similar commands to do Faraday rotation correction of the remaining bursts. 

## Visualize

We use `psrplot` to visualize the calibrated burst, and calibrated and Faraday rotation corrected burst.

`psrplot` can visualize a wide variety of plots which can be seen with `psrplot -P`, but we focus on the following which visualize the polarization information:
```
stokes    [s]  Stokes parameters
Scyl      [S]  Stokes; vector in cylindrical
Ssph      [m]  Stokes; vector in spherical
pa        [o]  Orientation (Position) angle
ell       [e]  Ellipticity angle
p3d       [P]  Stokes vector in Poincare space
```

As before, let us suppose that we are working with `scratch/59894.7963734623_sn100.87_lof750_R3.calP` and `scratch/59894.7963734623_sn100.87_lof750_R3.PR`, where the former is the calibrated burst archive and the latter is calibrated and Faraday  rotation corrected burst archive.

To begin with, we look at `Scyl` plot style. For which, we run the following:
```
psrplot -p Scyl -jF -c "x:range=(0.4,0.6)" scratch/59894.7963734623_sn100.87_lof750_R3.calP scratch/59894.7963734623_sn100.87_lof750_R3.PR
```
`-jF` performs frequency averaging, otherwise, it only plots one channel. `-c "x:range=(0.4,0.6)"` sets the `x` axis limits. 
Try running with other arguments to see what you get.

We see the following:

<img src="https://github.com/shiningsurya/GMRT-FRB-Pol-session/blob/main/reference/pngs/calcalrm.png">

The bottom panels show frequency averaged burst profile. The black line is total intensity, the blue line is circular polarization, and the red line is the linear polarization.
The top panels show PA with errors in degrees against the same `x` axis. 

The left plot which is simply calibrated does not show any linear polarization (red), whereas the right plot shows. 
This is because, Faraday rotation correction corrects for the rotation in Stokes Q and U, so that when you average over frequency, the average stays.

You can only add Stokes parameters, not Position Angles (**why?**).

Notice how the PA is flat within the burst. Also, notice that it hovers around -45 degrees. 
`psrchive` PA is with respect to its centre frequency, that is 650 MHz. 

Compare with above where we verified the PA (at infinite frequency) to be arround -10.5 degrees. Notice how for a RM of -62.114 rad per sq meter, at a frequency of 650 MHz, the PA measured would be around -47 degrees.
What we see with `psrchive` matches well with our measurements.
You can try to test the same for other bursts as well. The formula is 
```
PA_650 = PA_oo + RM * lambda_650^2
```
where `lambda_650` is wavelength of 650 MHz.

# Circular polarization bump

Do you notice the bump in blue line around where the burst emission peaks? Do you think it is significant enough? 
Does it mean there is no intrinsic circular polarization or there is a _minor_ calibration issue that is only visible when the flux is significant enough?


## If you have measured RMs and PAs from all the bursts, it marks the ends of this chapter and this tutorial as well. 

But i encourage you to try plotting your calibrated and calibrated and Faraday rotation corrected bursts using `psrplot`.
i also encourage you to play around with the data.
