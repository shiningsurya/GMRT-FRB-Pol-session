
# Calibration

We derive calibration solution (in `pacv` format) from noise diode scan and polarized quasar scan.

We make use of `make_pacv_circ.py` script which reads in `npz` file and outputs `pacv` file, `png` plot file which shows diagnostics.

We can visualize `pacv` file either using `pacv` or `vis_pacv.py`.

```
psrchive> pacv <pacv-file>

$> python scripts/vis_pacv.py <pacv-file>
```

A typical png file looks like 
![pacv png](../references/cals/FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.5.noise.Tar.pkg.pcal.png)



Having calibrated bursts, we start the procedure of measuring RMs.
To begin with, we use `marker.py` to measure ON region of the burst.
Then we use `make_pkg.py` to create a `numpy-zip` file.
`make_pkg.py` uses `psrchive-python` to read the archive and saves as `numpy` array.
It also uses output of `marker.py` to store the ON region within the `numpy-zip` file.
Lastly, `measure_rm_pa_spec.py` reads the `numpy-zip` file and fits RM and PA (at infinite frequency).

