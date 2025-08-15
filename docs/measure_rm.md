
# Measure RM

Having calibrated bursts, we start the procedure of measuring RMs.
To begin with, we use `marker.py` to measure ON region of the burst.
Then we use `make_pkg.py` to create a `numpy-zip` file.
`make_pkg.py` uses `psrchive-python` to read the archive and saves as `numpy` array.
It also uses output of `marker.py` to store the ON region within the `numpy-zip` file.
Lastly, `measure_rm_pa_spec.py` reads the `numpy-zip` file and fits RM and PA (at infinite frequency).

