
# Measure RM

Having calibrated bursts, we start the procedure of measuring RMs.
To begin with, we use `marker.py` to measure ON region of the burst.
Then we use `make_pkg.py` to create a `numpy-zip` file.
`make_pkg.py` uses `psrchive-python` to read the archive and saves as `numpy` array.
It also uses output of `marker.py` to keep track of the ON region.

## 
