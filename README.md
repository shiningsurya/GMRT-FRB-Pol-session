# GMRT FRB polarization session

## Idea

This pack contains the scripts, test data, and procedure to extract polarization sciences from FRB measurements with uGMRT.

This pack is exclusively limited to Band~4 (550-750) MHz at this point and will be extended with time.


## Contains

data/bursts has five bursts from MJD 59243 that are already put in archives. 

data/burstraw is the PA file snippet that has a burst, just to show how to convert the bursts from uGMRT format to psrchive format.

data/cals has snippets of noise diode raw and folded noise archive.

data/cals has snippets of unpolarized calibrator and pulsar.

caldata contains calibrated bursts

scripts/marker to measure ON region

scripts/makepkg to make package

scripts/measurermpa to measure RM and PA

scripts/measurepol to measure I,L,V

refdata reference data contains all the results 

## caveats

skipping quasar+pulsar calibration solution -- too complicated.
skipping accurate PA measurement -- psrchive bug that does wrong parallactic angle correction when working with not timer archives
