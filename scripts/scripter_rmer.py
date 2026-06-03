"""
write script to make command to run
"""

import os
import json
import sys

json_dir = "calibrated_ifits"
#json_dir = "aux_calibrated_ifits"

ar_file  = sys.argv[1]

bname    = os.path.basename ( ar_file )
jsonfile = os.path.join ( json_dir, bname + ".json" )

with open (jsonfile, 'r') as _f:
    jdict= json.load ( _f )

ton      = f"{jdict['tstart']:d}:{jdict['tstop']:d}"
fon      = f"{jdict['fstart']:d}:{jdict['fstop']:d}"

#CMD      = f"python /gpol/measure_rm_pa_spec.py -f 4 -n 99 --ton {ton} --fon {fon} -v -O aux_rm_measurements {ar_file}"
#CMD      = f"python /gpol/measure_rm_pa_spec_2d.py -f 4 -n 99 --ton {ton} --fon {fon} -v -O aux_rm_measurements {ar_file}"
CMD      = f"python /gpol/prepare_onerm.py -f 4 --ton {ton} --fon {fon} -v -O onerm_npz {ar_file}"

print ( CMD )

