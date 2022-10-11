#
# test the XMM SAS xmmsas_tools
#
# The workflow is the following:
#
# 1. Generate GTI files based on PPS flaring background and threshold
# 2. Filter the PPS event lists with the GTI
# 3. Using the cleaned event lists, generate images in a given energy band
#

from utils.xmmsas_tools import check_pps_dir, make_gti_pps, filter_events_gti,make_detxy_image

import os
pps_d = '/lhome/ivaltchanov/XMM/detxy_to_wcs_tests/0656201401/pps'
proc_d = '/lhome/ivaltchanov/XMM/detxy_to_wcs_tests/0656201401/proc'

os.chdir(proc_d)

pps_files = check_pps_dir(pps_dir=pps_d,verbose=True)

gtis = make_gti_pps(pps_d,instrument='all',out_dir=proc_d,plot_it=True,save_plot='all_gti_figure.png')
if (len(gtis) < 1):
    raise RuntimeError
# select the PN event list in the PPS
for ifile in pps_files['evl_files']:
    if ('PN' in ifile.upper()):
        evl = ifile
        break
for ifile in gtis:
    if ('PN' in ifile.upper()):
        pn_gti = ifile
        break
#
evl_filt = filter_events_gti(evl,pn_gti,pps_dir=pps_d,output_name='pn_cleaned_evl.fits',verbose=True)
if (evl_filt is None):
    raise RuntimeError
#
detxy = make_detxy_image(evl_filt,pps_dir=pps_d,low_energy=500,high_energy=2000,bin_size=80,radec_image=True,verbose=True)
if (detxy is None):
    raise RuntimeError
#
print ('Test done')
