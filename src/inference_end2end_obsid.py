#
# Example script to run the inference and predict super-resolution (SR) or denoised (DN) XMM EPIC-pn image
# 
# Using XMM SAS xmmsas_tools in utils
#
# The workflow is the following:
#
# 1. Download the PPS files for an OBSID from the XMM archive
# 1. Generate GTI files based on PPS flaring background and threshold
# 1. Generate a new GTI with max 20ks exposure
# 1. Filter the PPS event lists with the GTI
# 1. Using the cleaned event lists, generate DETX,DETY image in a given energy band
# 1. Run the inference on this image and produce the predicted SR and DN images
#
#%%
import argparse
import os

from run_inference_on_real import run_inference_on_real
from utils.xmmsas_tools import check_pps_dir, make_gti_pps, filter_events_gti, make_detxy_image, get_pps_nxsa

#%%
# Parse the input 
parser = argparse.ArgumentParser(description='Predict XMM SR or DN image')
parser.add_argument('obsid', type=str,
                    help='The OBS_ID to process')
parser.add_argument('--wdir', type=str,default=os.getcwd(),
                    help='The working top folder name, must exist')
parser.add_argument('--expo_time', type=float,default=20,
                    help='Will extract only this exposure time (in ks) from the event list. Set it to negative to use the GTI one.')
#
args = parser.parse_args()

top_d = args.wdir
#top_d = '/lhome/ivaltchanov/XMM/xmm_super'
if (not os.path.isdir(top_d)):
    print ('Error: working folder {top_d} does not exist, cannot continue!')
    raise FileNotFoundError
#
#obsid = '0852030101' # M51 (from paper, Fig. 7 bottom)
obsid = args.obsid
expo_time = args.expo_time

pps_d = f'{top_d}/{obsid}/pps'
#
pps_files = get_pps_nxsa(obsid,wdir=top_d,skip=True,keeptar=False,verbose=True)
#
proc_d = f'{top_d}/{obsid}/proc'
if (not os.path.isdir(proc_d)):
    os.mkdir(proc_d)
#
os.chdir(proc_d)

pps_files = check_pps_dir(pps_dir=pps_d,verbose=True)

png_file = f'pn_gti_full_figure.png'
if (expo_time > 0):
    png_file = f'pn_gti_{expo_time}ks_figure.png'
#
gtis = make_gti_pps(pps_d,instrument='pn',out_dir=proc_d,max_expo=expo_time,plot_it=True,save_plot=png_file)
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
evl_filt_name = f'pn_cleaned_evl_full.fits'
if (expo_time > 0):
    evl_filt_name = f'pn_cleaned_evl_{expo_time}ks.fits'

evl_filt = filter_events_gti(evl,pn_gti,pps_dir=pps_d,output_name=evl_filt_name,verbose=True)
if (evl_filt is None):
    raise RuntimeError
#
e1 = 500
e2 = 2000
outname = f'pn_{e1}_{e2}_detxy_image_full.fits'
if (expo_time > 0):
    outname = f'pn_{e1}_{e2}_detxy_image_{expo_time}ks.fits'
detxy = make_detxy_image(evl_filt,pps_dir=pps_d,output_name=outname,low_energy=e1,high_energy=e2,bin_size=80,radec_image=False,verbose=True)
if (detxy is None):
    raise RuntimeError
#
# now run the inference
#
status = run_inference_on_real(detxy,'SR',display=False)
if (status != 0):
    raise RuntimeError

status = run_inference_on_real(detxy,'DN',display=False)
if (status != 0):
    raise RuntimeError
