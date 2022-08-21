Example dataset of the simulated XMM-Newton images.
The separate parts of the simulated XMM-Newton images (img, agn and background) were simulated to a 100ks exposure time at both 1x and 2x resolutions. Next, they were split into multiple exposure time images in steps of 10ks. See the thesis for more information.
The train, val and test directories contain symbolic links to the images. These are the splits used during the research. The train_sweep, val_sweep and test_sweep are the splits used for the sweep runs, these point to a subset of the whole dataset. 
Finally, the detector_mask directory contains the detector_mask used to simulate the chip gaps, bad pixels and remove events outside the field of view.

The simulated XMM-Newton images consist of separate parts, the img, agn and background. These simulations use the similarly named simput files as input. All these parts were simulated at both the real XMM-Newton resolution with its normal PSF (1x) and at twice the resolution with half the PSF (2x).

img:
The simulated XMM-Newton image filenames are based on the simput filenames. These simputs contain Illustris TNG projections/slices as sources. These sources got an assigned brightness, are zoomed in and are offset (augmentations). The naming convention of these files is similar to the simputs plus the resolution, exposure time and split part.
For example:
'TNG50_1_z_99_subhalos_96762_m_proj_r_2048_w_400_n_1_0_0_gz_p0_0_5ev_p1_2_0ev_sb_37_4_zoom_1_81_offx_-0_12_offy_-0_08_mult_1_100ks_p_0-0.fits.gz'
The first part is the illustis simulation name (TNG50_1), then the illustris time slice (z_99), the subhalo id (subhalos_96762), the mode (m_proj); this can either be a projection or a slice, the resolution (r_2048), the width projection/slice (w_400), the orientation of the projection/slice (n_1_0_0), where the first number is the x-axis, the second the y axis and the last the z-axis. Thus, in this case, the projection was from the positive x-axis.
The energy ranges (gz_p0_0_5ev_p1_2_0ev, note that the '_' are used as dots here), the source brightness (sb_37_4), the zoom (zoom_1_81), the x and y offsets (offx_-0_12_offy_-0_08), the resolution multiplier (mult_1), the exposure time (100ks) and finally the exposure time part (p_0-0), i.e. part 0 - 0 (start counting from 0).

agn:
The simulated agn parts contain randomly located point sources with absorption determined by shifting the log(N)/log(S) distribution.
Naming convention:
Given the following filename `agn_abs_99.9_16370988299599936_p0_0.5ev_p1_2.0ev_mult_1_100ks_p_0-0.fits.gz`
The first part signals that it is an agn (agn), then the absorption rate (abs_99.9), then a unique id (16370988299599936), the energy ranges (p0_0.5ev_p1_2.0ev), the resolution multiplier (mult_1), the exposure time (100ks) and finally the exposure time part (p_0-0), i.e. part 0 - 0 (start counting from 0).


background:
The simulated background parts are all based on the same simulation input, based on black sky eventlists. 
Naming convention:
Given the following filename 'background_mult_1_100ks_16293484197538493.fits.gz'

The first part signals that it is a background (background), the resolution multiplier (mult_1), the exposure time (100ks) and finally, a unique id (16293484197538493).
