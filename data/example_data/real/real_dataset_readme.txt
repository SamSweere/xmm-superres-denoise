Dataset of real XMM observations within the 0.5 to 2.0 kEv range. Where the eventlist is first cleaned and then split into 10ks observations. The directory 'full' contains the full-length cleaned observation. See the thesis for more information. 
Not all observations are successfully processed into an image. The failed observations can be seen in failed_obs.txt. Most of them are because of an incorrect observation mode. 
The train, val and test directories contain symbolic links to the images. These are the splits used in the research. 

Naming convention: 
The filenames are structured seperated by an '_'. For example, given the following filename: '0860460101_image_split_500_2000_40ks_2_1.fits'

The first part is the observation id (0860460101), then 'image_split', the energy ranges (500 ev to 2000 ev), exposure time (40ks), the amount of different exposure time splits there are for this observation (2) and finally the split number (1).
