import os

import numpy as np
from astropy.io import fits
from tqdm import tqdm

def load_fits(fits_path):
    try:
        hdu = fits.open(fits_path)
        # Extract the image data from the fits file and convert to float
        # (these images will be in int but since we will work with floats in pytorch we convert them to float)
        img = hdu['PRIMARY'].data.astype(np.float32)
        exposure = hdu['PRIMARY'].header['EXPOSURE']
        header = dict(hdu['PRIMARY'].header)

        # The `HISTORY` key is causing problems
        header.pop('HISTORY', None)
        hdu.close()

        # Devide the image by the exposure time to get a counts/sec image
        img = img / exposure

        return {'img': img, 'exp': exposure, 'file_name': os.path.basename(fits_path), 'header': header}
    except Exception as e:
        print("ERROR failed to load fits file with error: ", e)
        print(fits_path)
        raise IOError(e)

def check_fits_file(fits_path):
    try:
        img = load_fits(fits_path)['img']

        max_val = 100000
        min_val = 0

        if np.isnan(np.sum(img)):
            raise ValueError(f"ERROR {fits_path} contains a NAN")

        if np.max(img) > max_val:
            raise ValueError(f"ERROR {fits_path} contains a value bigger then {max_val} ({np.max(img)})")

        if np.min(img) < min_val:
            raise ValueError(f"ERROR {fits_path} contains a value smaller then {min_val} ({np.min(img)})")
    except Exception as e:
        print("ERROR: ", e)
        print(fits_path)


dataset_path = os.path.join(os.path.expanduser("~"), 'data/sim/xmm_sim_dataset')

print("Started checking:", dataset_path)

f = []
# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(dataset_path):
    # path = root.split(os.sep)
    # print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        # print(len(path) * '---', file)
        if file.endswith('.fits') or file.endswith('.fits.gz'):
            f.append(os.path.join(root, file))

for file in tqdm(f):
    check_fits_file(file)

