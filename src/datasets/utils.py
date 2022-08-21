import os

import numpy as np
from astropy.io import fits


def load_fits(fits_path):
    try:
        hdu = fits.open(fits_path)
        # Extract the image data from the fits file and convert to float
        # (these images will be in int but since we will work with floats in pytorch we convert them to float)
        img = hdu["PRIMARY"].data.astype(np.float32)
        exposure = hdu["PRIMARY"].header["EXPOSURE"]
        header = dict(hdu["PRIMARY"].header)

        # The `HISTORY`, `COMMENT` and 'DPSCORRF' key are causing problems
        header.pop("HISTORY", None)
        header.pop("COMMENT", None)
        header.pop("DPSCORRF", None)
        header.pop("ODSCHAIN", None)
        header.pop("SRCPOS", None)

        hdu.close()

        # Devide the image by the exposure time to get a counts/sec image
        img = img / exposure

        return {
            "img": img,
            "exp": exposure,
            "file_name": os.path.basename(fits_path),
            "header": header,
        }
    except Exception as e:
        print("ERROR failed to load fits file with error: ", e)
        print(fits_path)
        raise IOError(e)


def reshape_img_to_res(dataset_lr_res, img, res_mult):
    # The image has the shape (411, 403), we pad/crop this to (dataset_lr_res, dataset_lr_res)
    y_diff = dataset_lr_res * res_mult - img.shape[0]
    y_top_pad = int(np.floor(y_diff / 2.0))
    y_bottom_pad = y_diff - y_top_pad

    x_diff = dataset_lr_res * res_mult - img.shape[1]
    x_left_pad = int(np.floor(x_diff / 2.0))
    x_right_pad = x_diff - x_left_pad

    if y_diff >= 0:
        # Pad the image in the y direction
        img = np.pad(
            img, ((y_top_pad, y_bottom_pad), (0, 0)), "constant", constant_values=0.0
        )
    else:
        # Crop the image in the y direction
        img = img[abs(y_top_pad) : img.shape[0] - abs(y_bottom_pad)]

    if x_diff >= 0:
        # Pad the image in the x direction
        img = np.pad(
            img, ((0, 0), (x_left_pad, x_right_pad)), "constant", constant_values=0.0
        )
    else:
        # Crop the image in the x direction
        img = img[:, abs(x_left_pad) : img.shape[1] - abs(x_right_pad)]

    return img


def get_fits_files(dataset_dir):
    if not os.path.exists(dataset_dir):
        raise OSError(f"Dataset directory {dataset_dir} does not exists")

    fits_files = []
    file_names = []

    # Only save the files that end with .fits
    for file in os.listdir(dataset_dir):
        if file.endswith(".fits") or file.endswith(".fits.gz"):
            fits_files.append(file)
            file_name = os.path.splitext(file)[0]
            file_names.append(file_name)

    print(f"Detected {len(fits_files)} fits files in {dataset_dir}")

    return fits_files, file_names


def _filter_file_list(filelist1, filelist2, base_names, split_key):
    base_names_set = set(base_names)

    filtered_filelist1 = []
    for i_list in filelist1:
        filtered_filelist1.append(
            [[] for i in range(len(base_names))]
        )  # Note that every list has to be created separately
    # to be separate items
    filtered_filelist2 = [[] for i in range(len(base_names))]

    # Only keep the ones that the files in both resolutions
    # Since we can have multiple files with the same base_name we want to cluster these too
    for i_list in range(len(filelist1)):
        for file1 in filelist1[i_list]:
            base_name = file1.split(split_key)[0]
            if base_name in base_names_set:  # Make use of the set speed
                index = base_names.index(base_name)
                filtered_filelist1[i_list][index].append(file1)

    for file2 in filelist2:
        base_name = file2.split(split_key)[0]
        if base_name in base_names_set:  # Make use of the set speed
            index = base_names.index(base_name)
            filtered_filelist2[index].append(file2)

    return filtered_filelist1, filtered_filelist2


def match_file_list(filelist1, filelist2, split_key):
    """

    Filters and groups two filelists on having the same base name when split with the split_key.

    Args:
        filelist1: List of files1 or list of lists
        filelist2: List of files2
        split_key: The string where the filenames should be split on

    Returns:
        A list of lists of filelist1, a list of lists of filelist2 and list of the base_filenames all grouped on the
        base filenames.

    """

    list_of_lists = False

    # Check if it is an list of lists (multiple exposure times for inputs)
    if any(isinstance(el, list) for el in filelist1):
        list_of_lists = True
    else:
        filelist1 = [filelist1]

    # Find the intersect between file_list_1
    base_names1 = None
    for list_i in filelist1:
        base_names_i = set([x.split(split_key)[0] for x in list_i])
        if base_names1 is None:
            base_names1 = base_names_i
        else:
            base_names1 = base_names1 & base_names_i

    base_names2 = [x.split(split_key)[0] for x in filelist2]

    # Get the intersection between the lists
    name_intersect_set = set(base_names1) & set(base_names2)
    name_intersect = list(name_intersect_set)

    filtered_filelist1, filtered_filelist2 = _filter_file_list(
        filelist1, filelist2, name_intersect, split_key
    )

    if not list_of_lists:
        filtered_filelist1 = filtered_filelist1[0]

    return filtered_filelist1, filtered_filelist2, name_intersect


def group_same_sources(filelist1, filelist2, split_key):
    """

    Groups two filelists on having the same base name when split with the split_key.

    Args:
        filelist1: List of files1 or list of lists
        filelist2: List of files2
        split_key: The string where the filenames should be split on

    Returns:
        A list of lists of filelist1, a list of lists of filelist2 and list of the base_filenames all grouped on the
        base filenames.
    """

    list_of_lists = False

    # Check if it is an list of lists (multiple exposure times for inputs)
    if any(isinstance(el, list) for el in filelist1):
        list_of_lists = True
    else:
        filelist1 = [filelist1]

    # Find the base names from the file_list_2
    base_names = []
    for file_name in filelist2:
        base_names.append(file_name.split(split_key)[0])
    # Use the set operator to get the unique base_names
    base_names = list(set(base_names))

    filtered_filelist1, filtered_filelist2 = _filter_file_list(
        filelist1, filelist2, base_names, split_key
    )

    if not list_of_lists:
        filtered_filelist1 = filtered_filelist1[0]

    return filtered_filelist1, filtered_filelist2, base_names
