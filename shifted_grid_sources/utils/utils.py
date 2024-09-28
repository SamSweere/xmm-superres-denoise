


def return_grid_dir(num, res, exp, hr = False):

    if hr: 
            num2 = 0 
    else:
            num2 = 4
   
    grid_dir =  f'test_grid_f_1e-13_step_10_mult_{res}_{exp}ks_p_{num}-{num2}.fits.gz'

    return grid_dir
