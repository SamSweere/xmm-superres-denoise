import piq
import torch
from utils.ssim import ssim as get_ssim
from utils.ssim import ms_ssim as get_ms_ssim


class LossFunctionHandler:
    def __init__(self, data_scaling):
        # Normalization based on val_sweep set of logical-sweep-80 epoch 49, batch_size = 4, filters = 32, residual_blocks = 4, learning_rate=0.00005
        # first_epoch = {
        #     "l1": 0.00001247,
        #     "poisson": 0.000333,
        #     "psnr": 28.644,
        #     "ssim": 0.7027,
        #     "ms_ssim": 0.93957
        # }

        # First epoch based on testing an untrained model with the same parameters as the above: winter-serenity-88
        # TODO: this does not work well
        # first_epoch = {
        #     "l1": 0.00009351,
        #     "poisson": 0.001509,
        #     "psnr": 14.056,
        #     "ssim": 0.1009,
        #     "ms_ssim": 0.3597
        # }
        #
        # first_epoch = {
        #     "l1": 1.0,
        #     "poisson": 1.0,
        #     "psnr": 0.0,
        #     "ssim": 0.0,
        #     "ms_ssim": 0.0
        # }

        # The scaling and the scaled loss functions
        # These are based on randomly initialized untrained models
        zero_epoch = {
            "linear": {
                "l1": 0.05746,
                "poisson": 0.3323,
                "psnr": 22.189,
                "ssim": 0.3856,
                "ms_ssim": 0.6093
            },
            "sqrt": {
                "l1": 0.1573,
                "poisson": 0.5002,
                "psnr": 14.761,
                "ssim": 0.1362,
                "ms_ssim": 0.5425
            },
            "asinh": {
                "l1": 0.2573,
                "poisson": 2.801,
                "psnr": 10.464,
                "ssim": 0.06155,
                "ms_ssim": 0.2081
            },
            "log": {
                "l1": 0.3528,
                "poisson": 3.167,
                "psnr": 7.817,
                "ssim": 0.05174,
                "ms_ssim": 0.3088
            },
        }

        # Epoch 38 (training was not completely stable for all runs, probably due to the ADAM optimiser) of the runs: silver-butterfly0=-101, graceful-flower-100, cool-universe-99 and stellar-cloud-98
        last_epoch = {
            "linear": {
                "l1": 0.02097,
                "poisson": 0.1804,
                "psnr": 30.565,
                "ssim": 0.7218,
                "ms_ssim": 0.96
            },
            "sqrt": {
                "l1": 0.05374,
                "poisson": 0.4187,
                "psnr": 22.977,
                "ssim": 0.4621,
                "ms_ssim": 0.874
            },
            "asinh": {
                "l1": 0.08037,
                "poisson": 0.5223,
                "psnr": 19.52,
                "ssim": 0.3662,
                "ms_ssim": 0.8258
            },
            "log": {
                "l1": 0.1072,
                "poisson": 0.6567,
                "psnr": 16.838,
                "ssim": 0.3446,
                "ms_ssim": 0.7982
            },
        }

        #
        # last_epoch = {
        #     "l1": 0.00001171,
        #     "poisson": 0.0003323,
        #     "psnr": 29.342,
        #     "ssim": 0.7269,
        #     "ms_ssim": 0.9502
        # }

        # # Calculate the deltas between the start and end of training, this will be used to scale the loss functions
        # # with respect to each other
        # delta_loss = {}
        # for key in first_epoch.keys():
        #     delta_loss[key] = abs(last_epoch[key] - first_epoch[key])

        # We target the loss to be descending starting around 1.0 and ending around 0.0

        self.__scaled_loss_fs = {}

        # L1 loss
        l1_scaling, l1_correction = self.__get_scaling(x1=zero_epoch[data_scaling]['l1'],
                                                       x2=last_epoch[data_scaling]['l1'])
        self.__scaled_loss_fs['l1'] = lambda x, y: l1_scaling * torch.nn.L1Loss()(x, y) + l1_correction

        # Poisson loss
        poisson_scaling, poisson_correction = self.__get_scaling(x1=zero_epoch[data_scaling]['poisson'],
                                                                 x2=last_epoch[data_scaling]['poisson'])
        self.__scaled_loss_fs['poisson'] = lambda x, y: poisson_scaling * torch.nn.PoissonNLLLoss(log_input=False)(x, y) \
                                                        + poisson_correction

        # PSNR loss
        # psnr is a rising metric, therefore inverse the scale
        psnr_scaling, psnr_correction = self.__get_scaling(x1=zero_epoch[data_scaling]['psnr'],
                                                           x2=last_epoch[data_scaling]['psnr'])
        self.__scaled_loss_fs['psnr'] = lambda x, y: psnr_scaling * piq.psnr(x=x, y=y, data_range=1.0) + psnr_correction

        # SSIM settings
        winsize = 13
        sigma = 2.5
        ms_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        K = (0.01, 0.05)

        # ssim is a rising metric, therefore inverse the scale

        # SSIM loss
        ssim_scaling, ssim_correction = self.__get_scaling(x1=zero_epoch[data_scaling]['ssim'],
                                                           x2=last_epoch[data_scaling]['ssim'])
        self.__scaled_loss_fs['ssim'] = lambda x, y: ssim_scaling * \
                                                     get_ssim(X=x, Y=y, win_size=winsize, win_sigma=sigma,
                                                              data_range=1.0, K=K)[0] + ssim_correction

        # MS_SSIM loss
        ms_ssim_scaling, ms_ssim_correction = self.__get_scaling(x1=zero_epoch[data_scaling]['ms_ssim'],
                                                                 x2=last_epoch[data_scaling]['ms_ssim'])
        self.__scaled_loss_fs['ms_ssim'] = lambda x, y: ms_ssim_scaling * get_ms_ssim(X=x, Y=y, win_size=winsize,
                                                                                      win_sigma=sigma,
                                                                                      data_range=1.0,
                                                                                      weights=ms_weights,
                                                                                      K=K) + ms_ssim_correction

    def __get_scaling(self, x1, x2, y1=1.0, y2=0.0):
        # Based on linear formula y=ax+b
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        return a, b

    def get_loss_f(self, l1_p, poisson_p, psnr_p, ssim_p, ms_ssim_p):
        # Loss the loss p give the relative percentages
        sum_total = l1_p + poisson_p + psnr_p + ssim_p + ms_ssim_p

        # Add only losses that have a factor, this prevents the loss metric calculation to then be multplied by 0, wasting compute time
        loss_names = []
        loss_scaling = []
        if l1_p > 0.0:
            loss_names.append('l1')
            loss_scaling.append(l1_p/sum_total)

        if poisson_p > 0.0:
            loss_names.append('poisson')
            loss_scaling.append(poisson_p/sum_total)

        if psnr_p > 0.0:
            loss_names.append('psnr')
            loss_scaling.append(psnr_p/sum_total)

        if ssim_p > 0.0:
            loss_names.append('ssim')
            loss_scaling.append(ssim_p/sum_total)

        if ms_ssim_p > 0.0:
            loss_names.append('ms_ssim')
            loss_scaling.append(ms_ssim_p/sum_total)

        combined_loss_s = ''
        for scaling, loss_name in zip(loss_scaling, loss_names):
            if combined_loss_s != '':
                combined_loss_s += ' + '
            combined_loss_s += str(round(scaling, 4)) + "*" + "scaled_" + loss_name
        print("combined_loss_f =", combined_loss_s)

        # Combine the loss functions
        combined_loss_f = lambda x, y: sum(list(scaling * self.__scaled_loss_fs[loss_name](x, y) for scaling, loss_name in zip(loss_scaling, loss_names)))
        return combined_loss_f


if __name__ == '__main__':
    LossFunctionHandler()
