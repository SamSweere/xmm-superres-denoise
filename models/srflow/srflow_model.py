# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan

import torch
import pytorch_lightning as pl
from torch import nn

from models.esrgan_model import GeneratorRRDB
from models.srflow.lr_scheduler import MultiStepLR_Restart
from models.srflow.modules.SRFlowNet_arch import SRFlowNet


class LitSRFlow(pl.LightningModule):
    # SRFlow model

    def __init__(self, opt, step=0):
        """
        Args:
            opt (dict): Options
            step (int) (optional): The current step, only used when resume training
        """

        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        self.step = step


        self.heats = opt['val']['heats']
        # self.n_sample = opt['val']['n_sample']
        self.hr_size = opt['hr_shape']
        self.lr_size = opt['lr_shape']

        self.opt = opt
        # train_opt = opt['train']
        opt_net = opt['network_G']

        # define network and load pretrained models
        self.netG = SRFlowNet(lr_size=self.lr_size[0], hr_size=self.hr_size[0], in_nc=opt_net['in_channels'],
                              out_nc=opt_net['out_channels'],
                              nf=opt_net['num_filters'], nb=opt_net['num_res_blocks'], scale=opt_net['upscale'], K=opt_net['flow']['K'], opt=opt,
                              step=step)

        # print network
        # self.print_network()

        # TODO: copy over resume module

        ###
        #
        # # Calculate the upscaling
        # up_scale = hr_shape[-1] / lr_shape[-1]
        #
        # if up_scale % 2 != 0:
        #     raise ValueError(
        #         f"Upsaling is not a multple of two but {up_scale}, based on in_dims {lr_shape} and out_dims{hr_shape}")
        #
        # up_scale = int(up_scale / 2)
        # # Make the model
        #
        # # Initialize generator and discriminator
        # self.generator = GeneratorRRDB(channels, num_filters=num_filters, num_res_blocks=residual_blocks, num_upsample=up_scale)
        #
        # # Loss
        # self.criterion = criterion

    def setup(self, stage):
        # Calculate the total training steps, we do this in the setup because we do not have the train_dataloader in the init
        self.train_batches = len(self.train_dataloader())
        self.total_train_steps = self.opt['epochs'] * self.train_batches

    def get_network_description(self, network): ###
        '''Get the string and total parameters of the network'''
        # if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        #     network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def print_network(self): ###
        s, n = self.get_network_description(self.netG)
        # if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
        #     net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
        #                                      self.netG.__class__.__name__)
        # else:
        net_struc_str = '{}'.format(self.netG.__class__.__name__)

        print('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        print(s)




    def forward(self, x, heat):
        # # in lightning, forward defines the prediction/inference actions

        x = self.get_sr(lq=x, heat=heat)
        return x

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None): ###
        return self.get_sr_with_z(lq, heat, seed, z, epses)[0]

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None):
        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses)
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None): ###
        if seed: torch.manual_seed(seed)
        if self.opt['network_G']['flow']['split']['enable']:
            C = self.netG.flowUpsamplerNet.C
            H = int(self.opt['scale'] * lr_shape[2] // self.netG.flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.netG.flowUpsamplerNet.scaleW)
            z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W), device=self.device) if heat > 0 else torch.zeros(
                (batch_size, C, H, W), device=self.device)
        else:
            L = self.opt['network_G']['flow']['L']
            fac = 2 ** (L - 3)
            z_size = int(self.lr_size // (2 ** (L - 3)))
            z = torch.normal(mean=0, std=heat, size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size), device=self.device)
        return z

    def _generator_loss(self, batch):
        # # It is independent of forward
        # imgs_lr, imgs_hr = batch['lr'], batch['hr']
        #
        # # Generate a high resolution image from low resolution input
        # gen_hr = self(imgs_lr)
        #
        # # Measure pixel-wise loss against ground truth
        # loss = self.criterion(gen_hr, imgs_hr)
        #
        # return loss, gen_hr, imgs_hr
        return

    def training_step(self, batch, batch_idx, optimizer_idx):
        # # training_step defined the train loop.
        #TODO: this is never triggered since the RRDB training is set to true anyway
        # train_RRDB_delay_epoch = self.opt['network_G', 'train_RRDB_delay_epoch']
        # if train_RRDB_delay_epoch is not None and self.current_epoch > int(train_RRDB_delay_epoch) \
        #         and not self.netG.RRDB_training:
        #     if self.netG.set_rrdb_training(True):
        #         self.add_optimizer_and_scheduler_RRDB(self.opt['train'])

        # self.print_rrdb_state()

        # self.netG.train()
        # self.log_dict = OrderedDict()
        # self.optimizer_G.zero_grad()

        imgs_lr, imgs_hr = batch['lr'], batch['hr']

        losses = {}
        weight_fl = self.opt['train']['weight_fl']
        if weight_fl > 0:
            z, nll, y_logits = self.netG(gt=imgs_hr, lr=imgs_lr, reverse=False)
            nll_loss = torch.mean(nll)
            losses['nll_loss'] = nll_loss * weight_fl

        weight_l1 = self.opt['train']['weight_l1']
        if weight_l1 > 0:
            z = self.get_z(heat=0, seed=None, batch_size=imgs_lr.shape[0], lr_shape=imgs_lr.shape)
            sr, logdet = self.netG(lr=imgs_lr, z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            l1_loss = (sr - imgs_hr).abs().mean()
            losses['l1_loss'] = l1_loss * weight_l1

        total_loss = sum(losses.values())


        # total_loss.backward()
        # self.optimizer_G.step()
        #
        # mean = total_loss.item()
        self.log('train/loss', total_loss, prog_bar=True)

        # TODO: add learning rate logging
        # self.log('train/lr', )


        return total_loss



        #
        # # train generator
        #
        # loss, gen_hr, imgs_hr = self._generator_loss(batch)
        # self.log('train/loss', loss, prog_bar=True)
        #
        # return loss
        # return

    def forward_batch(self, batch, batch_idx):
        imgs_lr, imgs_hr = batch['lr'], batch['hr']

        # self.netG.eval()
        gen_hr = {}
        for heat in self.heats:
            # for i in range(self.n_sample):
            z = self.get_z(heat, seed=None, batch_size=imgs_lr.shape[0], lr_shape=imgs_lr.shape)
            with torch.no_grad():
                gen_hr[heat], logdet = self.netG(lr=imgs_lr, z=z, eps_std=heat, reverse=True)

        with torch.no_grad():
            _, nll, _ = self.netG(gt=imgs_hr, lr=imgs_lr, reverse=False)

        # out_dict = {}
        # # out_dict['lr'] = imgs_lr
        # for heat in self.heats:
        #     for i in range(self.n_sample):
        #         out_dict[(heat, i)] = gen_hr[(heat, i)]

        # out_dict['hr'] = imgs_hr

        return nll.mean(), gen_hr


    def validation_step(self, batch, batch_idx):

        # imgs_lr, imgs_hr = batch['lr'], batch['hr']
        #
        # # self.netG.eval()
        # self.gen_hr = {}
        # for heat in self.heats:
        #     for i in range(self.n_sample):
        #         z = self.get_z(heat, seed=None, batch_size=imgs_lr.shape[0], lr_shape=imgs_lr.shape)
        #         with torch.no_grad():
        #             self.gen_hr[(heat, i)], logdet = self.netG(lr=imgs_lr, z=z, eps_std=heat, reverse=True)
        #
        # with torch.no_grad(): #TODO: not sure if we need this line
        #     _, nll, _ = self.netG(gt=imgs_hr, lr=imgs_lr, reverse=False)
        # # self.netG.train()
        # # return nll.mean().item()
        #
        # out_dict = {}
        # out_dict['lr'] = imgs_lr
        # for heat in self.heats:
        #     for i in range(self.n_sample):
        #         out_dict[('gen_hr', heat, i)] = self.gen_hr[(heat, i)]
        #
        # out_dict['hr'] = imgs_hr

        loss, gen_hr = self.forward_batch(batch, batch_idx)

        self.log('val/loss', loss, prog_bar=True)


        return gen_hr

    def test_step(self, batch, batch_idx):

        loss, gen_hr = self.forward_batch(batch, batch_idx)

        self.log('test/loss', loss, prog_bar=True)

        return gen_hr


    def configure_optimizers(self): ###
        # Optimizers
        wd_G = self.opt['train']['weight_decay_G']

        optim_params_RRDB = []
        optim_params_other = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            # print(k, v.requires_grad)
            if v.requires_grad:
                # optim_params_other.append(v)
                if 'RRDB' in k:
                    optim_params_RRDB.append(v)
                    # print('opt', k)
                else:
                    optim_params_other.append(v)

        # print('rrdb params', len(optim_params_RRDB))


        # optimizers = torch.optim.Adam(
        #     [
        #         {"params": optim_params_other, "lr": self.opt['train']['lr_G'], 'beta1': self.opt['train']['beta1'],
        #          'beta2': self.opt['train']['beta2'], 'weight_decay': wd_G},
        #         {"params": optim_params_RRDB, "lr": self.opt['train']['lr_G'], #TODO: does not get used I think
        #          'beta1': self.opt['train']['beta1'],
        #          'beta2': self.opt['train']['beta2'], 'weight_decay': wd_G}
        #     ],
        # )

        #TODO: this is quite different from the original implementation, could go wrong

        optimizer_SRFLOW = torch.optim.Adam(params= optim_params_other, lr= self.opt['train']['lr_G'],
                                     betas=(self.opt['train']['beta1'], self.opt['train']['beta2']),
                                     weight_decay= wd_G)

        optimizer_RRDB = torch.optim.Adam(params= optim_params_RRDB, lr= self.opt['train']['lr_G'],
                                     betas=(self.opt['train']['beta1'], self.opt['train']['beta2']),
                                     weight_decay= wd_G)




        #
        # # schedulers

        # Adapted from the options.py
        niter = self.total_train_steps
        lr_steps = [int(x * niter) for x in self.opt['train']['lr_steps_rel']]

        #
        scheduler_SRFLOW = MultiStepLR_Restart(optimizer_RRDB, lr_steps,
                                        restarts= None,
                                        weights = None,
                                        gamma=self.opt['train']['lr_gamma'],
                                        clear_state=None,
                                        lr_steps_invese=[])

        scheduler_RRDB = MultiStepLR_Restart(optimizer_SRFLOW, lr_steps,
                                        restarts= None,
                                        weights = None,
                                        gamma=self.opt['train']['lr_gamma'],
                                        clear_state=None,
                                        lr_steps_invese=[])


        # for optimizer in optimizers:
        #     self.schedulers.append(
        #         MultiStepLR_Restart(optimizer, lr_steps,
        #                             restarts= None,
        #                             weights = None,
        #                             gamma=self.opt['train']['lr_gamma'],
        #                             clear_state=None,
        #                             lr_steps_invese=[]))


        return (
            {
                "optimizer": optimizer_SRFLOW,
                "lr_scheduler": {
                    "scheduler": scheduler_SRFLOW,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "train_loss", #TODO: check this one,
                    "name": "optimizer_SRFLOW"
                }
            },
            {
                "optimizer": optimizer_RRDB,
                "lr_scheduler": {
                    "scheduler": scheduler_RRDB,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "train_loss",  # TODO: check this one
                    "name": "optimizer_RRDB"
                }
            },
        )

