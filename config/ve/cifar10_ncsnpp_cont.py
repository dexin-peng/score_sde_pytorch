import os
from config.base_cifar10 import BaseCIFAR10Config
from config.dynamic_io import DynamicIOConfig
from selector.config_selector import register_config

@register_config(name='cifar10_ncsnpp_cont')
class CIFAR10NCSNPPContConfig(BaseCIFAR10Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sampling.method = 'pc'
        self.sampling.predictor = 'reverse_diffusion'
        self.sampling.corrector = 'langevin'
        self.sampling.eps = 1e-5
        
        self.model.name = 'ncsnpp'
        self.model.sde_type = 'vesde'
        self.model.continuous = True
        self.model.scale_by_sigma = True
        self.model.ema_rate = 0.999
        self.model.normalization = 'GroupNorm'
        self.model.nonlinearity = 'swish'
        self.model.nf = 128
        self.model.ch_mult = (1, 2, 2, 2)
        self.model.num_res_blocks = 4
        self.model.attn_resolutions = (16,)
        self.model.resamp_with_conv = True
        self.model.conditional = True
        self.model.fir = True
        self.model.fir_kernel = [1, 3, 3, 1]
        self.model.skip_rescale = True
        self.model.resblock_type = 'biggan'
        self.model.progressive = 'none'
        self.model.progressive_input = 'residual'
        self.model.progressive_combine = 'sum'
        self.model.attention_type = 'ddpm'
        self.model.init_scale = 0.0
        self.model.fourier_scale = 16
        self.model.conv_size = 3

        self.io = IOConfig()


class IOConfig(DynamicIOConfig):

    @property
    def in_dataset_path(self): return os.path.join("data", "CIFAR10")

    @property
    def in_dataset_stat_path(self): return os.path.join("data", "CIFAR10.npz")

    @property
    def in_raw_dataset_path(self): return os.path.join("data", "raw", "CIFAR10")

    @property
    def out_asset_suffix(self): return os.path.join("ve", "cifar10_ncsnpp_cont")

    @property
    def use_tensorboard(self): return True
