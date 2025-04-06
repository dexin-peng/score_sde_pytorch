import ml_collections

class BaseCIFAR10Config(ml_collections.ConfigDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_classes = 10

        self.data = ml_collections.ConfigDict()
        self.data.name = "CIFAR10"
        self.data.image_size = 32
        self.data.color_channels = 3
        self.data.num_workers = 4

        self.data.random_flip = True
        self.data.centered = False
        self.data.uniform_dequantization = False

        self.model = ml_collections.ConfigDict()

        self.model.sigma_min = 0.01
        self.model.sigma_max = 50
        self.model.num_scales = 1000
        self.model.discretization_steps = 1000
        self.model.beta_min = 0.1
        self.model.beta_max = 20.
        self.model.dropout = 0.1
        self.model.embedding_type = 'fourier'
        self.model.device = "cuda"

        self.training = ml_collections.ConfigDict()
        self.training.batch_size = 128
        self.training.device = "cuda"
        self.training.brand_new_epochs = 2000
        self.training.continue_training_epochs = 8000
        self.training.use_all_data = False

        self.training.snapshot_freq = 100
        self.training.snapshot_batch_size = 36
        self.training.eps = 1e-5
        self.training.log_freq = 1
        self.training.eval_freq = 5
        self.training.eval_save_least_epoch = 50
        self.training.best_evaluate_loss = 150
        self.training.snapshot_sampling = True
        self.training.reduce_mean = False


        self.sampling = ml_collections.ConfigDict()
        self.sampling.device = "cuda"
        # better not be different from the model.discretization_steps
        self.sampling.discretization_steps = 1000
        self.sampling.corrector_steps = 1
        self.sampling.batch_size = 1000
        self.sampling.eval = True
        self.sampling.total_samples = 60000
        self.sampling.record_freq = 100
        self.sampling.snr = 0.16
        self.sampling.noise_removal = True

        self.optim = ml_collections.ConfigDict()
        self.optim.optimizer = "Adam"
        self.optim.lr = 2e-4
        self.optim.weight_decay = 0
        self.optim.beta1 = 0.9
        self.optim.eps = 1e-8
        self.optim.warmup = 2
        self.optim.grad_clip = 1.

        self.seed = 42
