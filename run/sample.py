from functools import cached_property

from PIL import Image
from tqdm import tqdm
import torch
from torch_ema import ExponentialMovingAverage
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from run.sde import ScoreFN
from model.model_setup import ModelSetup
from selector.sde_selector import _SDEs
from selector.pc_selector import _CORRECTORS, _PREDICTORS


class Sampler:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.device = config.sampling.device
        self.model = ModelSetup(self.config, self.logger).model
        self.sde = _SDEs[self.config.model.sde_type](beta_min=self.config.model.beta_min, beta_max=self.config.model.beta_max, sigma_min=self.config.model.sigma_min, sigma_max=self.config.model.sigma_max, discretization_steps=self.config.model.discretization_steps, device=self.config.sampling.device)
        self.score_fn = ScoreFN(self.sde, self.config.model.continuous)
        self.predictor = _PREDICTORS[self.config.sampling.predictor.lower()](self.sde, self.score_fn).run
        self.corrector = _CORRECTORS[self.config.sampling.corrector.lower()](self.sde, self.score_fn, self.config.sampling.snr, self.config.sampling.corrector_steps).run
        with torch.no_grad(): self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.config.model.ema_rate)

    def sample(self):
        self.iter_num = 0
        self.logger.info(f"Sampling total {self.total_samples} samples; Already generated {self.saved_samples} samples; Remaining samples: {self.remaining_samples}")
        with self.ema.average_parameters():
            while self.remaining_samples > 0:
                self._sample()
                self._save_samples_pt()
                self._save_samples_png()
                self._update_stat()

    @property
    def saved_samples(self):
        return self.config.io.latest_generated_sample_num

    @property
    def temp_batch_size(self):
        return min(self.config.sampling.batch_size, self.remaining_samples)

    @property
    def remaining_samples(self):
        return self.total_samples - self.saved_samples

    @cached_property
    def total_samples(self):
        return self.config.sampling.total_samples
        
    @cached_property
    def total_repeat_iter_num(self):
        return (self.remaining_samples // self.config.sampling.batch_size + 1) if self.remaining_samples % self.config.sampling.batch_size != 0 else self.remaining_samples // self.config.sampling.batch_size

    def _sample(self):
        with torch.no_grad():
            rx_k = self.sde.prior_sampling((self.temp_batch_size, self.config.data.color_channels, self.config.data.image_size, self.config.data.image_size))
            timesteps = torch.linspace(self.sde.T, self.config.sampling.eps, self.config.sampling.discretization_steps, device=self.config.sampling.device)
            self.logger.info(f"Sampling {self.temp_batch_size} samples with {self.config.sampling.discretization_steps} steps")
            for step in tqdm(range(self.config.sampling.discretization_steps), desc=f"Sampling {self.iter_num + 1} / {self.total_repeat_iter_num}"):
                rt = torch.ones(self.temp_batch_size, device=self.config.sampling.device) * timesteps[step]
                rx_kp1, rx_kp1_no_noise = self.corrector(self.model, rx_k, rt)
                rx_kp1, rx_kp1_no_noise = self.predictor(self.model, rx_kp1, rt)
                rx_k = rx_kp1
                self._save_samples_and_preview(step, rx_kp1_no_noise, rx_kp1)
            rx_T = self.data_inverse_scaler(rx_kp1_no_noise if self.config.sampling.noise_removal else rx_kp1)
            self.samples = rx_T

    def _save_samples_pt(self):
        self.samples = torch.clamp(self.samples.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
        self.samples = self.samples.reshape((-1, self.config.data.image_size, self.config.data.image_size, self.config.data.color_channels))
        pt_path = self.config.io.generated_sample_pt_file_path(self.saved_samples, self.saved_samples + self.temp_batch_size)
        torch.save(self.samples, pt_path)
        self.logger.info(f"Saved {self.temp_batch_size} samples to {pt_path}")

    def _save_samples_png(self):
        for _, img_array in enumerate(self.samples):
            img = Image.fromarray(img_array.numpy())
            img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1)
            img.save(img_path)
        self.logger.info(f"Converted {self.samples.shape[0]} raw samples to {self.config.io.out_raw_sample_path}")

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.config.io.sampling_ckpt_file_path}")
        loaded_state = torch.load(self.config.io.sampling_ckpt_file_path, map_location=self.device, weights_only=True)
        self.ema.load_state_dict(loaded_state['ema'])
        self.model.load_state_dict(loaded_state['model'], strict=True)

    def data_inverse_scaler(self, x):
        from selector.data_selector import BaseDataLoader
        data_loader = BaseDataLoader(self.config)
        return data_loader.data_inverse_scaler(x)

    def _save_samples_and_preview(self, step, rev_x_kp1_no_noise, rev_x_kp1):
        time_to_record = step % self.config.sampling.record_freq == 0
        final_step = step == self.config.sampling.discretization_steps - 1

        if self.iter_num == 0 and (final_step or time_to_record):
            self.samples = rev_x_kp1_no_noise if self.config.sampling.noise_removal else rev_x_kp1
            self.samples = torch.clamp(self.samples.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
            samples_to_visualize = self.samples[:min(36, len(self.samples))]
            visualize_samples_file_path = self.config.io.sample_pdf_file_path(step)
            samples_grid_format = samples_to_visualize.permute(0, 3, 1, 2)
            grid_tensor = make_grid(samples_grid_format, nrow=int(len(samples_to_visualize) ** 0.5))
            grid_image = ToPILImage()(grid_tensor.cpu())
            grid_image.save(visualize_samples_file_path, format='PDF')

    def _update_stat(self):
        self.logger.info(f"Sampling total {self.total_samples} samples; Already generated {self.saved_samples} samples; Remaining samples: {self.remaining_samples}")
        self.iter_num += 1
        if self.config.sampling.eval: self._eval_fid()

    def _eval_fid(self):
        from pytorch_fid.fid_score import calculate_fid_given_paths
        paths = [self.config.io.out_raw_sample_path, self.config.io.in_dataset_stat_path]
        self.logger.info(f"Calculating FID Score for {paths}")
        fid_value = calculate_fid_given_paths(paths, batch_size=48, device=self.config.sampling.device, dims=2048)
        self.logger.info(f"FID Score: {fid_value}")
