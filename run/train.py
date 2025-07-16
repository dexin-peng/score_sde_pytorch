import os
import tqdm
import torch
from model.model_setup import ModelSetup
from run.losses import LossFN
from model.optimizer import OptimizerFN
from torch_ema import ExponentialMovingAverage
from selector.data_selector import _DATA_LOADERS
from selector.sde_selector import _SDEs
from selector.optimizer_selector import _OPTIMIZERS
from functools import cached_property

class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.device = self.config.training.device
        self.model = ModelSetup(self.config, self.logger).model
        self.optimizer = _OPTIMIZERS(self.config)(self.model.parameters())
        self.data_loader = _DATA_LOADERS(self.config)
        self.sde = _SDEs[self.config.model.sde_type](beta_min=self.config.model.beta_min, beta_max=self.config.model.beta_max, sigma_min=self.config.model.sigma_min, sigma_max=self.config.model.sigma_max, discretization_steps=self.config.model.discretization_steps, device=self.config.training.device)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.config.model.ema_rate)
        self.optimize_fn = OptimizerFN(self.config)
        self.epoch_fn = EpochFN(self.sde, train=True, optimize_fn=self.optimize_fn, config=self.config)
        self.eval_epoch_fn = EpochFN(self.sde, train=False, optimize_fn=self.optimize_fn, config=self.config)
        self.best_evaluate_loss = self.config.training.best_evaluate_loss
        if self.config.io.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config.io.tensorboard_path)

    def train(self):
        if self.config.io.training_from_scratch or self.config.io.latest_checkpoint_file_path is None:
            self.start_epoch = 0
            self.end_epoch = self.config.training.brand_new_epochs
        elif not self.config.io.training_from_scratch and self.config.io.latest_checkpoint_file_path is not None:
            self.start_epoch = self.config.io.latest_checkpoint_epoch
            self.end_epoch = self.start_epoch + self.config.training.continue_training_epochs
            self.logger.info(f"Continuing training from epoch {self.start_epoch}")
            self._load_state()
        self._train()

    def _train(self):
        for epoch in tqdm.tqdm(range(self.start_epoch, self.end_epoch), desc="Epochs"):
            self.epoch = epoch
            self.model.train()
            self.epoch_loss = 0.0
            for batch_data, labels in self.train_loader:
                batch_data = batch_data.to(self.device)
                batch_data = self.data_loader.data_scaler(batch_data)
                loss = self.epoch_fn(self.model, self.optimizer, self.ema, epoch, batch_data)
                self.epoch_loss += loss.item()
            self.avg_loss = self.epoch_loss / len(self.train_loader)
            self._record_and_evaluate()

    def _save_state(self, epoch):
        ckpt_file_path = os.path.join(self.config.io.out_ckpt_path, f'{self.config.io.out_ckpt_filename_prefix}_{epoch}.pth')
        state_dict = {
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state_dict, ckpt_file_path)
        self.logger.info(f"Saved model to {ckpt_file_path}")
        if self.config.training.snapshot_sampling: self._snapshot_sampling()

    def _load_state(self):
        ckpt = torch.load(self.config.io.latest_checkpoint_file_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.ema.load_state_dict(ckpt['ema'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def _record_and_evaluate(self):
        if self.config.io.use_tensorboard: self.writer.add_scalar("training_loss", self.avg_loss, self.epoch)
        if self.epoch % self.config.training.log_freq == 0: self.logger.info(f"Epoch {self.epoch}/{self.end_epoch - self.start_epoch}, Loss: {self.avg_loss:.4f}")
        self._evaluate()
        if self.epoch % self.config.training.snapshot_freq == 0 or self.epoch == self.end_epoch - 1 and not self.saved and self.epoch != 0:
            self._save_state(self.epoch)
        

    def _evaluate(self):
        self.saved = False
        if self.epoch % self.config.training.eval_freq == 0:
            self.evaluate_loss = 0
            self.model.eval()
            for batch_data, labels in self.eval_loader:
                batch_data = batch_data.to(self.device)
                batch_data = self.data_loader.data_scaler(batch_data)
                with torch.no_grad():
                    loss = self.eval_epoch_fn(self.model, self.optimizer, self.ema, self.epoch, batch_data)
                    self.evaluate_loss += loss.item()
            self.evaluate_loss /= len(self.eval_loader)
            if self.config.io.use_tensorboard: self.writer.add_scalar("evaluate_loss", self.evaluate_loss, self.epoch)
            self._update_best_evaluate()
            self.model.train()

    def _update_best_evaluate(self):
        if self.epoch - self.start_epoch > self.config.training.eval_save_least_epoch and self.evaluate_loss < self.best_evaluate_loss:
            self._save_state(self.epoch)
            self.saved = True
        self.best_evaluate_loss = min(self.best_evaluate_loss, self.evaluate_loss)
        self.logger.info(f"Epoch {self.epoch}/{self.end_epoch - self.start_epoch}, Eval Loss: {self.evaluate_loss:.4f}, Best Eval Loss: {self.best_evaluate_loss:.4f}")

    @cached_property
    def train_loader(self):
        return self.data_loader.train_loader if not self.config.training.use_all_data else self.data_loader.all_loader

    @cached_property
    def eval_loader(self):
        return self.data_loader.eval_loader

    def _snapshot_sampling(self):
        from run.sample import Sampler
        self.config.sampling.batch_size = self.config.training.snapshot_batch_size
        self.config.sampling.total_samples = self.config.training.snapshot_batch_size
        self.config.sampling.eval = True
        sampler = Sampler(self.config)
        sampler.ema = self.ema
        sampler.model = self.model
        sampler.sample()


class EpochFN:
    def __init__(self, sde, train, optimize_fn, config):
        self.sde = sde
        self.train = train
        self.optimize_fn = optimize_fn
        self.config = config
        self.loss_fn = LossFN(sde, reduce_mean=self.config.training.reduce_mean, continuous=self.config.model.continuous, eps=self.config.training.eps)
    
    def __call__(self, model, optimizer, ema, epoch, batch):
        return self.epoch_fn(model, optimizer, ema, epoch, batch)

    def epoch_fn(self, model, optimizer, ema, epoch, batch):
        if self.train:
            optimizer.zero_grad()
            loss = self.loss_fn(model, batch)
            loss.backward()
            self.optimize_fn(optimizer, model.parameters(), epoch=epoch)
            ema.update()
        else:
            with ema.average_parameters():
                loss = self.loss_fn(model, batch)
        return loss
