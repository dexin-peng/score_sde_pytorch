"""
Mainly based on https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py
"""

import abc
import torch
from selector.pc_selector import register_predictor

class Predictor(abc.ABC):
  def __init__(self, sde, score_fn):
    super().__init__()
    self.sde = sde
    self.rsde = sde.reverse(score_fn)
    self.score_fn = score_fn

  @abc.abstractmethod
  def run(self, x, t):
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn):
    super().__init__(sde, score_fn)

  def run(self, model, x, t):
    dt = -1. / self.rsde.discretization_steps
    drift, diffusion = self.rsde.reverse_drift_and_diffusion(model, x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion * torch.sqrt(torch.tensor(-dt, device=x.device)) * torch.randn_like(x)
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn):
    super().__init__(sde, score_fn)

  def run(self, model, rx_k, rt):
    drf, drg = self.rsde.reverse_discretize(model, rx_k, rt)
    rx_kp1_no_noise = rx_k - drf
    rx_kp1 = rx_kp1_no_noise + drg * torch.randn_like(rx_k)
    return rx_kp1, rx_kp1_no_noise

@register_predictor(name='none')
class NonePredictor(Predictor):

  def __init__(self, sde, score_fn):
    pass

  def run(self, model, x, t):
    return x, x