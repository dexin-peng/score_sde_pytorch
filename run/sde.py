import abc
import torch
from functools import cached_property
from selector.sde_selector import register_sde

@register_sde(name='SDE')
class SDE(abc.ABC):
  def __init__(self, discretization_steps, device):
    super().__init__()
    self.device = device
    self.discretization_steps = discretization_steps

  @cached_property
  def N(self):
    return torch.tensor(self.discretization_steps, device=self.device)

  @cached_property
  @abc.abstractmethod
  def T(self): pass

  @abc.abstractmethod
  def current_drift_and_diffusion(self, x, t):
    """
    For the SDE: dx = f(x, t) dt + g(x, t) dW
    Return the current drift coefficient f(x, t) and the current diffusion coefficient g(x, t) for the SDE
    """
    pass

  @abc.abstractmethod
  def current_perturbation_kernel(self, x, t):
    """
    Return the mean and standard deviation of process xt at time t given r.v. x0 = samples x0
    Terminal distribution at t=T must be Gaussian
    """
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    pass

  @abc.abstractmethod
  def prior_logp_0t(self, z): pass

  def discretize(self, x, t):
    """
    For the SDE: dx = f(x, t) dt + g(x, t) dW
    Discretize the SDE in Euler-Maruyama form
    x_k 
    = x_{k-1} + f_k(x_{k-1}, t_{k-1}) dt + g_k(x_{k-1}, t_{k-1}) * sqrt(|dt|) * z, where z ~ N(0, 1)
    = x_{k-1} + df + dg * z

    Return
    df at time t
    dg at time t

    In Euler-Maruyama form
    df(t) = f_k(x_{k-1}, t_{k-1}) dt
    dg(t) = g_k(x_{k-1}, t_{k-1}) * sqrt(|dt|)
    """
    dt = 1 / self.N
    f, g = self.current_drift_and_diffusion(x, t)
    return f * dt, g * torch.sqrt(torch.tensor(dt, device=t.device))[:, None, None, None]

  def reverse(self, score_fn):
    forward_self = self
    class RSDE(self.__class__):
      def __init__(self, score_fn, *args, **kwargs):
          super().__init__(forward_self.sigma_min.item(), forward_self.sigma_max.item(), forward_self.N.item(), forward_self.device)
          self.score_fn = score_fn

      def reverse_drift_and_diffusion(self, model, rx, rt):
        """
        For the reverse SDE: dy = (f(y, T-t) - g(T-t)^2 * score(y, T-t)) (-dt) + g(T-t) dW
        rt := T-t
        rf(y, rt) := f(y, rt) - g(rt)^2 * score(y, rt)
        rg(rt) := g(rt)
        Return the current drift coefficient rf(x, t) and the current diffusion coefficient rg(x, t) for the reverse SDE
        """
        f, g = forward_self.current_drift_and_diffusion(rx, rt)
        rf = f - g ** 2 * self.score_fn(model, rx, rt)
        rg = g
        return rf, rg

      def reverse_discretize(self, model, rx, rt):
        """
        For the reverse SDE: dy = (f(y, T-t) - g(T-t)^2 * score(y, T-t)) d(T-t) + g(T-t) dW
        rt := T-t
        rf(y, rt) := f(y, rt) - g(rt)^2 * score(y, rt)
        rg(rt) := g(rt)
        Discretize the reverse SDE in the form:
        By definition, y_k = y_{k-1} + drf + drg * z, where z ~ N(0, 1)

        rf = f(rt) - g(rt) ** 2 * score(y, rt)  =>  drf = df(rt) - dg(rt) ** 2 * score(y, rt)
        rg = g(rt)  =>  drg = dg(rt)

        when k increases, time approaches 0,
        drt=-dt<0 in the reverse SDE, but W_{- \delta t} =_D W_{\delta t} by the property of Brownian motion
        """
        df, dg = forward_self.discretize(rx, rt)
        drf = df - dg ** 2 * self.score_fn(model, rx, rt)
        drg = dg
        return drf, drg
    return RSDE(score_fn)

@register_sde(name='vesde')
class VESDE(SDE):
  def __init__(self, sigma_min, sigma_max, discretization_steps, device, *args, **kwargs):
    super().__init__(discretization_steps, device)
    self.sigmas = [sigma_min, sigma_max]

  @cached_property
  def sigma_min(self):
    return torch.tensor(self.sigmas[0], device=self.device)

  @cached_property
  def sigma_max(self):
    return torch.tensor(self.sigmas[1], device=self.device)

  @cached_property
  def T(self):
    return torch.tensor(1, device=self.device)

  def current_drift_and_diffusion(self, x, t):
    """
    For the SDE: dx = f(x, t) dt + g(x, t) dW
    Return the current drift coefficient f(x, t) and the current diffusion coefficient g(x, t) for the SDE
    f(x, t) = 0
    g(x, t) = sigma_min * (sigma_max / sigma_min) ** t * sqrt(2 * (log(sigma_max) - log(sigma_min)))
    """
    f = torch.zeros_like(x)
    g = self.sigma_min * (self.sigma_max / self.sigma_min) ** t * torch.sqrt(2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min)))
    return f, g[:, None, None, None]

  def current_perturbation_kernel(self, x0, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x0
    return mean, std

  def prior_sampling(self, shape):
    noise = torch.randn(*shape, device=self.device) * self.sigma_max
    mean = torch.zeros_like(noise, device=self.device)
    mean[:, 0, :, :] = 0.4914
    mean[:, 1, :, :] = 0.4822
    mean[:, 2, :, :] = 0.4465
    return noise + mean

  def prior_logp_0t(self, z):
    shape = z.shape
    N = torch.prod(shape[1:])
    return -N / 2. * torch.log(2 * torch.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2

  @cached_property
  def discrete_sigmas(self):
    return torch.exp(torch.linspace(torch.log(self.sigma_min), torch.log(self.sigma_max), self.N, device=self.device))

  def discretize(self, x, t):
    """
    For the SDE: dx = f(x, t) dt + g(x, t) dW
    Discretize the SDE in SMLD form
    x_k = x_{k-1} + sqrt(sigma_{k}^2 - sigma_{k-1}^2) * z, where z ~ N(0, 1), sigma_k is the sigma at time t_k

    Return
    df = 0
    dg = sqrt(sigma_{k}^2 - sigma_{k-1}^2)
    """
    k = (t * (self.N - 1) / self.T).type(torch.int32)
    sigma = self.discrete_sigmas[k]
    sigma_previous = torch.where(k == 0, torch.zeros_like(sigma), self.discrete_sigmas[k - 1])
    return torch.zeros_like(x), torch.sqrt(sigma ** 2 - sigma_previous ** 2)[:, None, None, None]

class ScoreFN:
  def __init__(self, sde, continuous):
    self.sde = sde
    self.continuous = continuous

  def __call__(self, model, x, t):
    if self.continuous:
        std = self.sde.current_perturbation_kernel(torch.zeros_like(x), t)[1]
        score = model(x, std)
    else:
        score = model(x, t)
    return score
