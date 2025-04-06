import torch
from run.sde import ScoreFN

class LossFN:
    def __init__(self, sde, reduce_mean=True, continuous=True, eps=1e-5):
        self.sde = sde
        self.reduce_mean = reduce_mean
        self.continuous = continuous
        self.eps = eps
        self.score_fn = ScoreFN(self.sde, self.continuous)

    def __call__(self, model, batch):
        return self.loss_fn(model, batch)

    def loss_fn(self, model, batch):
        t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - self.eps) + self.eps
        z = torch.randn_like(batch)
        mean, std = self.sde.current_perturbation_kernel(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = self.score_fn(model, perturbed_data, t)
        losses = torch.square(score * std[:, None, None, None] + z)
        losses = losses.reshape(losses.shape[0], -1)
        if self.reduce_mean:
            losses = torch.mean(losses, dim=-1)
        else:
            losses = 0.5 * torch.sum(losses, dim=-1)
        loss = torch.mean(losses)
        return loss
