import torch
from selector.optimizer_selector import register_optimizer

@register_optimizer(name='Adam')
class AdamOptimizer:
    def __init__(self, config):
        self.optimizer = config.optim.optimizer
        self.lr = config.optim.lr
        self.beta1 = config.optim.beta1
        self.eps = config.optim.eps
        self.weight_decay = config.optim.weight_decay

    def __call__(self, model_parameters):
        return torch.optim.Adam(model_parameters, lr=self.lr, betas=(self.beta1, 0.999), eps=self.eps, weight_decay=self.weight_decay)

class OptimizerFN:
    def __init__(self, config):
        self.config = config

    def __call__(self, optimizer, model_parameters, epoch):
        return self.optimization_fn(optimizer, model_parameters, epoch)
    
    def optimization_fn(self, optimizer, model_parameters, epoch):
        if self.config.optim.warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = self.config.optim.lr * torch.minimum(torch.tensor(epoch) / self.config.optim.warmup, torch.tensor(1.0))
        if self.config.optim.grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=self.config.optim.grad_clip)
            optimizer.step()
