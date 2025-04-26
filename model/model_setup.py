from selector.model_selector import _MODELS
import torch

class ModelSetup:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._create_model()
        self._set_mode()
        self._log_model_info()

    def _create_model(self):
        self.model = _MODELS[self.config.model.name](self.config).to(self.config.training.device)
        self.model = torch.nn.DataParallel(self.model)

    def _log_model_info(self):
        self.logger.debug(self.model)

    def _set_mode(self):
        if self.config.mode == 'train':
            self.model.train()
        elif self.config.mode == 'sample':
            self.model.eval()
