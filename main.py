import sys
import argparse

from selector.config_selector import _CONFIGS
from utils.logger import Logger
# import warnings

# warnings.filterwarnings('error', category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description='Train or sample from diffusion model')
    parser.add_argument('--config', type=str, required=True, choices=_CONFIGS.keys(), help='Configuration name')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'sample'], help='To train model or generate samples')

    parser.add_argument('--user_logging_level', type=str, required=False, default='info', choices=['debug', 'info', 'warning', 'error'], help='Set logging level (debug, info, warning, error)')
    parser.add_argument('--training_from_scratch', action='store_true', default=False, required=False, help='To train model from scratch or continue training')
    parser.add_argument('--sampling_from_epoch', type=int, required=False, default=None, help='To sample from model with which training epoch, default is the latest training epoch')

    args = parser.parse_args()

    logger = Logger(args.user_logging_level)
    logger.debug(f"Starting {args.mode}" + f" with config: {args.config}")

    class Config(_CONFIGS[args.config]):
        def __init__(self, parse_args, logger, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.parse_args = parse_args
            self.logger = logger

            self.mode = self.parse_args.mode
            self.io.user_logging_level = self.parse_args.user_logging_level
            self.io.training_from_scratch = self.parse_args.training_from_scratch
            self.io.sampling_from_epoch = self.parse_args.sampling_from_epoch

    config = Config(args, logger)

    logger.debug("Current configuration:")
    for key, value in config.__dict__.items():
        log_message = f"{key}: {value}"
        logger.debug(log_message)

    try:
        if args.mode == 'train':
            from run.train import Trainer
            trainer = Trainer(config)
            trainer.train()
        elif args.mode == 'sample':
            from run.sample import Sampler
            sampler = Sampler(config)
            sampler.load_checkpoint()
            sampler.sample()
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    from run.sde import VESDE
    from model.optimizer import AdamOptimizer
    from model.ncsnpp import NCSNpp
    from config.ve.cifar10_nscnpp_cont import CIFAR10NSCNPPContConfig
    from model.predictors import ReverseDiffusionPredictor
    from model.correctors import NoneCorrector
    main()
