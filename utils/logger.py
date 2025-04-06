import logging
import os
from datetime import datetime

class Logger(logging.Logger):
    def __init__(self, user_level='info'):
        super().__init__('runner')
        self.user_level = user_level
        self.setLevel(logging.DEBUG)
        self._change_handler()

    def _change_handler(self):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not os.path.exists('log'):
            os.makedirs('log')
        fh = logging.FileHandler(f'log/{timestamp}.log')
        ch = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        fh.setLevel(logging.DEBUG)
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO, 
            'warning': logging.WARNING,
            'error': logging.ERROR,
        }
        ch.setLevel(level_map.get(self.user_level, logging.INFO))
        
        self.addHandler(fh)
        self.addHandler(ch)

    def debug(self, message):
        self.log(logging.DEBUG, message)

    def info(self, message):
        self.log(logging.INFO, message)

    def warning(self, message):
        self.log(logging.WARNING, message)

    def error(self, message):
        self.log(logging.ERROR, message, exc_info=True)
        