import abc
import os
import glob
import re
import ml_collections
from functools import cached_property

class DynamicIOConfig(ml_collections.ConfigDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_asset_prefix(self): return "assets"

    @property
    def out_sample_suffix(self): return "sample"

    @property
    def out_stat_suffix(self): return "stat"

    @property
    def out_sample_raw_suffix(self): return "raw"

    @property
    def out_sample_filename_prefix(self): return 'sample'

    @property
    def out_ckpt_suffix(self): return 'ckpt'

    @property
    def out_ckpt_filename_prefix(self): return 'epoch'

    @property
    def tensorboard_path_suffix(self): return 'tb'

    @property
    @abc.abstractmethod
    def in_dataset_path(self): pass

    @property
    @abc.abstractmethod
    def in_dataset_stat_path(self): pass

    @property
    @abc.abstractmethod
    def in_raw_dataset_path(self): pass

    @property
    @abc.abstractmethod
    def out_asset_suffix(self): pass

    @property
    @abc.abstractmethod
    def use_tensorboard(self): pass

    @cached_property
    def out_ckpt_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_ckpt_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def out_sample_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_sample_suffix, str(self.sampling_epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def out_raw_sample_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_sample_suffix, str(self.sampling_epoch), self.out_sample_raw_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property   
    def out_stat_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_stat_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def tensorboard_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.tensorboard_path_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def get_epoch_num(filename):
        match = re.search(r'_(\d+)\.pth', filename)
        return int(match.group(1)) if match else 0
    
    @property
    def latest_checkpoint_epoch(self):
        pattern = os.path.join(self.out_ckpt_path, f'{self.out_ckpt_filename_prefix}_*.pth')
        ckpt_files = glob.glob(pattern)
        if not ckpt_files:
            return None
        latest_ckpt_file = max(ckpt_files, key=self.get_epoch_num)
        epoch_num = self.get_epoch_num(latest_ckpt_file)
        return epoch_num
 
    @property
    def latest_checkpoint_file_path(self):
        if self.latest_checkpoint_epoch is None:
            return None
        return os.path.join(self.out_ckpt_path, f'{self.out_ckpt_filename_prefix}_{self.latest_checkpoint_epoch}.pth')

    @property
    def sampling_epoch(self):
        if self.sampling_from_epoch is not None:
            return self.sampling_from_epoch
        else:
            return self.latest_checkpoint_epoch


    @cached_property
    def sampling_ckpt_file_path(self):
        if self.sampling_from_epoch is not None:
            return os.path.join(self.out_ckpt_path, f'{self.out_ckpt_filename_prefix}_{self.sampling_from_epoch}.pth')
        else:
            return self.latest_checkpoint_file_path

    def generated_sample_pt_file_path(self, start_image_count, end_image_count):
        if self.latest_generated_sample_num is None:
            return None
        return os.path.join(self.out_sample_path, f"{self.out_sample_filename_prefix}s_{start_image_count}_{end_image_count}.pt")

    def generated_sample_png_file_path(self, image_count):
        if self.latest_generated_sample_num is None:
            return None
        return os.path.join(self.out_raw_sample_path, f"{self.out_sample_filename_prefix}_{image_count:05d}.jpg")
    
    def sample_pdf_file_path(self, step):
        return os.path.join(self.out_sample_path, f"{self.out_sample_filename_prefix}_{step}.pdf")

    @staticmethod
    def get_sample_num(filename):
        match = re.search(r'_(\d+)\.jpg', filename)
        return int(match.group(1)) if match else 0

    @property
    def latest_generated_sample_num(self):
        pattern = os.path.join(self.out_raw_sample_path, f'{self.out_sample_filename_prefix}_*.jpg')
        sample_files = glob.glob(pattern)
        if not sample_files:
            return 0
        latest_sample_file = max(sample_files, key=self.get_sample_num)
        latest_sample_num = self.get_sample_num(latest_sample_file)
        return latest_sample_num
    
    @property
    def latest_generated_sample_file_path(self):
        if self.latest_generated_sample_num is None:
            return None
        return os.path.join(self.out_raw_sample_path, f'{self.out_sample_filename_prefix}_{self.latest_generated_sample_num}.jpg')

