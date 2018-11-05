from skimage import color, transform, io
from pathlib import Path
import numpy as np


class Processors:

    @staticmethod
    def rgba2rgb(img):
        if img.shape[-1] == 4:
            return color.rgba2rgb(img)
        else:
            return img

    @staticmethod
    def normalize_image(image_file, target_size):
        return np.expand_dims(
            transform.resize(Processors.rgba2rgb(io.imread(Path(image_file).absolute().as_posix())), target_size),
            axis=0
        )


class Tensors:

    @staticmethod
    def multiple_files(**kwargs):
        """Multiple images as inputs to predict"""
        files = kwargs['files']
        return [getattr(Processors, kwargs['preprocess'])(file, kwargs['target_size']) for file in files]

    @staticmethod
    def single_file(**kwargs):
        raise NotImplementedError('to do...')



