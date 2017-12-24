# NumpyPipeline.py
# Author: vugia.truong <https://github.com/gachiemchiep>
# Licensed under the terms of the MIT Licence.
"""
The Pipeline module is the user facing API for the Augmentor package. It
contains the :class:`~Augmentor.Pipeline.Pipeline` class which is used to
create pipeline objects, which can be used to build an augmentation pipeline
by adding operations to the pipeline object.

For a good overview of how to use Augmentor, along with code samples and
example images, can be seen in the :ref:`mainfeatures` section.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *

from .Operations import *
from .ImageUtilities import scan_directory, scan, AugmentorImage
from .Pipeline import Pipeline

import os
import sys
import random
import uuid
import warnings
import numbers
import numpy as np

from tqdm import tqdm
from PIL import Image


class NumpyPipeline(Pipeline):
    """
    The NumpyPipeline class handles the creation of augmentation pipelines
    and the generation of augmented data by applying operations to
    this pipeline.
    """

    # Some class variables we use often
    _probability_error_text = "The probability argument must be between 0 and 1."
    _threshold_error_text = "The value of threshold must be between 0 and 255."
    _valid_formats = ["PNG", "BMP", "GIF", "JPEG"]
    _legal_filters = ["NEAREST", "BICUBIC", "ANTIALIAS", "BILINEAR"]

    def __init__(self, images=None, labels=None):
        """
        Init NumpyPipeline
        :param images: List of numpy array of image
        :param labels: List of correspoding label of image
        """

        images_count = len(images)
        labels_count = len(labels)

        # Check input image depth
        for idx, image in enumerate(images):
            channel = np.shape(image)[2]
            if channel != 3:
                sys.stdout.write("Channel of %d sample does not match : %d instead of %d . Remove it " %
                     (idx, channel, 3))
                images.pop(idx)
                labels.pop(idx)

        if images_count != labels_count:
            raise Exception("Number of input images and labels does not match : %d vs %d" % (arrays, labels_count))

        self.images = images
        self.labels = labels
        self.operations = []

    def sample(self, n):
        """
        Generate :attr:`n` number of samples from the current pipeline.

        This function generate samples from the NumpyPipeline, 
        using a list of image (numpy array) and a corresponding list of label which 
        were defined during instantiation. 
        For each image with size (w, h, d) a new (w, h, d, n) numpy array is generated.

        :param n: The number of new samples to produce.
        :type n: Integer
        :return: image_samples_all: rendered images
        :type n: List
        :return: label_samples_all: list of image's label
        :type n: List
        """
        if len(self.operations) == 0:
            raise IndexError("There are no operations associated with this pipeline.")

        labels_count = len(self.labels)
        samples_total = n * labels_count
        progress_bar = tqdm(total=samples_total, desc="Executing Pipeline", unit=' Samples', leave=False)

        image_samples_all = []
        label_samples_all = []

        for idx, image in enumerate(self.images):
            sample_count = 0

            width, height, depth = np.shape(image)
            image_samples = np.zeros((width, height, depth, n), dtype=np.uint8)
            while sample_count < n:
                image_samples[:, :, :, sample_count] = self._execute_with_array(image)
                sample_count += 1
                progress = idx * labels_count + sample_count
                progress_bar.set_description("Processing %d in total %d" % (progress, samples_total))
                progress_bar.update(1)


            image_samples_all.append(image_samples)
            label_samples_all = self.labels[idx] * n


        progress_bar.close()
        return image_samples_all, label_samples_all




