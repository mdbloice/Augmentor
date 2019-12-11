from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import Augmentor
from PIL import Image
import tempfile
import io
import shutil
import glob
import random
import numpy as np

import pytest


@pytest.mark.skip(reason="DataPipeline has not been written to handle this circumstance yet.")
def test_sample_with_no_masks():
    # NOTE:
    # ---
    # Temporarily disable this test as it will fail currently.
    # The DataPipeline class currently does not handle images
    # that do not have associated masks. When this functionality
    # has been added, this test will be reinstated.
    # ---

    # This is to test if the user passes data that does not contain
    # any masks, in other words a list of images rather than the
    # data structure you have in other examples in this file.
    width = 80
    height = 80

    tmpdir = tempfile.mkdtemp()
    tmps = []

    num_of_images = 10

    for i in range(num_of_images):
        tmps.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))

        bytestream = io.BytesIO()

        im = Image.new('RGB', (width, height))
        im.save(bytestream, 'JPEG')

        tmps[i].file.write(bytestream.getvalue())
        tmps[i].flush()

    # Make our data structures
    # Labels
    y = [0 if random.random() <= 0.5 else 1 for x in range(0, num_of_images)]
    # Image data
    images = [np.asarray(x) for x in tmps]

    p = Augmentor.DataPipeline(images)
    assert len(p.augmentor_images) == len(glob.glob(os.path.join(tmpdir, "*.JPEG")))

    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)

    sample_size = 100
    augmented_images = p.sample(sample_size)

    assert len(augmented_images) == sample_size

    # Close all temporary files which will also delete them automatically
    for i in range(len(tmps)):
        tmps[i].close()

    # Finally remove the directory (and everything in it) as mkdtemp does
    # not delete itself after closing automatically
    shutil.rmtree(tmpdir)


def test_sample_with_masks():
    width = 80
    height = 80

    # Original images
    tmpdir = tempfile.mkdtemp()
    tmps = []

    num_of_images = 10

    for i in range(num_of_images):
        tmps.append(tempfile.NamedTemporaryFile(dir=tmpdir, prefix=str(i), suffix='.JPEG'))

        bytestream = io.BytesIO()

        im = Image.new('RGB', (width, height))
        im.save(bytestream, 'JPEG')

        tmps[i].file.write(bytestream.getvalue())
        tmps[i].flush()

    # Mask images
    mask_tmpdir = tempfile.mkdtemp()
    mask_tmps = []

    for i in range(num_of_images):
        mask_tmps.append(tempfile.NamedTemporaryFile(dir=mask_tmpdir, prefix=str(i), suffix='.JPEG'))

        bytestream = io.BytesIO()

        im = Image.new('RGB', (width, height))
        im.save(bytestream, 'JPEG')

        mask_tmps[i].file.write(bytestream.getvalue())
        mask_tmps[i].flush()

    original_image_list = glob.glob(os.path.join(tmpdir, "*.JPEG"))
    mask_image_list = glob.glob(os.path.join(mask_tmpdir, "*.JPEG"))
    assert len(original_image_list) == len(mask_image_list)
    assert len(original_image_list) == num_of_images
    assert len(mask_image_list) == num_of_images

    collated_paths = list(zip(original_image_list, mask_image_list))  # list() required as Python 3 returns an iterator

    assert len(collated_paths) == num_of_images

    # Generate our labels and image data structure
    # y = [0 if random.random() <= 0.5 else 1 for x in range(0, num_of_images)]  # Random list of 0s and 1s
    image_class = 0 if random.random() <= 0.5 else 1
    y = [image_class] * num_of_images  # List of either all 0s or all 1s
    assert len(y) == num_of_images

    images = [[np.asarray(Image.open(im)) for im in im_list] for im_list in collated_paths]
    assert len(images) == num_of_images

    p = Augmentor.DataPipeline(images, y)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)

    sample_size = 10
    augmented_images, augmented_labels = p.sample(sample_size)

    assert len(augmented_images) == sample_size
    assert len(augmented_labels) == sample_size

    print(augmented_labels)
    for i in range(0, len(augmented_labels)):
        assert augmented_labels[i] == image_class

    for im_list in augmented_images:
        for im in im_list:
            pil_image_from_array = Image.fromarray(im)
            assert pil_image_from_array is not None

    # Now without labels
    p = Augmentor.DataPipeline(images)
    p.zoom_random(probability=1, percentage_area=0.5)

    augmented_images_no_labels = p.sample(sample_size)
    assert len(augmented_images_no_labels) == sample_size

    for im_list_no_labels in augmented_images_no_labels:
        for im in im_list_no_labels:
            pil_image_from_array_no_lbl = Image.fromarray(im)
            assert pil_image_from_array_no_lbl is not None

    # Close all temporary files which will also delete them automatically
    for i in range(len(tmps)):
        tmps[i].close()

    for i in range(len(tmps)):
        mask_tmps[i].close()

    # Finally remove the directory (and everything in it) as mkdtemp does
    # not delete itself after closing automatically
    shutil.rmtree(tmpdir)
    shutil.rmtree(mask_tmpdir)
