from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import Augmentor
import tempfile
import io
import shutil
import glob
import random
import numpy as np

from PIL import Image

from Augmentor import ImageUtilities


def test_image_generator_function():

    width = 80
    height = 80

    tmpdir = tempfile.mkdtemp()
    tmps = []

    for i in range(10):
        tmps.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))

        bytestream = io.BytesIO()

        im = Image.new('RGB', (width, height))
        im.save(bytestream, 'JPEG')

        tmps[i].file.write(bytestream.getvalue())
        tmps[i].flush()

    p = Augmentor.Pipeline(tmpdir)
    assert len(p.augmentor_images) == len(tmps)

    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.333)
    p.flip_top_bottom(probability=0.5)

    g = p.image_generator()

    X = next(g)

    assert X is not None

    # Close all temporary files which will also delete them automatically
    for i in range(len(tmps)):
        tmps[i].close()

    # Finally remove the directory (and everything in it) as mkdtemp does
    # not delete itself after closing automatically
    shutil.rmtree(tmpdir)


def test_keras_generator_from_disk():

    batch_size = random.randint(1, 50)
    width = 80
    height = 80

    tmpdir = tempfile.mkdtemp()
    tmps = []

    for i in range(10):
        tmps.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))

        bytestream = io.BytesIO()

        im = Image.new('RGB', (width, height))
        im.save(bytestream, 'JPEG')

        tmps[i].file.write(bytestream.getvalue())
        tmps[i].flush()

    p = Augmentor.Pipeline(tmpdir)
    assert len(p.augmentor_images) == len(tmps)

    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.333)
    p.flip_top_bottom(probability=0.5)

    g = p.keras_generator(batch_size=batch_size, image_data_format="channels_last")

    X, y = next(g)

    assert len(X) == batch_size
    assert len(X) == batch_size
    assert len(X) == len(y)

    assert np.shape(X) == (batch_size, width, height, 3)

    # Call next() more than the total number of images in the pipeline
    for i in range(20):
        X, y = next(g)
        assert len(X) == batch_size
        assert len(X) == batch_size
        assert len(X) == len(y)
        assert np.shape(X) == (batch_size, width, height, 3)

    g2 = p.keras_generator(batch_size=batch_size, image_data_format="channels_first")

    X2, y2 = next(g2)

    assert len(X2) == batch_size
    assert len(X2) == len(y2)

    assert np.shape(X2) == (batch_size, 3, width, height)

    # Close all temporary files which will also delete them automatically
    for i in range(len(tmps)):
        tmps[i].close()

    # Finally remove the directory (and everything in it) as mkdtemp does
    # not delete itself after closing automatically
    shutil.rmtree(tmpdir)


def test_generator_with_array_data():

    batch_size = random.randint(1, 100)
    width = 800
    height = 800

    image_matrix = np.zeros((100, width, height, 3), dtype='uint8')
    labels = np.zeros(100)

    p = Augmentor.Pipeline()
    p.rotate(probability=1, max_right_rotation=10, max_left_rotation=10)

    g = p.keras_generator_from_array(image_matrix, labels, batch_size=batch_size, scaled=True)

    X, y = next(g)

    assert len(X) == batch_size
    assert len(y) == batch_size

    for i in range(len(y)):
        assert y[i] == 0

    for i in range(len(X)):
        x_converted = X[i] * 255
        x_converted = x_converted.astype("uint8")
        im_pil = Image.fromarray(x_converted)
        assert im_pil is not None

    image_matrix_2d = np.zeros((100, width, height), dtype='uint8')
    labels_2d = np.zeros(100)

    p2 = Augmentor.Pipeline()
    p2.rotate(probability=0.1, max_left_rotation=5, max_right_rotation=5)

    g2 = p2.keras_generator_from_array(image_matrix_2d, labels_2d, batch_size=batch_size)

    X2, y2 = next(g2)

    assert len(X2) == batch_size
    assert len(y2) == batch_size

    for i in range(len(y2)):
        assert y2[i] == 0

    for i in range(len(X2)):
        im_pil = Image.fromarray(X2[i].reshape(width, height))
        assert im_pil is not None


def test_generator():

    tmpdir = tempfile.mkdtemp()
    tmps = []

    num_of_images = 10
    width = 800
    height = 800

    for i in range(num_of_images):
        tmps.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))

        bytestream = io.BytesIO()

        im = Image.new('RGB', (width,height))
        im.save(bytestream, 'JPEG')

        tmps[i].file.write(bytestream.getvalue())
        tmps[i].flush()

    p = Augmentor.Pipeline(tmpdir)

    # Test a generator on the same number of images in the folder.
    batch_size = len(p.augmentor_images)
    g = p.keras_generator(batch_size=batch_size)

    batch = next(g)
    # A tuple should be returned, containing the augmented images and their labels
    assert len(batch) == 2

    X = batch[0]
    y = batch[1]

    # They should be the same size/length e.g. 100 images and 100 labels
    assert len(X) == len(y)

    assert len(X) == batch_size
    assert len(y) == batch_size

    # Because we have in this case one class, y should be of shape (batch_size, 1)
    assert np.shape(y)[0] == batch_size
    assert np.shape(y)[1] == 1

    assert np.shape(X)[0] == batch_size
    assert np.shape(X)[1] == width
    assert np.shape(X)[2] == height
    assert np.shape(X)[3] == 3  # For RGB we should have 3 layers

    # All labels in y should = 0 because we only have one class.
    for label in y:
        assert label == 0

    # Close all temporary files which will also delete them automatically
    for i in range(len(tmps)):
        tmps[i].close()

    # Finally remove the directory (and everything in it) as mkdtemp does
    # not delete itself after closing automatically
    shutil.rmtree(tmpdir)


def test_generator_image_scan():

    num_of_sub_dirs = random.randint(1, 10)
    num_of_im_files = random.randint(1, 10)

    # Test with an absolute path
    output_directory = os.path.join(tempfile.mkdtemp(), "output_abs")

    # Make an empty temporary directory
    initial_temp_directory = tempfile.mkdtemp()

    sub_dirs = []

    # Make num_of_sub_dirs subdirectories of this initial directory
    for _ in range(num_of_sub_dirs):
        sub_dirs.append(tempfile.mkdtemp(dir=initial_temp_directory))

    tmp_files = []
    image_counter = 0

    # Just to mix things up, we can create different file types
    suffix_filetypes = [('.PNG', 'PNG'),
                        ('.JPEG', 'JPEG'),
                        #('.GIF', 'GIF'),
                        ('.JPG', 'JPEG'),
                        ('.png', 'PNG'),
                        ('.jpeg', 'JPEG'),
                        #('.gif', 'GIF'),
                        ('.jpg', 'JPEG')]

    # Make num_of_im_files images in each sub directory.
    for sub_dir in sub_dirs:
        for iterator in range(num_of_im_files):
            suffix_filetype = random.choice(suffix_filetypes)
            tmp_files.append(tempfile.NamedTemporaryFile(dir=os.path.abspath(sub_dir), suffix=suffix_filetype[0]))
            im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))
            im.save(tmp_files[image_counter].name, suffix_filetype[1])
            image_counter += 1

    p = Augmentor.Pipeline(initial_temp_directory, output_directory=output_directory)

    batch_size = random.randint(1, 1000)

    g = p.keras_generator(batch_size=batch_size)

    X, y = next(g)

    # The number of classes must equal the number of sub directories
    assert np.shape(y)[0] == batch_size
    assert np.shape(y)[1] == num_of_sub_dirs
    assert len(y) == batch_size

    # Call the generator again: this time the output directory will contain the
    # the created output_directory directory.
    g_2 = p.keras_generator(batch_size=batch_size)
    X_2, y_2 = next(g_2)

    assert np.shape(y_2)[0] == batch_size
    assert np.shape(y_2)[1] == num_of_sub_dirs
    assert len(y_2) == batch_size

    # Clean up
    for tmp_file in tmp_files:
        tmp_file.close()

    for sub_dir in sub_dirs:
        shutil.rmtree(sub_dir)

    shutil.rmtree(os.path.join(initial_temp_directory, output_directory))
    shutil.rmtree(initial_temp_directory)
