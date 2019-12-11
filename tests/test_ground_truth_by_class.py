# In this file we will test each class in the Operations module one by one
# to ensure it works with the new ground truth functionality.

import pytest

# Context
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# Imports, some may not be needed.
import Augmentor
import tempfile
import io
import shutil
import glob
import random
import numpy as np
from PIL import Image

# Used to test the temporary ZoomGroundTruth operation.
from Augmentor.Operations import CropRandom, HSVShifting


def create_temporary_data():
    standard_image_directory = tempfile.mkdtemp()
    ground_truth_image_directory = tempfile.mkdtemp(prefix="ground-truth_")

    # Create images in each directory, but with the same names.
    # First create a number of image names.
    image_names = []
    num_of_images = random.randint(1, 10)
    for i in range(num_of_images):
        image_names.append("im%s.png" % i)

    # Create random images, one set of 'standard' images
    # and another set of ground truth images.
    standard_images = []
    ground_truth_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(standard_image_directory), image_name)
        im.save(im_path, 'PNG')
        standard_images.append(im_path)

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(ground_truth_image_directory), image_name)
        im.save(im_path, 'PNG')
        ground_truth_images.append(im_path)

    return standard_image_directory, ground_truth_image_directory


def destroy_temporary_data(*directories):

    for directory in directories:
        shutil.rmtree(directory)


def test_histogram_equalisation_ground_truth():
    standard_image_directory = tempfile.mkdtemp()
    ground_truth_image_directory = tempfile.mkdtemp(prefix="ground-truth_")

    # Create images in each directory, but with the same names.
    # First create a number of image names.
    image_names = []
    num_of_images = random.randint(1, 10)
    for i in range(num_of_images):
        image_names.append("im%s.png" % i)

    # Create random images, one set of 'standard' images
    # and another set of ground truth images.
    standard_images = []
    ground_truth_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(standard_image_directory), image_name)
        im.save(im_path, 'PNG')
        standard_images.append(im_path)

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(ground_truth_image_directory), image_name)
        im.save(im_path, 'PNG')
        ground_truth_images.append(im_path)

    num_samples = 10

    ###############################
    # SCENARIO 1
    # Standard
    ###############################
    p = Augmentor.Pipeline(standard_image_directory)
    # Add the ground truth directory.
    p.ground_truth(ground_truth_image_directory)

    p.histogram_equalisation(probability=1)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    ###############################
    # SCENARIO 2
    # Test using two operations
    # (itself twice)
    ###############################
    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)
    p.histogram_equalisation(probability=1)
    p.histogram_equalisation(probability=1)
    p.sample(num_samples)

    ###############################
    # SCENARIO 3
    # Test without any ground
    # truth data
    ###############################
    new_image_directory = tempfile.mkdtemp()
    new_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(new_image_directory), image_name)
        im.save(im_path, 'PNG')
        new_images.append(im_path)

    p = Augmentor.Pipeline(new_image_directory)
    p.histogram_equalisation(probability=1)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(new_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    shutil.rmtree(standard_image_directory)
    shutil.rmtree(ground_truth_image_directory)
    shutil.rmtree(new_image_directory)


def test_greyscale_operation_ground_truth():
    standard_image_directory = tempfile.mkdtemp()
    ground_truth_image_directory = tempfile.mkdtemp(prefix="ground-truth_")

    # Create images in each directory, but with the same names.
    # First create a number of image names.
    image_names = []
    num_of_images = random.randint(1, 10)
    for i in range(num_of_images):
        image_names.append("im%s.png" % i)

    # Create random images, one set of 'standard' images
    # and another set of ground truth images.
    standard_images = []
    ground_truth_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(standard_image_directory), image_name)
        im.save(im_path, 'PNG')
        standard_images.append(im_path)

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(ground_truth_image_directory), image_name)
        im.save(im_path, 'PNG')
        ground_truth_images.append(im_path)

    num_samples = 10

    ###############################
    # SCENARIO 1
    # Standard.
    ###############################
    p = Augmentor.Pipeline(standard_image_directory)
    # Add the ground truth directory.
    p.ground_truth(ground_truth_image_directory)

    p.greyscale(probability=1)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    ###############################
    # SCENARIO 2
    # Chain with another operation.
    ###############################
    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.greyscale(probability=1)
    p.histogram_equalisation(probability=1)

    p.sample(num_samples)

    shutil.rmtree(standard_image_directory)
    shutil.rmtree(ground_truth_image_directory)


def test_invert_operation_ground_truth():
    standard_image_directory = tempfile.mkdtemp()
    ground_truth_image_directory = tempfile.mkdtemp(prefix="ground-truth_")

    # Create images in each directory, but with the same names.
    # First create a number of image names.
    image_names = []
    num_of_images = random.randint(1, 10)
    for i in range(num_of_images):
        image_names.append("im%s.png" % i)

    # Create random images, one set of 'standard' images
    # and another set of ground truth images.
    standard_images = []
    ground_truth_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(standard_image_directory), image_name)
        im.save(im_path, 'PNG')
        standard_images.append(im_path)

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(ground_truth_image_directory), image_name)
        im.save(im_path, 'PNG')
        ground_truth_images.append(im_path)

    num_samples = 10

    ###############################
    # SCENARIO 1
    # Standard.
    ###############################
    p = Augmentor.Pipeline(standard_image_directory)
    # Add the ground truth directory.
    p.ground_truth(ground_truth_image_directory)

    p.invert(probability=1)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    ###############################
    # SCENARIO 2
    # Chain with another operation.
    ###############################
    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.invert(probability=1)
    p.greyscale(probability=1)
    p.histogram_equalisation(probability=1)

    p.sample(num_samples)

    shutil.rmtree(standard_image_directory)
    shutil.rmtree(ground_truth_image_directory)


def test_black_and_white_ground_truth():
    standard_image_directory = tempfile.mkdtemp()
    ground_truth_image_directory = tempfile.mkdtemp(prefix="ground-truth_")

    # Create images in each directory, but with the same names.
    # First create a number of image names.
    image_names = []
    num_of_images = random.randint(1, 10)
    for i in range(num_of_images):
        image_names.append("im%s.png" % i)

    # Create random images, one set of 'standard' images
    # and another set of ground truth images.
    standard_images = []
    ground_truth_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(standard_image_directory), image_name)
        im.save(im_path, 'PNG')
        standard_images.append(im_path)

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(ground_truth_image_directory), image_name)
        im.save(im_path, 'PNG')
        ground_truth_images.append(im_path)

    num_samples = 10

    ###############################
    # SCENARIO 1
    # Standard.
    ###############################
    p = Augmentor.Pipeline(standard_image_directory)
    # Add the ground truth directory.
    p.ground_truth(ground_truth_image_directory)

    p.black_and_white(probability=1, threshold=100)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    p.greyscale(probability=1)
    p.sample(10)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) * 2 == len(generated_files)

    shutil.rmtree(standard_image_directory)
    shutil.rmtree(ground_truth_image_directory)


def test_skew_ground_truth():
    standard_image_directory = tempfile.mkdtemp()
    ground_truth_image_directory = tempfile.mkdtemp(prefix="ground-truth_")

    # Create images in each directory, but with the same names.
    # First create a number of image names.
    image_names = []
    num_of_images = random.randint(1, 10)
    for i in range(num_of_images):
        image_names.append("im%s.png" % i)

    # Create random images, one set of 'standard' images
    # and another set of ground truth images.
    standard_images = []
    ground_truth_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(standard_image_directory), image_name)
        im.save(im_path, 'PNG')
        standard_images.append(im_path)

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(ground_truth_image_directory), image_name)
        im.save(im_path, 'PNG')
        ground_truth_images.append(im_path)

    num_samples = 10

    ###############################
    # SCENARIO 1
    # Standard.
    ###############################
    p = Augmentor.Pipeline(standard_image_directory)
    # Add the ground truth directory.
    p.ground_truth(ground_truth_image_directory)

    p.skew(probability=1, magnitude=0.5)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    p.greyscale(probability=1)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 4) == len(generated_files)

    shutil.rmtree(standard_image_directory)
    shutil.rmtree(ground_truth_image_directory)


def test_rotate_standard_ground_truth():
    standard_image_directory = tempfile.mkdtemp()
    ground_truth_image_directory = tempfile.mkdtemp(prefix="ground-truth_")

    # Create images in each directory, but with the same names.
    # First create a number of image names.
    image_names = []
    num_of_images = random.randint(1, 10)
    for i in range(num_of_images):
        image_names.append("im%s.png" % i)

    # Create random images, one set of 'standard' images
    # and another set of ground truth images.
    standard_images = []
    ground_truth_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(standard_image_directory), image_name)
        im.save(im_path, 'PNG')
        standard_images.append(im_path)

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(ground_truth_image_directory), image_name)
        im.save(im_path, 'PNG')
        ground_truth_images.append(im_path)

    num_samples = 10

    ###############################
    # SCENARIO 1
    # Standard.
    ###############################
    p = Augmentor.Pipeline(standard_image_directory)
    # Add the ground truth directory.
    p.ground_truth(ground_truth_image_directory)

    p.rotate_without_crop(probability=1, max_left_rotation=20, max_right_rotation=20)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    p.greyscale(probability=1)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 4) == len(generated_files)

    shutil.rmtree(standard_image_directory)
    shutil.rmtree(ground_truth_image_directory)


def test_rotate_ground_truth():
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.rotate90(probability=1)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Repeat for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)

    p.rotate90(probability=1)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_rotate_ground_truth_multiple_passes():
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.rotate90(probability=1)
    p.rotate180(probability=1)
    p.rotate270(probability=1)
    p.rotate_random_90(probability=1)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Repeat for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)

    p.rotate90(probability=1)
    p.rotate180(probability=1)
    p.rotate270(probability=1)
    p.rotate_random_90(probability=1)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_rotate_range_ground_truth():

    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Repeat for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)

    p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_flip_ground_truth():

    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.flip_left_right(probability=1)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Repeat for single images.
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)

    p.flip_left_right(probability=1)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_flip_ground_truth_multiple_passes():

    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.flip_left_right(probability=1)
    p.flip_top_bottom(probability=1)
    p.flip_random(probability=1)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Repeat for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)

    p.flip_left_right(probability=1)
    p.flip_top_bottom(probability=1)
    p.flip_random(probability=1)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_crop_ground_truth():

    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.crop_by_size(probability=1, width=10, height=10)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Repeat for single images
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)

    p.crop_by_size(probability=1, width=10, height=10)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_crop_percentage_ground_truth():

    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.crop_centre(probability=1, percentage_area=0.5)
    p.crop_random(probability=1, percentage_area=0.5)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single images (no ground truth values at all)
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)

    p.crop_centre(probability=1, percentage_area=0.5)
    p.crop_random(probability=1, percentage_area=0.5)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_crop_random_ground_truth():

    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.add_operation(CropRandom(probability=1, percentage_area=0.5))

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single images
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.add_operation(CropRandom(probability=1, percentage_area=0.5))
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_shear_ground_truth():

    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.shear(probability=1, max_shear_left=10, max_shear_right=10)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_scale_ground_truth():

    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.scale(probability=1, scale_factor=1.4)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.scale(probability=1, scale_factor=1.4)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_distort_ground_truth():
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=4)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=4)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_distort_gaussian_ground_truth():
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.gaussian_distortion(probability=1, grid_width=4, grid_height=4, magnitude=5, corner="bell", method="in", mex=1.1,
                          mey=1.1, sdx=1.1, sdy=1.1)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.gaussian_distortion(probability=1, grid_width=4, grid_height=4, magnitude=5, corner="bell", method="in", mex=1.1,
                          mey=1.1, sdx=1.1, sdy=1.1)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_zoom_ground_truth():
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.zoom(probability=1, min_factor=1.1, max_factor=1.5)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.zoom(probability=1, min_factor=1.1, max_factor=1.5)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_zoom_random_ground_truth():
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.zoom_random(probability=1, percentage_area=0.5)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.zoom_random(probability=1, percentage_area=0.5)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_hsv_shift_ground_truth():
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.add_operation(HSVShifting(probability=1, hue_shift=1.1, saturation_scale=1.1,
                                saturation_shift=1.2, value_scale=1.1, value_shift=1.1))

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.add_operation(HSVShifting(probability=1, hue_shift=1.1, saturation_scale=1.1,
                                saturation_shift=1.2, value_scale=1.1, value_shift=1.1))
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)


def test_random_erasing_ground_truth():
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.random_erasing(probability=1, rectangle_area=0.33)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)

    # Do the same for single image lists
    standard_image_directory, ground_truth_image_directory = create_temporary_data()

    num_samples = 10

    p = Augmentor.Pipeline(standard_image_directory)
    p.random_erasing(probability=1, rectangle_area=0.33)

    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert num_samples == len(generated_files)

    destroy_temporary_data(standard_image_directory, ground_truth_image_directory)
