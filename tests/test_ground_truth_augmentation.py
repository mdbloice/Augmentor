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
from Augmentor.Operations import ZoomGroundTruth


def test_loading_ground_truth_images():
    # Create directories for the standard images and the ground truth images.
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

    # Create a pipeline, then add the ground truth image directory.
    p = Augmentor.Pipeline(standard_image_directory)
    assert len(p.augmentor_images) == len(image_names)

    # Add the ground truth directory.
    p.ground_truth(ground_truth_image_directory)

    # Check how many were found and make sure the
    # count is the same as the number of ground truth
    # images we created.
    count = 0
    for augmentor_image in p.augmentor_images:
        if augmentor_image.ground_truth is not None:
            count += 1

    assert count == len(ground_truth_images)

    # Check that each ground truth image is contained
    # in the augmentor_images list.
    stored_ground_truth_images = []
    for augmentor_image in p.augmentor_images:
        if augmentor_image.ground_truth is not None:
            stored_ground_truth_images.append(augmentor_image.ground_truth)

    for ground_truth_image in ground_truth_images:
        assert ground_truth_image in stored_ground_truth_images

    # Remove the directories that we used entirely
    shutil.rmtree(standard_image_directory)
    shutil.rmtree(ground_truth_image_directory)


def test_zoom_ground_truth_temporary_class_without_ground_truth_images():
    file_ending = "PNG"

    # Create directories for the standard images and the ground truth images.
    standard_image_directory = tempfile.mkdtemp()
    ground_truth_image_directory = tempfile.mkdtemp(prefix="ground-truth_")

    # Create images in each directory, but with the same names.
    # First create a number of image names.
    image_names = []
    num_of_images = random.randint(1, 10)
    for i in range(num_of_images):
        image_names.append("im%s.%s" % (i, file_ending))

    # Create random images, one set of 'standard' images
    # and another set of ground truth images.
    standard_images = []
    ground_truth_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(standard_image_directory), image_name)
        im.save(im_path, file_ending)
        standard_images.append(im_path)

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(ground_truth_image_directory), image_name)
        im.save(im_path, file_ending)
        ground_truth_images.append(im_path)

    # Test the functionality using the ZoomGroundTruth test
    # operation WITHOUT passing ground truth images. This is
    # to test the operation performs as expected with only
    # the standard images being passed to the operation
    #
    # Start by creating a Pipeline object.
    p = Augmentor.Pipeline(standard_image_directory)

    # Now add the operation test class manually (as there is no helper
    # function for this operation)
    p.add_operation(ZoomGroundTruth(probability=1, min_factor=1.1, max_factor=1.4))
    assert len(p.operations) == 1

    # Sample random number of times, generate, confirm presence.
    num_of_samples_to_generate = random.randint(1, 100)
    p.sample(num_of_samples_to_generate)
    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert len(generated_files) == num_of_samples_to_generate

    # Remove the directories that we used entirely
    shutil.rmtree(standard_image_directory)
    shutil.rmtree(ground_truth_image_directory)


def test_zoom_ground_truth_temporary_class():
    file_ending = "PNG"

    # Create directories for the standard images and the ground truth images.
    standard_image_directory = tempfile.mkdtemp()
    ground_truth_image_directory = tempfile.mkdtemp(prefix="ground-truth_")

    # Create images in each directory, but with the same names.
    # First create a number of image names.
    image_names = []
    num_of_images = random.randint(1, 10)
    for i in range(num_of_images):
        image_names.append("im%s.%s" % (i, file_ending))

    # Create random images, one set of 'standard' images
    # and another set of ground truth images.
    standard_images = []
    ground_truth_images = []

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(standard_image_directory), image_name)
        im.save(im_path, file_ending)
        standard_images.append(im_path)

    for image_name in image_names:
        im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))  # (80, 80) for Greyscale
        im_path = os.path.join(os.path.abspath(ground_truth_image_directory), image_name)
        im.save(im_path, file_ending)
        ground_truth_images.append(im_path)

    # Perform the operation using some ground truth images.
    p = Augmentor.Pipeline(standard_image_directory)
    p.ground_truth(ground_truth_image_directory)

    p.add_operation(ZoomGroundTruth(probability=1, min_factor=1.1, max_factor=1.5))

    num_samples = random.randint(2, 10)
    p.sample(num_samples)

    generated_files = glob.glob(os.path.join(standard_image_directory, "output/*"))
    assert (num_samples * 2) == len(generated_files)

    # Remove the directories that we used entirely
    shutil.rmtree(standard_image_directory)
    shutil.rmtree(ground_truth_image_directory)



