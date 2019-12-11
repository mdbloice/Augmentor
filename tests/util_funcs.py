import os
import sys
import tempfile
import io
import shutil
import numpy as np

from PIL import Image


def create_colour_temp_image(size, file_format):
    tmpdir = tempfile.mkdtemp()
    tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG')

    im = Image.fromarray(np.uint8(np.random.rand(800, 800, 3) * 255))
    im.save(tmp.name, file_format)

    return tmp, tmpdir


def create_greyscale_temp_image(size, file_format):
    tmpdir = tempfile.mkdtemp()
    tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG')

    im = Image.fromarray(np.uint8(np.random.rand(800, 800) * 255))
    im.save(tmp.name, file_format)

    return tmp, tmpdir


def create_sub_folders(number_of_sub_folders, number_of_images):

    parent_temp_directory = tempfile.mkdtemp()

    temp_directories = []
    temp_files = []

    for x in range(number_of_sub_folders):
        sub_temp_directory = tempfile.mkdtemp(dir=parent_temp_directory)
        temp_directories.append(sub_temp_directory)
        for y in range(number_of_images):
            temp_file = tempfile.NamedTemporaryFile(dir=sub_temp_directory, suffix='.JPEG')
            im_array = Image.fromarray(np.uint8(np.random.rand(800, 800) * 255))
            im_array.save(temp_file.name, 'JPEG')
            temp_files.append(temp_file)

    return temp_directories, temp_files, parent_temp_directory
