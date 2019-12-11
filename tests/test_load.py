import pytest

# Context
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# Imports
import Augmentor
import tempfile
import io
import shutil
import glob
import random
import numpy as np

from PIL import Image

from Augmentor import ImageUtilities

from util_funcs import create_sub_folders


def test_initialise_with_no_parameters():
    p = Augmentor.Pipeline()
    assert len(p.augmentor_images) == 0
    assert isinstance(p, Augmentor.Pipeline)


def test_initialise_with_nondefault_output_directory():
    empty_temp_directory = tempfile.mkdtemp()
    output_directory = 'out'
    p = Augmentor.Pipeline(empty_temp_directory, output_directory=output_directory)
    assert os.path.exists(os.path.join(empty_temp_directory, output_directory))


def test_initialise_with_missing_folder():
    with pytest.raises(IOError):
        p = Augmentor.Pipeline("/path/does/not/exist")


def test_initialise_with_empty_folder():
    empty_temp_directory = tempfile.mkdtemp()
    p = Augmentor.Pipeline(empty_temp_directory)

    assert os.path.exists(os.path.join(empty_temp_directory, 'output'))
    assert len(p.augmentor_images) == 0


def test_initialise_with_subfolders():

    num_of_folders = 10
    num_of_images = 10

    temp_directories, temp_files, parent_temp_directory = \
        create_sub_folders(number_of_sub_folders=num_of_folders, number_of_images=num_of_images)

    assert len(temp_directories) == num_of_folders
    assert len(temp_files) == num_of_images * num_of_folders

    # Add some images in the root directory, and some folders in the sub directories,
    # they should not be found when doing the scan
    tmp_not_to_be_found = tempfile.NamedTemporaryFile(dir=parent_temp_directory, suffix='.JPEG')
    im_not_to_be_found = Image.fromarray(np.uint8(np.random.rand(800, 800) * 255))
    im_not_to_be_found.save(tmp_not_to_be_found.name, "JPEG")

    sub_temp_directory_not_to_be_found = tempfile.mkdtemp(dir=temp_directories[random.randint(0, len(temp_directories)-1)])

    # TODO: fix
    files_found = ImageUtilities.scan_directory_with_classes(parent_temp_directory)

    assert len(files_found.keys()) == num_of_folders

    image_count = 0
    for val in files_found.values():
        image_count += len(val)
        for image_path in val:
            assert os.path.isfile(image_path)

    assert image_count == num_of_folders * num_of_images

    scanned_directories = []
    glob_scanned_files = glob.glob(os.path.join(parent_temp_directory, '*'))

    for glob_scanned_file in glob_scanned_files:
        if os.path.isdir(glob_scanned_file):
            scanned_directories.append(os.path.split(glob_scanned_file)[1])

    for key in files_found.keys():
        assert key in scanned_directories
        assert os.path.exists(os.path.join(parent_temp_directory, key))

    # Tidy up and delete temporary files.
    tmp_not_to_be_found.close()
    shutil.rmtree(sub_temp_directory_not_to_be_found)

    for temp_file in temp_files:
        temp_file.close()

    for temp_directory in temp_directories:
        shutil.rmtree(temp_directory)

    shutil.rmtree(parent_temp_directory)


def test_initialise_with_ten_images():

    tmpdir = tempfile.mkdtemp()
    tmps = []

    for i in range(10):
        tmps.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))

        bytestream = io.BytesIO()

        im = Image.new('RGB', (800, 800))
        im.save(bytestream, 'JPEG')

        tmps[i].file.write(bytestream.getvalue())
        tmps[i].flush()

    p = Augmentor.Pipeline(tmpdir)
    assert len(p.augmentor_images) == len(tmps)

    # Check if we can re-read all these images using PIL.
    # This will fail for Windows, as you cannot open a file that is already open.
    if os.name != "nt":
        for i in range(len(tmps)):
            im = Image.open(p.augmentor_images[i].image_path)
            assert im is not None

    # Check if the paths found during the scan are exactly the paths
    # stored by Augmentor after initialisation
    for i in range(len(tmps)):
        p_paths = [x.image_path for x in p.augmentor_images]
        assert tmps[i].name in p_paths

    # Check if all the paths stored by the Pipeline object
    # actually exist and are valid paths
    for i in range(len(tmps)):
        assert os.path.exists(p.augmentor_images[i].image_path)

    # Close all temporary files which will also delete them automatically
    for i in range(len(tmps)):
        tmps[i].close()

    # Finally remove the directory (and everything in it) as mkdtemp does
    # not delete itself after closing automatically
    shutil.rmtree(tmpdir)


def test_dataframe_initialise_with_ten_images():
    pandas = pytest.importorskip("pandas")

    tmpdir = tempfile.mkdtemp()
    tmps = []

    for i in range(10):
        tmps.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))

        bytestream = io.BytesIO()

        im = Image.new('RGB', (800, 800))
        im.save(bytestream, 'JPEG')

        tmps[i].file.write(bytestream.getvalue())
        tmps[i].flush()

    temp_df = pandas.DataFrame(dict(path = [i.name for i in tmps],
                           cat_id = [len(i.name) for i in tmps]))

    p = Augmentor.DataFramePipeline(temp_df,
                                    image_col = 'path',
                                    category_col='cat_id')
    assert len(p.augmentor_images) == len(tmps)

    # Check if we can re-read all these images using PIL.
    # This will fail for Windows, as you cannot open a file that is already open.
    if os.name != "nt":
        for i in range(len(tmps)):
            im = Image.open(p.augmentor_images[i].image_path)
            assert im is not None

    # Check if the paths found during the scan are exactly the paths
    # stored by Augmentor after initialisation
    for i in range(len(tmps)):
        p_paths = [x.image_path for x in p.augmentor_images]
        assert tmps[i].name in p_paths

    # Check if all the paths stored by the Pipeline object
    # actually exist and are valid paths
    for i in range(len(tmps)):
        assert os.path.exists(p.augmentor_images[i].image_path)

    # Close all temporary files which will also delete them automatically
    for i in range(len(tmps)):
        tmps[i].close()

    # Finally remove the directory (and everything in it) as mkdtemp does
    # not delete itself after closing automatically
    shutil.rmtree(tmpdir)


def test_class_image_scan():
    # Some constants
    num_of_sub_dirs = random.randint(1, 10)
    num_of_im_files = random.randint(1, 10)

    output_directory = "some_folder"

    # Make an empty temporary directory
    initial_temp_directory = tempfile.mkdtemp()

    sub_dirs = []

    # Make num_of_sub_dirs subdirectories of this initial directory
    for _ in range(num_of_sub_dirs):
        sub_dirs.append(tempfile.mkdtemp(dir=initial_temp_directory))

    tmp_files = []
    image_counter = 0

    # Just to mix things up, we can create different file types
    suffix_filetypes = [('.PNG', 'PNG'), ('.JPEG', 'JPEG'), ('.GIF', 'GIF'), ('.JPG', 'JPEG'), ('.png', 'PNG'), ('.jpeg', 'JPEG'), ('.gif', 'GIF'), ('.jpg', 'JPEG')]

    # Make num_of_im_files images in each sub directory.
    for sub_dir in sub_dirs:
        for iterator in range(num_of_im_files):
            suffix_filetype = random.choice(suffix_filetypes)
            tmp_files.append(tempfile.NamedTemporaryFile(dir=os.path.abspath(sub_dir), suffix=suffix_filetype[0]))
            im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))
            im.save(tmp_files[image_counter].name, suffix_filetype[1])
            image_counter += 1

    # Make a folder within the root directory with the same name as the output directory,
    # it should be ignored and all the tests should still run fine
    os.mkdir(os.path.join(initial_temp_directory, output_directory))

    def run():
        p = Augmentor.Pipeline(initial_temp_directory, output_directory=output_directory)

        assert len(p.augmentor_images) == (num_of_sub_dirs * num_of_im_files)
        assert len(p.class_labels) == num_of_sub_dirs

        class_label_strings = [x[0] for x in p.class_labels]
        for sub_dir in sub_dirs:
            assert os.path.basename(sub_dir) in class_label_strings

        unique_class_labels = [x.class_label for x in p.augmentor_images]
        unique_class_labels = set(unique_class_labels)
        unique_class_labels = list(unique_class_labels)
        assert len(unique_class_labels) == num_of_sub_dirs

        for unique_class_label in unique_class_labels:
            assert unique_class_label in class_label_strings

        for class_label_string in class_label_strings:
            assert class_label_string in unique_class_labels

        assert set(class_label_strings) == set(unique_class_labels)

        # Count
        labels_int = [x.class_label_int for x in p.augmentor_images]
        bins = np.bincount(labels_int)
        for bin in bins:
            assert bin == num_of_im_files

        labels = [x.class_label for x in p.augmentor_images]
        for sub_dir in sub_dirs:
            assert labels.count(os.path.basename(sub_dir)) == num_of_im_files

    # Run the tests now, we will repeat later so it's been made into a func.
    run()

    # Add some extra images in places where they should not be and re-run the tests.
    temp_file_in_root_dir1 = tempfile.NamedTemporaryFile(dir=initial_temp_directory, suffix=".PNG")
    temp_file_in_root_dir2 = tempfile.NamedTemporaryFile(dir=initial_temp_directory, suffix=".PNG")

    # All tests should run exactly as before, those two files above should be ignored.
    run()

    # Sub directories in the sub directories should be ignored, so all tests should pass after
    # randomly placing a folder in any of these sud directories
    r1 = random.randint(0, len(sub_dirs)-1)
    r2 = random.randint(0, len(sub_dirs)-1)
    os.mkdir(os.path.join(initial_temp_directory, sub_dirs[r1], output_directory))
    os.mkdir(os.path.join(initial_temp_directory, sub_dirs[r2], "ignore_me"))
    temp_to_ignore = tempfile.NamedTemporaryFile(dir=os.path.join(initial_temp_directory,
                     sub_dirs[r1], output_directory), suffix=".JPEG")
    im = Image.fromarray(np.uint8(np.random.rand(80, 80, 3) * 255))
    im.save(temp_to_ignore.name, "JPEG")
    run()

    # Clean up
    for tmp_file in tmp_files:
        tmp_file.close()

    temp_file_in_root_dir1.close()
    temp_file_in_root_dir2.close()
    temp_to_ignore.close()

    shutil.rmtree(os.path.join(initial_temp_directory, sub_dirs[r1], output_directory))
    shutil.rmtree(os.path.join(initial_temp_directory, sub_dirs[r2], "ignore_me"))

    for sub_dir in sub_dirs:
        shutil.rmtree(sub_dir)

    shutil.rmtree(os.path.join(initial_temp_directory, output_directory))
    shutil.rmtree(initial_temp_directory)
