import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import tempfile
import shutil
from PIL import Image
from Augmentor import Operations


def rotate_images(tmpdir, rot):
    original_dimensions = (800, 800)

    im_tmp = tmpdir.mkdir("subfolder").join('test.JPEG')
    im = Image.new('RGB', original_dimensions)
    im.save(str(im_tmp), 'JPEG')

    r = Operations.Rotate(probability=1, rotation=rot)
    im = [im]
    im_r = r.perform_operation(im)

    assert im_r is not None
    assert im_r[0].size == original_dimensions


def test_rotate_images_90(tmpdir):
    rotate_images(tmpdir, 90)


def test_rotate_images_180(tmpdir):
    rotate_images(tmpdir, 180)


def test_rotate_images_270(tmpdir):
    rotate_images(tmpdir, 270)


def test_rotate_images_custom_temp_files():

    original_dimensions = (800, 800)

    tmpdir = tempfile.mkdtemp()
    tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG')
    im = Image.new('RGB', original_dimensions)
    im.save(tmp.name, 'JPEG')

    r = Operations.Rotate(probability=1, rotation=90)
    im = [im]
    im_r = r.perform_operation(im)

    assert im_r is not None
    assert im_r[0].size == original_dimensions

    tmp.close()
    shutil.rmtree(tmpdir)
