# Context
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# Imports
import Augmentor
import tempfile
import io
import shutil
from PIL import Image
from Augmentor import Operations
import glob

original_dimensions = (800, 800)
larger_dimensions = (1200, 1200)
smaller_dimensions = (400, 400)


def test_resize_in_memory():

    tmpdir = tempfile.mkdtemp()
    tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG')
    im = Image.new('RGB', original_dimensions)
    im.save(tmp.name, 'JPEG')

    resize = Operations.Resize(probability=1, width=larger_dimensions[0], height=larger_dimensions[1], resample_filter="BICUBIC")

    im = [im]

    im_resized = resize.perform_operation(im)
    assert im_resized[0].size == larger_dimensions

    resize_smaller = Operations.Resize(probability=1, width=smaller_dimensions[0], height=smaller_dimensions[1], resample_filter="BICUBIC")
    im_resized_smaller = resize_smaller.perform_operation(im)

    assert im_resized_smaller[0].size == smaller_dimensions

    tmp.close()


def test_resize_save_to_disk():
    tmpdir = tempfile.mkdtemp()

    n = 10
    tmpfiles = []
    for i in range(n):
        tmpfiles.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))
        im = Image.new('RGB', original_dimensions)
        im.save(tmpfiles[i].name, 'JPEG')

    p = Augmentor.Pipeline(tmpdir)
    assert len(p.augmentor_images) == n

    p.resize(probability=1, width=larger_dimensions[0], height=larger_dimensions[1])
    p.sample(n)

    generated_images = glob.glob(os.path.join(tmpdir, "output", "*.JPEG"))
    number_of_gen_images = len(generated_images)

    assert number_of_gen_images == n

    for im_path in generated_images:
        im_g = Image.open(im_path)
        assert im_g.size == larger_dimensions

    # Clean up
    for t in tmpfiles:
        t.close()

    shutil.rmtree(tmpdir)
