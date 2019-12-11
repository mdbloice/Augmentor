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

original_dimensions = (640, 480)
larger_dimensions = (1200, 1000)
smaller_dimensions = (360, 240)

def test_simple_multi_threading_example():

    tmpdir = tempfile.mkdtemp()

    n = 100
    tmpfiles = []
    for i in range(n):
        tmpfiles.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))
        im = Image.new('RGB', original_dimensions)
        im.save(tmpfiles[i].name, 'JPEG')

    p = Augmentor.Pipeline(tmpdir)
    assert len(p.augmentor_images) == n

    p.resize(probability=1, width=larger_dimensions[0], height=larger_dimensions[1])
    p.sample(n, multi_threaded=True)

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


def test_all_operations_multi_thread():
    tmpdir = tempfile.mkdtemp()

    n = 100
    tmpfiles = []
    for i in range(n):
        tmpfiles.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))
        im = Image.new('RGB', (480, 800))
        im.save(tmpfiles[i].name, 'JPEG')

    p = Augmentor.Pipeline(tmpdir)
    assert len(p.augmentor_images) == n

    p.resize(probability=1, width=300, height=300)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.flip_random(probability=0.5)

    p.sample(n, multi_threaded=True)

    generated_images = glob.glob(os.path.join(tmpdir, "output", "*.JPEG"))
    number_of_gen_images = len(generated_images)

    assert number_of_gen_images == n

    # Clean up
    for t in tmpfiles:
        t.close()

    shutil.rmtree(tmpdir)

def test_multi_threading_override():

    tmpdir = tempfile.mkdtemp()

    n = 100
    tmpfiles = []
    for i in range(n):
        tmpfiles.append(tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG'))
        im = Image.new('RGB', original_dimensions)
        im.save(tmpfiles[i].name, 'JPEG')

    p = Augmentor.Pipeline(tmpdir)
    assert len(p.augmentor_images) == n

    p.resize(probability=1, width=larger_dimensions[0], height=larger_dimensions[1])
    p.sample(n, multi_threaded=False)

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
