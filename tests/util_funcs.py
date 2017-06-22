import tempfile

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
