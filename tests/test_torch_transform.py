import pytest

# Context
import numpy as np
import PIL.Image as Image

import Augmentor


def test_torch_transform():
    torchvision = pytest.importorskip("torchvision")

    red = np.zeros([10, 10, 3], np.uint8)
    red[..., 0] = 255
    red = Image.fromarray(red)

    p = Augmentor.Pipeline()

    # include multiple transforms to test integration
    p.greyscale(probability=1)
    p.zoom(probability=1, min_factor=1.0, max_factor=1.0)
    p.rotate_random_90(probability=1)

    transforms = torchvision.transforms.Compose([
        p.torch_transform()
    ])

    assert red != transforms(red)

    # assert that all operations were correctly applied
    result = red
    for op in p.operations:
        result = op.perform_operation([result])[0]
    assert transforms(red) == result
