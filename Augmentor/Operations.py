# Operations.py
# Author: Marcus D. Bloice <https://github.com/mdbloice> and contributors
# Licensed under the terms of the MIT Licence.
"""
The Operations module contains classes for all operations used by Augmentor.

The classes contained in this module are not called or instantiated directly
by the user, instead the user interacts with the
:class:`~Augmentor.Pipeline.Pipeline` class and uses the utility functions contained
there.

In this module, each operation is a subclass of type :class:`Operation`.
The :class:`~Augmentor.Pipeline.Pipeline` objects expect :class:`Operation`
types, and therefore all operations are of type :class:`Operation`, and
provide their own implementation of the :func:`~Operation.perform_operation`
function.

Hence, the documentation for this module is intended for developers who
wish to extend Augmentor or wish to see how operations function internally.

For detailed information on extending Augmentor, see :ref:`extendingaugmentor`.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from PIL import Image, ImageOps, ImageEnhance
import math
from math import floor, ceil

import numpy as np
# from skimage import img_as_ubyte
# from skimage import transform

import os
import random
import warnings

# Python 2-3 compatibility - not currently needed.
# try:
#    from StringIO import StringIO
# except ImportError:
#    from io import StringIO


class Operation(object):
    """
    The class :class:`Operation` represents the base class for all operations
    that can be performed. Inherit from :class:`Operation`, overload
    its methods, and instantiate super to create a new operation. See
    the section on extending Augmentor with custom operations at
    :ref:`extendingaugmentor`.
    """
    def __init__(self, probability):
        """
        All operations must at least have a :attr:`probability` which is
        initialised when creating the operation's object.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        self.probability = probability

    def __str__(self):
        """
        Used to display a string representation of the operation, which is
        used by the :func:`Pipeline.status` to display the current pipeline's
        operations in a human readable way.

        :return: A string representation of the operation. Can be overridden
         if required, for example as is done in the :class:`Rotate` class.
        """
        return self.__class__.__name__

    def perform_operation(self, images):
        """
        Perform the operation on the passed images. Each operation must at least
        have this function, which accepts a list containing objects of type
        PIL.Image, performs its operation, and returns a new list containing
        objects of type PIL.Image.

        :param images: The image(s) to transform.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        raise RuntimeError("Illegal call to base class.")


class HistogramEqualisation(Operation):
    """
    The class :class:`HistogramEqualisation` is used to perform histogram
    equalisation on images passed to its :func:`perform_operation` function.
    """
    def __init__(self, probability):
        """
        As there are no further user definable parameters, the class is
        instantiated using only the :attr:`probability` argument.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        Operation.__init__(self, probability)

    def perform_operation(self, images):
        """
        Performs histogram equalisation on the images passed as an argument
        and returns the equalised images. There are no user definable
        parameters for this method.

        :param images: The image(s) on which to perform the histogram
         equalisation.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        # If an image is a colour image, the histogram will
        # will be computed on the flattened image, which fires
        # a warning.
        # We may want to apply this instead to each colour channel.
        def do(image):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ImageOps.equalize(image)

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Greyscale(Operation):
    """
    This class is used to convert images into greyscale. That is, it converts
    images into having only shades of grey (pixel value intensities)
    varying from 0 to 255 which represent black and white respectively.

    .. seealso:: The :class:`BlackAndWhite` class.
    """
    def __init__(self, probability):
        """
        As there are no further user definable parameters, the class is
        instantiated using only the :attr:`probability` argument.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        Operation.__init__(self, probability)

    def perform_operation(self, images):
        """
        Converts the passed image to greyscale and returns the transformed
        image. There are no user definable parameters for this method.

        :param images: The image to convert to greyscale.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            return ImageOps.grayscale(image)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Invert(Operation):
    """
    This class is used to negate images. That is to reverse the pixel values
    for any image processed by it.
    """
    def __init__(self, probability):
        """
        As there are no further user definable parameters, the class is
        instantiated using only the :attr:`probability` argument.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        Operation.__init__(self, probability)

    def perform_operation(self, images):
        """
        Negates the image passed as an argument. There are no user definable
        parameters for this method.

        :param images: The image(s) to negate.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            return ImageOps.invert(image)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class BlackAndWhite(Operation):
    """
    This class is used to convert images into black and white. In other words,
    into using a 1-bit, monochrome binary colour palette. This is not to be
    confused with greyscale, where an 8-bit greyscale pixel intensity range
    is used.

    .. seealso:: The :class:`Greyscale` class.
    """
    def __init__(self, probability, threshold):
        """
        As well as the required :attr:`probability` parameter, a
        :attr:`threshold` can also be defined to define the cutoff point where
        a pixel is converted to black or white. The :attr:`threshold` defaults
        to 128 at the user-facing
        :func:`~Augmentor.Pipeline.Pipeline.black_and_white` function.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param threshold: A value between 0 and 255 that defines the cut off
         point where an individual pixel is converted into black or white.
        :type probability: Float
        :type threshold: Integer
        """
        Operation.__init__(self, probability)
        self.threshold = threshold

    def perform_operation(self, images):
        """
        Convert the image passed as an argument to black and white, 1-bit
        monochrome. Uses the :attr:`threshold` passed to the constructor
        to control the cut-off point where a pixel is converted to black or
        white.

        :param images: The image to convert into monochrome.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            # An alternative would be to use
            # PIL.ImageOps.posterize(image=image, bits=1)
            # but this might be faster.
            image = ImageOps.grayscale(image)
            return image.point(lambda x: 0 if x < self.threshold else 255, '1')

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class RandomBrightness(Operation):
    """
    This class is used to random change image brightness.
    """
    def __init__(self, probability, min_factor, max_factor):
        """
        required :attr:`probability` parameter

        :func:`~Augmentor.Pipeline.Pipeline.random_brightness` function.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_factor: The value between 0.0 and max_factor
         that define the minimum adjustment of image brightness.
         The value  0.0 gives a black image,The value 1.0 gives the original image, value bigger than 1.0 gives more bright image.
        :param max_factor: A value should be bigger than min_factor.
         that define the maximum adjustment of image brightness.
         The value  0.0 gives a black image, value 1.0 gives the original image, value bigger than 1.0 gives more bright image.

        :type probability: Float
        :type max_factor: Float
        :type max_factor: Float
        """
        Operation.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def perform_operation(self, images):
        """
        Random change the passed image brightness.

        :param images: The image to convert into monochrome.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        factor = np.random.uniform(self.min_factor, self.max_factor)

        def do(image):

            image_enhancer_brightness = ImageEnhance.Brightness(image)
            return image_enhancer_brightness.enhance(factor)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class RandomColor(Operation):
    """
    This class is used to random change saturation of an image.
    """
    def __init__(self, probability, min_factor, max_factor):
        """
        required :attr:`probability` parameter

        :func:`~Augmentor.Pipeline.Pipeline.random_color` function.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_factor: The value between 0.0 and max_factor
         that define the minimum adjustment of image saturation.
         The value 0.0 gives a black and white image, value 1.0 gives the original image.
        :param max_factor: A value should be bigger than min_factor.
         that define the maximum adjustment of image saturation.
         The value 0.0 gives a black and white image, value 1.0 gives the original image.

        :type probability: Float
        :type max_factor: Float
        :type max_factor: Float
        """
        Operation.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def perform_operation(self, images):
        """
        Random change the passed image saturation.

        :param images: The image to convert into monochrome.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        factor = np.random.uniform(self.min_factor, self.max_factor)

        def do(image):

            image_enhancer_color = ImageEnhance.Color(image)
            return image_enhancer_color.enhance(factor)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class RandomContrast(Operation):
    """
    This class is used to random change contrast of an image.
    """
    def __init__(self, probability, min_factor,max_factor):
        """
        required :attr:`probability` parameter

        :func:`~Augmentor.Pipeline.Pipeline.random_contrast` function.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_factor: The value between 0.0 and max_factor
         that define the minimum adjustment of image contrast.
         The value  0.0 gives s solid grey image, value 1.0 gives the original image.
        :param max_factor: A value should be bigger than min_factor.
         that define the maximum adjustment of image contrast.
         The value  0.0 gives s solid grey image, value 1.0 gives the original image.

        :type probability: Float
        :type max_factor: Float
        :type max_factor: Float
        """
        Operation.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def perform_operation(self, images):
        """
        Random change the passed image contrast.

        :param images: The image to convert into monochrome.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        factor = np.random.uniform(self.min_factor, self.max_factor)

        def do(image):

            image_enhancer_contrast = ImageEnhance.Contrast(image)
            return image_enhancer_contrast.enhance(factor)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Skew(Operation):
    """
    This class is used to perform perspective skewing on images. It allows
    for skewing from a total of 12 different perspectives.
    """
    def __init__(self, probability, skew_type, magnitude):
        """
        As well as the required :attr:`probability` parameter, the type of
        skew that is performed is controlled using a :attr:`skew_type` and a
        :attr:`magnitude` parameter. The :attr:`skew_type` controls the
        direction of the skew, while :attr:`magnitude` controls the degree
        to which the skew is performed.

        To see examples of the various skews, see :ref:`perspectiveskewing`.

        Images are skewed **in place** and an image of the same size is
        returned by this function. That is to say, that after a skew
        has been performed, the largest possible area of the same aspect ratio
        of the original image is cropped from the skewed image, and this is
        then resized to match the original image size. The
        :ref:`perspectiveskewing` section describes this in detail.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param skew_type: Must be one of ``TILT``, ``TILT_TOP_BOTTOM``,
         ``TILT_LEFT_RIGHT``, or ``CORNER``.

         - ``TILT`` will randomly skew either left, right, up, or down.
           Left or right means it skews on the x-axis while up and down
           means that it skews on the y-axis.
         - ``TILT_TOP_BOTTOM`` will randomly skew up or down, or in other
           words skew along the y-axis.
         - ``TILT_LEFT_RIGHT`` will randomly skew left or right, or in other
           words skew along the x-axis.
         - ``CORNER`` will randomly skew one **corner** of the image either
           along the x-axis or y-axis. This means in one of 8 different
           directions, randomly.

         To see examples of the various skews, see :ref:`perspectiveskewing`.

        :param magnitude: The degree to which the image is skewed.
        :type probability: Float
        :type skew_type: String
        :type magnitude: Integer
        """
        Operation.__init__(self, probability)
        self.skew_type = skew_type
        self.magnitude = magnitude

    def perform_operation(self, images):
        """
        Perform the skew on the passed image(s) and returns the transformed
        image(s). Uses the :attr:`skew_type` and :attr:`magnitude` parameters
        to control the type of skew to perform as well as the degree to which
        it is performed.

        If a list of images is passed, they must have identical dimensions.
        This is checked when we add the ground truth directory using
        :func:`Pipeline.:func:`~Augmentor.Pipeline.Pipeline.ground_truth`
        function.

        However, if this check fails, the skew function will be skipped and
        a warning thrown, in order to avoid an exception.

        :param images: The image(s) to skew.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        # Width and height taken from first image in list.
        # This requires that all ground truth images in the list
        # have identical dimensions!
        w, h = images[0].size

        x1 = 0
        x2 = h
        y1 = 0
        y2 = w

        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        # Old implementation, remove.
        # if not self.magnitude:
        #    skew_amount = random.randint(1, max_skew_amount)
        # elif self.magnitude:
        #    max_skew_amount /= self.magnitude
        #    skew_amount = max_skew_amount

        if self.skew_type == "RANDOM":
            skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
        else:
            skew = self.skew_type

        # We have two choices now: we tilt in one of four directions
        # or we skew a corner.

        if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":

            if skew == "TILT":
                skew_direction = random.randint(0, 3)
            elif skew == "TILT_LEFT_RIGHT":
                skew_direction = random.randint(0, 1)
            elif skew == "TILT_TOP_BOTTOM":
                skew_direction = random.randint(2, 3)

            if skew_direction == 0:
                # Left Tilt
                new_plane = [(y1, x1 - skew_amount),  # Top Left
                             (y2, x1),                # Top Right
                             (y2, x2),                # Bottom Right
                             (y1, x2 + skew_amount)]  # Bottom Left
            elif skew_direction == 1:
                # Right Tilt
                new_plane = [(y1, x1),                # Top Left
                             (y2, x1 - skew_amount),  # Top Right
                             (y2, x2 + skew_amount),  # Bottom Right
                             (y1, x2)]                # Bottom Left
            elif skew_direction == 2:
                # Forward Tilt
                new_plane = [(y1 - skew_amount, x1),  # Top Left
                             (y2 + skew_amount, x1),  # Top Right
                             (y2, x2),                # Bottom Right
                             (y1, x2)]                # Bottom Left
            elif skew_direction == 3:
                # Backward Tilt
                new_plane = [(y1, x1),                # Top Left
                             (y2, x1),                # Top Right
                             (y2 + skew_amount, x2),  # Bottom Right
                             (y1 - skew_amount, x2)]  # Bottom Left

        if skew == "CORNER":

            skew_direction = random.randint(0, 7)

            if skew_direction == 0:
                # Skew possibility 0
                new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 1:
                # Skew possibility 1
                new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 2:
                # Skew possibility 2
                new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 3:
                # Skew possibility 3
                new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
            elif skew_direction == 4:
                # Skew possibility 4
                new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
            elif skew_direction == 5:
                # Skew possibility 5
                new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
            elif skew_direction == 6:
                # Skew possibility 6
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
            elif skew_direction == 7:
                # Skew possibility 7
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

        if self.skew_type == "ALL":
            # Not currently in use, as it makes little sense to skew by the same amount
            # in every direction if we have set magnitude manually.
            # It may make sense to keep this, if we ensure the skew_amount below is randomised
            # and cannot be manually set by the user.
            corners = dict()
            corners["top_left"] = (y1 - random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
            corners["top_right"] = (y2 + random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
            corners["bottom_right"] = (y2 + random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))
            corners["bottom_left"] = (y1 - random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))

            new_plane = [corners["top_left"], corners["top_right"], corners["bottom_right"], corners["bottom_left"]]

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        matrix = []

        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(original_plane).reshape(8)

        perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
        perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

        def do(image):
            return image.transform(image.size,
                                   Image.PERSPECTIVE,
                                   perspective_skew_coefficients_matrix,
                                   resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class RotateStandard(Operation):
    """
    Class to perform rotations without automatically cropping the images,
    as opposed to the :class:`RotateRange` class.

    .. seealso:: For arbitrary rotations with automatic cropping, see
     the :class:`RotateRange` class.
    .. seealso:: For 90 degree rotations, see the :class:`Rotate` class.
    """

    def __init__(self, probability, max_left_rotation, max_right_rotation, expand=False, fillcolor=None):
        """
        Documentation to appear.
        """
        Operation.__init__(self, probability)
        self.max_left_rotation = -abs(max_left_rotation)   # Ensure always negative
        self.max_right_rotation = abs(max_right_rotation)  # Ensure always positive
        self.expand = expand
        self.fillcolor = fillcolor

    def perform_operation(self, images):
        """
        Documentation to appear.

        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        random_left = random.randint(self.max_left_rotation, 0)
        random_right = random.randint(0, self.max_right_rotation)

        left_or_right = random.randint(0, 1)

        rotation = 0

        if left_or_right == 0:
            rotation = random_left
        elif left_or_right == 1:
            rotation = random_right

        def do(image):
            return image.rotate(rotation, expand=self.expand, resample=Image.BICUBIC, fillcolor=self.fillcolor)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Rotate(Operation):
    """
    This class is used to perform rotations on images in multiples of 90
    degrees. Arbitrary rotations are handled by the :class:`RotateRange`
    class.
    """

    def __init__(self, probability, rotation):
        """
        As well as the required :attr:`probability` parameter, the
        :attr:`rotation` parameter controls the rotation to perform,
        which must be one of ``90``, ``180``, ``270`` or ``-1`` (see below).

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param rotation: Controls the rotation to perform. Must be one of
         ``90``, ``180``, ``270`` or ``-1``.

         - ``90`` rotate the image by 90 degrees.
         - ``180`` rotate the image by 180 degrees.
         - ``270`` rotate the image by 270 degrees.
         - ``-1`` rotate the image randomly by either 90, 180, or 270 degrees.

        .. seealso:: For arbitrary rotations, see the :class:`RotateRange` class.

        """
        Operation.__init__(self, probability)
        self.rotation = rotation

    def __str__(self):
        return "Rotate " + str(self.rotation)

    def perform_operation(self, images):
        """
        Rotate an image by either 90, 180, or 270 degrees, or randomly from
        any of these.

        :param images: The image(s) to rotate.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        random_factor = random.randint(1, 3)

        def do(image):
            if self.rotation == -1:
                return image.rotate(90 * random_factor, expand=True)
            else:
                return image.rotate(self.rotation, expand=True)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class RotateRange(Operation):
    """
    This class is used to perform rotations on images by arbitrary numbers of
    degrees.

    Images are rotated **in place** and an image of the same size is
    returned by this function. That is to say, that after a rotation
    has been performed, the largest possible area of the same aspect ratio
    of the original image is cropped from the skewed image, and this is
    then resized to match the original image size.

    The method by which this is performed is described as follows:

    .. math::

        E = \\frac{\\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}}\\Big(X-\\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}} Y\\Big)}{1-\\frac{(\\sin{\\theta_{a}})^2}{(\\sin{\\theta_{b}})^2}}

    which describes how :math:`E` is derived, and then follows
    :math:`B = Y - E` and :math:`A = \\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}} B`.

    The :ref:`rotating` section describes this in detail and has example
    images to demonstrate this.
    """
    def __init__(self, probability, max_left_rotation, max_right_rotation):
        """
        As well as the required :attr:`probability` parameter, the
        :attr:`max_left_rotation` parameter controls the maximum number of
        degrees by which to rotate to the left, while the
        :attr:`max_right_rotation` controls the maximum number of degrees to
        rotate to the right.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param max_left_rotation: The maximum number of degrees to rotate
         the image anti-clockwise.
        :param max_right_rotation: The maximum number of degrees to rotate
         the image clockwise.
        :type probability: Float
        :type max_left_rotation: Integer
        :type max_right_rotation: Integer
        """
        Operation.__init__(self, probability)
        self.max_left_rotation = -abs(max_left_rotation)   # Ensure always negative
        self.max_right_rotation = abs(max_right_rotation)  # Ensure always positive

    def perform_operation(self, images):
        """
        Perform the rotation on the passed :attr:`image` and return
        the transformed image. Uses the :attr:`max_left_rotation` and
        :attr:`max_right_rotation` passed into the constructor to control
        the amount of degrees to rotate by. Whether the image is rotated
        clockwise or anti-clockwise is chosen at random.

        :param images: The image(s) to rotate.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        # TODO: Small rotations of 1 or 2 degrees can create black pixels
        random_left = random.randint(self.max_left_rotation, 0)
        random_right = random.randint(0, self.max_right_rotation)

        left_or_right = random.randint(0, 1)

        rotation = 0

        if left_or_right == 0:
            rotation = random_left
        elif left_or_right == 1:
            rotation = random_right

        def do(image):
            # Get size before we rotate
            x = image.size[0]
            y = image.size[1]

            # Rotate, while expanding the canvas size
            image = image.rotate(rotation, expand=True, resample=Image.BICUBIC)

            # Get size after rotation, which includes the empty space
            X = image.size[0]
            Y = image.size[1]

            # Get our two angles needed for the calculation of the largest area
            angle_a = abs(rotation)
            angle_b = 90 - angle_a

            # Python deals in radians so get our radians
            angle_a_rad = math.radians(angle_a)
            angle_b_rad = math.radians(angle_b)

            # Calculate the sins
            angle_a_sin = math.sin(angle_a_rad)
            angle_b_sin = math.sin(angle_b_rad)

            # Find the maximum area of the rectangle that could be cropped
            E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
                (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
            E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
            B = X - E
            A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

            # Crop this area from the rotated image
            # image = image.crop((E, A, X - E, Y - A))
            image = image.crop((int(round(E)), int(round(A)), int(round(X - E)), int(round(Y - A))))

            # Return the image, re-sized to the size of the image passed originally
            return image.resize((x, y), resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Resize(Operation):
    """
    This class is used to resize images by absolute values passed as parameters.
    """
    def __init__(self, probability, width, height, resample_filter):
        """
        Accepts the required probability parameter as well as parameters
        to control the size of the transformed image.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param width: The width in pixels to resize the image to.
        :param height: The height in pixels to resize the image to.
        :param resample_filter: The resample filter to use. Must be one of
         the standard PIL types, i.e. ``NEAREST``, ``BICUBIC``, ``ANTIALIAS``,
         or ``BILINEAR``.
        :type probability: Float
        :type width: Integer
        :type height: Integer
        :type resample_filter: String
        """
        Operation.__init__(self, probability)
        self.width = width
        self.height = height
        self.resample_filter = resample_filter

    def perform_operation(self, images):
        """
        Resize the passed image and returns the resized image. Uses the
        parameters passed to the constructor to resize the passed image.

        :param images: The image to resize.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            # TODO: Automatically change this to ANTIALIAS or BICUBIC depending on the size of the file
            return image.resize((self.width, self.height), eval("Image.%s" % self.resample_filter))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Flip(Operation):
    """
    This class is used to mirror images through the x or y axes.

    The class allows an image to be mirrored along either
    its x axis or its y axis, or randomly.
    """
    def __init__(self, probability, top_bottom_left_right):
        """
        The direction of the flip, or whether it should be randomised, is
        controlled using the :attr:`top_bottom_left_right` parameter.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param top_bottom_left_right: Controls the direction the image should
         be mirrored. Must be one of ``LEFT_RIGHT``, ``TOP_BOTTOM``, or
         ``RANDOM``.

         - ``LEFT_RIGHT`` defines that the image is mirrored along its x axis.
         - ``TOP_BOTTOM`` defines that the image is mirrored along its y axis.
         - ``RANDOM`` defines that the image is mirrored randomly along
           either the x or y axis.
        """
        Operation.__init__(self, probability)
        self.top_bottom_left_right = top_bottom_left_right

    def perform_operation(self, images):
        """
        Mirror the image according to the `attr`:top_bottom_left_right`
        argument passed to the constructor and return the mirrored image.

        :param images: The image(s) to mirror.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        random_axis = random.randint(0, 1)

        def do(image):
            if self.top_bottom_left_right == "LEFT_RIGHT":
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.top_bottom_left_right == "TOP_BOTTOM":
                return image.transpose(Image.FLIP_TOP_BOTTOM)
            elif self.top_bottom_left_right == "RANDOM":
                if random_axis == 0:
                    return image.transpose(Image.FLIP_LEFT_RIGHT)
                elif random_axis == 1:
                    return image.transpose(Image.FLIP_TOP_BOTTOM)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Crop(Operation):
    """
    This class is used to crop images by absolute values passed as parameters.
    """
    def __init__(self, probability, width, height, centre):
        """
        As well as the always required :attr:`probability` parameter, the
        constructor requires a :attr:`width` to control the width of
        of the area to crop as well as a :attr:`height` parameter
        to control the height of the area to crop. Also, whether the
        area to crop should be taken from the centre of the image or from a
        random location within the image is toggled using :attr:`centre`.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param width: The width in pixels of the area to crop from the image.
        :param height: The height in pixels of the area to crop from the image.
        :param centre: Whether to crop from the centre of the image or a random
         location within the image, while maintaining the size of the crop
         without cropping out of the original image's area.
        :type probability: Float
        :type width: Integer
        :type height: Integer
        :type centre: Boolean
        """
        Operation.__init__(self, probability)
        self.width = width
        self.height = height
        self.centre = centre

    def perform_operation(self, images):
        """
        Crop an area from an image, either from a random location or centred,
        using the dimensions supplied during instantiation.

        :param images: The image(s) to crop the area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        w, h = images[0].size  # All images must be the same size, so we can just check the first image in the list

        left_shift = random.randint(0, int((w - self.width)))
        down_shift = random.randint(0, int((h - self.height)))

        def do(image):

            # TODO: Fix. We may want a full crop.
            if self.width > w or self.height > h:
                return image

            if self.centre:
                return image.crop(((w/2)-(self.width/2), (h/2)-(self.height/2), (w/2)+(self.width/2), (h/2)+(self.height/2)))
            else:
                return image.crop((left_shift, down_shift, self.width + left_shift, self.height + down_shift))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class CropPercentage(Operation):
    """
    This class is used to crop images by a percentage of their area.
    """
    def __init__(self, probability, percentage_area, centre, randomise_percentage_area):
        """
        As well as the always required :attr:`probability` parameter, the
        constructor requires a :attr:`percentage_area` to control the area
        of the image to crop in terms of its percentage of the original image,
        and a :attr:`centre` parameter toggle whether a random area or the
        centre of the images should be cropped.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param percentage_area: The percentage area of the original image
         to crop. A value of 0.5 would crop an area that is 50% of the area
         of the original image's size.
        :param centre: Whether to crop from the centre of the image or
         crop a random location within the image.
        :type probability: Float
        :type percentage_area: Float
        :type centre: Boolean
        """
        Operation.__init__(self, probability)
        self.percentage_area = percentage_area
        self.centre = centre
        self.randomise_percentage_area = randomise_percentage_area

    def perform_operation(self, images):
        """
        Crop the passed :attr:`images` by percentage area, returning the crop as an
        image.

        :param images: The image(s) to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        if self.randomise_percentage_area:
            r_percentage_area = round(random.uniform(0.1, self.percentage_area), 2)
        else:
            r_percentage_area = self.percentage_area

        # The images must be of identical size, which is checked by Pipeline.ground_truth().
        w, h = images[0].size

        w_new = int(floor(w * r_percentage_area))  # TODO: Floor might return 0, so we need to check this.
        h_new = int(floor(h * r_percentage_area))

        left_shift = random.randint(0, int((w - w_new)))
        down_shift = random.randint(0, int((h - h_new)))

        def do(image):
            if self.centre:
                return image.crop(((w/2)-(w_new/2), (h/2)-(h_new/2), (w/2)+(w_new/2), (h/2)+(h_new/2)))
            else:
                return image.crop((left_shift, down_shift, w_new + left_shift, h_new + down_shift))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class CropRandom(Operation):
    """
    .. warning:: This :class:`CropRandom` class is currently not used by any
     of the user-facing functions in the :class:`~Augmentor.Pipeline.Pipeline`
     class.
    """
    def __init__(self, probability, percentage_area):
        """
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param percentage_area: The percentage area of the original image
         to crop. A value of 0.5 would crop an area that is 50% of the area
         of the original image's size.
        """
        Operation.__init__(self, probability)
        self.percentage_area = percentage_area

    def perform_operation(self, images):
        """
        Randomly crop the passed image, returning the crop as a new image.

        :param images: The image to crop.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        w, h = images[0].size

        w_new = int(floor(w * self.percentage_area))
        h_new = int(floor(h * self.percentage_area))

        random_left_shift = random.randint(0, int((w - w_new)))  # Note: randint() is from uniform distribution.
        random_down_shift = random.randint(0, int((h - h_new)))

        def do(image):
            return image.crop((random_left_shift, random_down_shift, w_new + random_left_shift, h_new + random_down_shift))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Shear(Operation):
    """
    This class is used to shear images, that is to tilt them in a certain
    direction. Tilting can occur along either the x- or y-axis and in both
    directions (i.e. left or right along the x-axis, up or down along the
    y-axis).

    Images are sheared **in place** and an image of the same size as the input
    image is returned by this class. That is to say, that after a shear
    has been performed, the largest possible area of the same aspect ratio
    of the original image is cropped from the sheared image, and this is
    then resized to match the original image size. The
    :ref:`shearing` section describes this in detail.

    For sample code with image examples see :ref:`shearing`.
    """
    def __init__(self, probability, max_shear_left, max_shear_right):
        """
        The shearing is randomised in magnitude, from 0 to the
        :attr:`max_shear_left` or 0 to :attr:`max_shear_right` where the
        direction is randomised. The shear axis is also randomised
        i.e. if it shears up/down along the y-axis or
        left/right along the x-axis.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param max_shear_left: The maximum shear to the left.
        :param max_shear_right: The maximum shear to the right.
        :type probability: Float
        :type max_shear_left: Integer
        :type max_shear_right: Integer
        """
        Operation.__init__(self, probability)
        self.max_shear_left = max_shear_left
        self.max_shear_right = max_shear_right

    def perform_operation(self, images):
        """
        Shears the passed image according to the parameters defined during
        instantiation, and returns the sheared image.

        :param images: The image to shear.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        ######################################################################
        # Old version which uses SciKit Image
        ######################################################################
        # We will use scikit-image for this so first convert to a matrix
        # using NumPy
        # amount_to_shear = round(random.uniform(self.max_shear_left, self.max_shear_right), 2)
        # image_array = np.array(image)
        # And here we are using SciKit Image's `transform` class.
        # shear_transformer = transform.AffineTransform(shear=amount_to_shear)
        # image_sheared = transform.warp(image_array, shear_transformer)
        #
        # Because of warnings
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     return Image.fromarray(img_as_ubyte(image_sheared))
        ######################################################################

        width, height = images[0].size

        # For testing.
        # max_shear_left = 20
        # max_shear_right = 20

        angle_to_shear = int(random.uniform((abs(self.max_shear_left)*-1) - 1, self.max_shear_right + 1))
        if angle_to_shear != -1: angle_to_shear += 1

        # Alternative method
        # Calculate our offset when cropping
        # We know one angle, phi (angle_to_shear)
        # We known theta = 180-90-phi
        # We know one side, opposite (height of image)
        # Adjacent is therefore:
        # tan(theta) = opposite / adjacent
        # A = opposite / tan(theta)
        # theta = math.radians(180-90-angle_to_shear)
        # A = height / math.tan(theta)

        # Transformation matrices can be found here:
        # https://en.wikipedia.org/wiki/Transformation_matrix
        # The PIL affine transform expects the first two rows of
        # any of the affine transformation matrices, seen here:
        # https://en.wikipedia.org/wiki/Transformation_matrix#/media/File:2D_affine_transformation_matrix.svg

        directions = ["x", "y"]
        direction = random.choice(directions)

        def do(image):

            # We use the angle phi in radians later
            phi = math.tan(math.radians(angle_to_shear))

            if direction == "x":
                # Here we need the unknown b, where a is
                # the height of the image and phi is the
                # angle we want to shear (our knowns):
                # b = tan(phi) * a
                shift_in_pixels = phi * height

                if shift_in_pixels > 0:
                    shift_in_pixels = math.ceil(shift_in_pixels)
                else:
                    shift_in_pixels = math.floor(shift_in_pixels)

                # For negative tilts, we reverse phi and set offset to 0
                # Also matrix offset differs from pixel shift for neg
                # but not for pos so we will copy this value in case
                # we need to change it
                matrix_offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    matrix_offset = 0
                    phi = abs(phi) * -1

                # Note: PIL expects the inverse scale, so 1/scale_factor for example.
                transform_matrix = (1, phi, -matrix_offset,
                                    0, 1, 0)

                image = image.transform((int(round(width + shift_in_pixels)), height),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                image = image.crop((abs(shift_in_pixels), 0, width, height))

                return image.resize((width, height), resample=Image.BICUBIC)

            elif direction == "y":
                shift_in_pixels = phi * width

                matrix_offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    matrix_offset = 0
                    phi = abs(phi) * -1

                transform_matrix = (1, 0, 0,
                                    phi, 1, -matrix_offset)

                image = image.transform((width, int(round(height + shift_in_pixels))),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                image = image.crop((0, abs(shift_in_pixels), width, height))

                return image.resize((width, height), resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Scale(Operation):
    """
    This class is used to increase or decrease images in size by a certain
    factor, while maintaining the aspect ratio of the original image.

    .. seealso:: The :class:`Resize` class for resizing images by
     **dimensions**, and hence will not necessarily maintain the aspect ratio.

    This function will return images that are **larger** than the input
    images.
    """
    def __init__(self, probability, scale_factor):
        """
        As the aspect ratio is always kept constant, only a
        :attr:`scale_factor` is required for scaling the image.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param scale_factor: The factor by which to scale, where 1.5 would
         result in an image scaled up by 150%.
        :type probability: Float
        :type scale_factor: Float
        """
        Operation.__init__(self, probability)
        self.scale_factor = scale_factor

    def perform_operation(self, images):
        """
        Scale the passed :attr:`images` by the factor specified during
        instantiation, returning the scaled image.

        :param images: The image to scale.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            w, h = image.size

            new_h = int(h * self.scale_factor)
            new_w = int(w * self.scale_factor)

            return image.resize((new_w, new_h), resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Distort(Operation):
    """
    This class performs randomised, elastic distortions on images.
    """
    def __init__(self, probability, grid_width, grid_height, magnitude):
        """
        As well as the probability, the granularity of the distortions
        produced by this class can be controlled using the width and
        height of the overlaying distortion grid. The larger the height
        and width of the grid, the smaller the distortions. This means
        that larger grid sizes can result in finer, less severe distortions.
        As well as this, the magnitude of the distortions vectors can
        also be adjusted.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param grid_width: The width of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param grid_height: The height of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param magnitude: Controls the degree to which each distortion is
         applied to the overlaying distortion grid.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        """
        Operation.__init__(self, probability)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)
        # TODO: Implement non-random magnitude.
        self.randomise_magnitude = True

    def perform_operation(self, images):
        """
        Distorts the passed image(s) according to the parameters supplied during
        instantiation, returning the newly distorted image.

        :param images: The image(s) to be distorted.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        w, h = images[0].size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        # For loop that generates polygons could be rewritten, but maybe harder to read?
        # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

        # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for a, b, c, d in polygon_indices:
            dx = random.randint(-self.magnitude, self.magnitude)
            dy = random.randint(-self.magnitude, self.magnitude)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                            x2, y2,
                            x3 + dx, y3 + dy,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                            x2 + dx, y2 + dy,
                            x3, y3,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                            x2, y2,
                            x3, y3,
                            x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                            x2, y2,
                            x3, y3,
                            x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        def do(image):

            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class GaussianDistortion(Operation):
    """
    This class performs randomised, elastic gaussian distortions on images.
    """
    def __init__(self, probability, grid_width, grid_height, magnitude, corner, method, mex, mey, sdx, sdy):
        """
        As well as the probability, the granularity of the distortions
        produced by this class can be controlled using the width and
        height of the overlaying distortion grid. The larger the height
        and width of the grid, the smaller the distortions. This means
        that larger grid sizes can result in finer, less severe distortions.
        As well as this, the magnitude of the distortions vectors can
        also be adjusted.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param grid_width: The width of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param grid_height: The height of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param magnitude: Controls the degree to which each distortion is
         applied to the overlaying distortion grid.
        :param corner: which corner of picture to distort.
         Possible values: "bell"(circular surface applied), "ul"(upper left),
         "ur"(upper right), "dl"(down left), "dr"(down right).
        :param method: possible values: "in"(apply max magnitude to the chosen
         corner), "out"(inverse of method in).
        :param mex: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param mey: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdx: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdy: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        :type corner: String
        :type method: String
        :type mex: Float
        :type mey: Float
        :type sdx: Float
        :type sdy: Float

        For values :attr:`mex`, :attr:`mey`, :attr:`sdx`, and :attr:`sdy` the
        surface is based on the normal distribution:

        .. math::

         e^{- \Big( \\frac{(x-\\text{mex})^2}{\\text{sdx}} + \\frac{(y-\\text{mey})^2}{\\text{sdy}} \Big) }
        """
        Operation.__init__(self, probability)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)
        self.randomise_magnitude = True
        self.corner = corner
        self.method = method
        self.mex = mex
        self.mey = mey
        self.sdx = sdx
        self.sdy = sdy

    def perform_operation(self, images):
        """
        Distorts the passed image(s) according to the parameters supplied
        during instantiation, returning the newly distorted image.

        :param images: The image(s) to be distorted.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        w, h = images[0].size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        def sigmoidf(x, y, sdx=0.05, sdy=0.05, mex=0.5, mey=0.5, const=1):
            sigmoid = lambda x1, y1:  (const * (math.exp(-(((x1-mex)**2)/sdx + ((y1-mey)**2)/sdy) )) + max(0,-const) - max(0, const))
            xl = np.linspace(0,1)
            yl = np.linspace(0, 1)
            X, Y = np.meshgrid(xl, yl)

            Z = np.vectorize(sigmoid)(X, Y)
            mino = np.amin(Z)
            maxo = np.amax(Z)
            res = sigmoid(x, y)
            res = max(((((res - mino) * (1 - 0)) / (maxo - mino)) + 0), 0.01)*self.magnitude
            return res

        def corner(x, y, corner="ul", method="out", sdx=0.05, sdy=0.05, mex=0.5, mey=0.5):
            ll = {'dr': (0, 0.5, 0, 0.5), 'dl': (0.5, 1, 0, 0.5), 'ur': (0, 0.5, 0.5, 1), 'ul': (0.5, 1, 0.5, 1), 'bell': (0, 1, 0, 1)}
            new_c = ll[corner]
            new_x = (((x - 0) * (new_c[1] - new_c[0])) / (1 - 0)) + new_c[0]
            new_y = (((y - 0) * (new_c[3] - new_c[2])) / (1 - 0)) + new_c[2]
            if method == "in":
                const = 1
            else:
                if method == "out":
                    const =- 1
                else:
                    const = 1
            res = sigmoidf(x=new_x, y=new_y,sdx=sdx, sdy=sdy, mex=mex, mey=mey, const=const)

            return res

        def do(image):

            for a, b, c, d in polygon_indices:
                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]

                sigmax = corner(x=x3/w, y=y3/h, corner=self.corner, method=self.method, sdx=self.sdx, sdy=self.sdy, mex=self.mex, mey=self.mey)
                dx = np.random.normal(0, sigmax, 1)[0]
                dy = np.random.normal(0, sigmax, 1)[0]
                polygons[a] = [x1, y1,
                               x2, y2,
                               x3 + dx, y3 + dy,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
                polygons[b] = [x1, y1,
                               x2 + dx, y2 + dy,
                               x3, y3,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
                polygons[c] = [x1, y1,
                               x2, y2,
                               x3, y3,
                               x4 + dx, y4 + dy]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
                polygons[d] = [x1 + dx, y1 + dy,
                               x2, y2,
                               x3, y3,
                               x4, y4]

            generated_mesh = []
            for i in range(len(dimensions)):
                generated_mesh.append([dimensions[i], polygons[i]])

            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Zoom(Operation):
    """
    This class is used to enlarge images (to zoom) but to return a cropped
    region of the zoomed image of the same size as the original image.
    """
    def __init__(self, probability, min_factor, max_factor):
        """
        The amount of zoom applied is randomised, from between
        :attr:`min_factor` and :attr:`max_factor`. Set these both to the same
        value to always zoom by a constant factor.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_factor: The minimum amount of zoom to apply. Set both the
         :attr:`min_factor` and :attr:`min_factor` to the same values to zoom
         by a constant factor.
        :param max_factor: The maximum amount of zoom to apply. Set both the
         :attr:`min_factor` and :attr:`min_factor` to the same values to zoom
         by a constant factor.
        :type probability: Float
        :type min_factor: Float
        :type max_factor: Float
        """
        Operation.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def perform_operation(self, images):
        """
        Zooms/scales the passed image(s) and returns the new image.

        :param images: The image(s) to be zoomed.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        factor = round(random.uniform(self.min_factor, self.max_factor), 2)

        def do(image):
            w, h = image.size

            image_zoomed = image.resize((int(round(image.size[0] * factor)),
                                         int(round(image.size[1] * factor))),
                                         resample=Image.BICUBIC)
            w_zoomed, h_zoomed = image_zoomed.size

            return image_zoomed.crop((floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) - (float(h) / 2)),
                                      floor((float(w_zoomed) / 2) + (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) + (float(h) / 2))))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class ZoomRandom(Operation):
    """
    This class is used to zoom into random areas of the image.
    """

    def __init__(self, probability, percentage_area, randomise):
        """
        Zooms into a random area of the image, rather than the centre of
        the image, as is done by :class:`Zoom`. The zoom factor is fixed
        unless :attr:`randomise` is set to ``True``.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param percentage_area: A value between 0.1 and 1 that represents the
         area that will be cropped, with 1 meaning the entire area of the
         image will be cropped and 0.1 mean 10% of the area of the image
         will be cropped, before zooming.
        :param randomise: If ``True``, uses the :attr:`percentage_area` as an
         upper bound, and randomises the zoom level from between 0.1 and
         :attr:`percentage_area`.
        """
        Operation.__init__(self, probability)
        self.percentage_area = percentage_area
        self.randomise = randomise

    def perform_operation(self, images):
        """
        Randomly zoom into the passed :attr:`images` by first cropping the image
        based on the :attr:`percentage_area` argument, and then resizing the
        image to match the size of the input area.

        Effectively, you are zooming in on random areas of the image.

        :param images: The image to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            if self.randomise:
                r_percentage_area = round(random.uniform(0.1, self.percentage_area), 2)
            else:
                r_percentage_area = self.percentage_area

            w, h = images[0].size
            w_new = int(floor(w * r_percentage_area))
            h_new = int(floor(h * r_percentage_area))

            random_left_shift = random.randint(0, (w - w_new))  # Note: randint() is from uniform distribution.
            random_down_shift = random.randint(0, (h - h_new))

            image = image.crop((random_left_shift, random_down_shift, w_new + random_left_shift, h_new + random_down_shift))

            return image.resize((w, h), resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class HSVShifting(Operation):
    """
    CURRENTLY NOT IMPLEMENTED.
    """
    def __init__(self, probability, hue_shift, saturation_scale, saturation_shift, value_scale, value_shift):
        Operation.__init__(self, probability)
        self.hue_shift = hue_shift
        self.saturation_scale = saturation_scale
        self.saturation_shift = saturation_shift
        self.value_scale = value_scale
        self.value_shift = value_shift

    def perform_operation(self, images):

        def do(image):
            hsv = np.array(image.convert("HSV"), 'float64')
            hsv /= 255.

            hsv[..., 0] += np.random.uniform(-self.hue_shift, self.hue_shift)
            hsv[..., 1] *= np.random.uniform(1 / (1 + self.saturation_scale), 1 + self.saturation_scale)
            hsv[..., 1] += np.random.uniform(-self.saturation_shift, self.saturation_shift)
            hsv[..., 2] *= np.random.uniform(1 / (1 + self.value_scale), 1 + self.value_scale)
            hsv[..., 2] += np.random.uniform(-self.value_shift, self.value_shift)

            hsv.clip(0, 1, hsv)
            hsv = np.uint8(np.round(hsv * 255.))

            return Image.fromarray(hsv, "HSV").convert("RGB")

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class RandomErasing(Operation):
    """
    Class that performs Random Erasing, an augmentation technique described
    in `https://arxiv.org/abs/1708.04896 <https://arxiv.org/abs/1708.04896>`_
    by Zhong et al. To quote the authors, random erasing:

    "*... randomly selects a rectangle region in an image, and erases its
    pixels with random values.*"

    Exactly this is provided by this class.

    Random Erasing can make a trained neural network more robust to occlusion.
    """
    def __init__(self, probability, rectangle_area):
        """
        The size of the random rectangle is controlled using the
        :attr:`rectangle_area` parameter. This area is random in its
        width and height.

        :param probability: The probability that the operation will be
         performed.
        :param rectangle_area: The percentage are of the image to occlude.
        """
        Operation.__init__(self, probability)
        self.rectangle_area = rectangle_area

    def perform_operation(self, images):
        """
        Adds a random noise rectangle to a random area of the passed image,
        returning the original image with this rectangle superimposed.

        :param images: The image(s) to add a random noise rectangle to.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):

            w, h = image.size

            w_occlusion_max = int(w * self.rectangle_area)
            h_occlusion_max = int(h * self.rectangle_area)

            w_occlusion_min = int(w * 0.1)
            h_occlusion_min = int(h * 0.1)

            w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
            h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

            if len(image.getbands()) == 1:
                rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
            else:
                rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, len(image.getbands())) * 255))

            random_position_x = random.randint(0, w - w_occlusion)
            random_position_y = random.randint(0, h - h_occlusion)

            image.paste(rectangle, (random_position_x, random_position_y))

            return image

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Custom(Operation):
    """
    Class that allows for a custom operation to be performed using Augmentor's
    standard :class:`~Augmentor.Pipeline.Pipeline` object.
    """
    def __init__(self, probability, custom_function, **function_arguments):
        """
        Creates a custom operation that can be added to a pipeline.

        To add a custom operation you can instantiate this class, passing
        a function pointer, :attr:`custom_function`, followed by an
        arbitrarily long list keyword arguments, :attr:`\*\*function_arguments`.

        .. seealso:: The :func:`~Augmentor.Pipeline.Pipeline.add_operation`
         function.

        :param probability: The probability that the operation will be
         performed.
        :param custom_function: The name of the function that performs your
         custom code. Must return an Image object and accept an Image object
         as its first parameter.
        :param function_arguments: The arguments for your custom operation's
         code.
        :type probability: Float
        :type custom_function: \*Function
        :type function_arguments: dict
        """
        Operation.__init__(self, probability)
        self.custom_function = custom_function
        self.function_arguments = function_arguments

    def __str__(self):
        return "Custom (" + self.custom_function.__name__ + ")"

    def perform_operation(self, images):
        """
        Perform the custom operation on the passed image(s), returning the
        transformed image(s).

        :param images: The image to perform the custom operation on.
        :return: The transformed image(s) (other functions in the pipeline
         will expect an image of type PIL.Image)
        """
        return self.function_name(images, **self.function_arguments)


class ZoomGroundTruth(Operation):
    """
    This class is used to enlarge images (to zoom) but to return a cropped
    region of the zoomed image of the same size as the original image.
    """
    def __init__(self, probability, min_factor, max_factor):
        """
        The amount of zoom applied is randomised, from between
        :attr:`min_factor` and :attr:`max_factor`. Set these both to the same
        value to always zoom by a constant factor.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_factor: The minimum amount of zoom to apply. Set both the
         :attr:`min_factor` and :attr:`min_factor` to the same values to zoom
         by a constant factor.
        :param max_factor: The maximum amount of zoom to apply. Set both the
         :attr:`min_factor` and :attr:`min_factor` to the same values to zoom
         by a constant factor.
        :type probability: Float
        :type min_factor: Float
        :type max_factor: Float
        """
        Operation.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def perform_operation(self, images):
        """
        Zooms/scales the passed images and returns the new images.

        :param images: An arbitrarily long list of image(s) to be zoomed.
        :type images: List containing PIL.Image object(s).
        :return: The zoomed in image(s) as a list of PIL.Image object(s).
        """
        factor = round(random.uniform(self.min_factor, self.max_factor), 2)

        def do(image):

            w, h = image.size

            # TODO: Join these two functions together so that we don't have this image_zoom variable lying around.
            image_zoomed = image.resize((int(round(image.size[0] * factor)), int(round(image.size[1] * factor))), resample=Image.BICUBIC)
            w_zoomed, h_zoomed = image_zoomed.size

            return image_zoomed.crop((floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) - (float(h) / 2)),
                                      floor((float(w_zoomed) / 2) + (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) + (float(h) / 2))))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Mixup(Operation):
    """
    Implements the *mixup* augmentation method, as described in:

    Zhang et al. (2018), *mixup*: Beyond Empirical Risk Minimization,
    arXiv:1710.09412

    See `http://arxiv.org/abs/1710.09412 <http://arxiv.org/abs/1710.09412>`_
    for details.

    Also see `https://github.com/facebookresearch/mixup-cifar10 <https://github.com/facebookresearch/mixup-cifar10>`_
    for code which was followed to create this functionality in Augmentor.

    The *mixup* augmentation technique trains a neural network on
    "*combinations of pairs of examples and their labels*" (Zhang et al., 2018).

    In summary, *mixup* constructs training samples as follows:

    .. math::
        \\tilde{x} = \\lambda x_i + (1-\\lambda)x_j
    .. math::
        \\tilde{y} = \\lambda y_i + (1-\\lambda)y_j

    where :math:`(x_i, y_i)` and :math:`(x_j, y_j)` are two samples from
    the training data, :math:`x_i, x_j` are raw input vectors,
    :math:`y_i, y_j` are one-hot label encodings (such as [0.0, 1.0]), and
    :math:`\\lambda \\in [0, 1]` where :math:`\\lambda` is sampled randomly
    from the Beta distribution, :math:`\\beta(\\alpha, \\alpha)`.

    The :math:`\\alpha` hyper-parameter controls the strength of the
    interpolation between image-label pairs, where
    :math:`\\alpha \in \\{0, \\infty \\}`

    According to the paper referenced above, values for :math:`\\alpha`
    between 0.1 and 0.4 led to best performance. Smaller values for
    :math:`\\alpha` result in less *mixup* effect whereas larger values would
    tend to result in overfitting.
    """
    def __init__(self, probability, alpha=0.4):
        """

        .. note:: Not yet enabled!
            This function is currently implemented but not **enabled**, as it
            requires each image's label in order to operate - something which
            Augmentor was not designed to handle.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param alpha: The alpha parameter controls the strength of the
         interpolation between image-label pairs. It's value can be any value
         greater than 0. A smaller value for :attr:`alpha` results in more
         values closer to 0 or 1, meaning the *mixup* is more often closer to
         either of the images in the pair. Its value is set to 0.4 by default.
        :type probability: Float
        :type alpha: Float
        """
        Operation.__init__(self, probability)
        self.alpha = alpha

    def perform_operation(self, images):
        """
        This function is currently implemented but not **enabled**, as it
        requires each image's label in order to operate - something which
        Augmentor was not designed to handle.

        This is therefore future work, and may only be possible when used in
        combination with generators.
        """

        if self.alpha > 0:
            lambda_value = np.random.beta(self.alpha, self.alpha)
        else:
            lambda_value = 1

        def do(image1, image2, y1, y2):

            image1 = np.asarray(image1)
            image1 = image1.astype('float32')

            image2 = np.asarray(image2)
            image2 = image2.astype('float32')

            mixup_x = lambda_value * image1 + (1 - lambda_value) * image2

            mixup_y = lambda_value * y1 + (1 - lambda_value) * y2

            return mixup_x, mixup_y

        augmented_images = []

        y1 = np.array([0.0, 1.0])
        y2 = np.array([1.0, 0.0])

        for image in images:
            augmented_images.append(do(image, image, y1, y2))

        return augmented_images
