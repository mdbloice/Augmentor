# Pipeline.py
# Author: Marcus D. Bloice <https://github.com/mdbloice>
# Licensed under the terms of the MIT Licence.
"""
The Pipeline module is the user facing API for the Augmentor package. It
contains the :class:`~Augmentor.Pipeline.Pipeline` class which is used to 
create pipeline objects, which can be used to build an augmentation pipeline
by adding operations to the pipeline object. 

For a good overview of how to use Augmentor, along with code samples and 
example images, can be seen in the :ref:`mainfeatures` section.  
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *

from .Operations import *
from .ImageUtilities import scan_directory, AugmentorImage

import os
import random
import uuid
import warnings

from skimage.io import imread

from tqdm import tqdm
from PIL import Image


class Pipeline(object):
    """
    The Pipeline class handles the creation of augmentation pipelines
    and the generation of augmented data by applying operations to
    this pipeline.
    """

    # Some class variables we use often
    _probability_error_text = "The probability argument must be between 0 and 1."
    _threshold_error_text = "The value of threshold must be between 0 and 255."
    _valid_formats = ["PNG", "BMP", "GIF", "JPEG"]
    _legal_filters = ["NEAREST", "BICUBIC", "ANTIALIAS", "BILINEAR"]

    def __init__(self, source_directory=None, output_directory="output", save_format="JPEG"):
        """
        Create a new Pipeline object pointing to a directory containing your
        original image dataset.

        Create a new Pipeline object, using the :attr:`source_directory`
        parameter as a source directory where your original images are
        stored. This folder will be scanned, and any valid file files
        will be collected and used as the original dataset that should
        be augmented. The scan will find any image files with the extensions
        JPEG/JPG, PNG, and GIF (case insensitive).

        :param source_directory: A directory on your filesystem where your
         original images are stored.
        :param output_directory: Specifies where augmented images should be
         saved to the disk. Default is the directory **source** relative to
         the path where the original image set was specified. If it does not
         exist it will be created.
        :param save_format: The file format to use when saving newly created,
         augmented images. Default is JPEG. Legal options are BMP, PNG, and
         GIF.
        :return: A :class:`Pipeline` object.
        """
        random.seed()

        # TODO: Allow a single image to be added when initialising.
        self.image_counter = 0
        self.augmentor_images = []
        self.distinct_dimensions = set()
        self.distinct_formats = set()
        # TODO: Refactor this out as we do not use it anymore
        self.ground_truth_image_list = {}
        self.output_dir = output_directory


        # Now we populate some fields, which we may need to do again later if another
        # directory is added, so we place it all in a function of its own.
        if source_directory is not None:
            self._populate(source_directory=source_directory,
                           output_directory=output_directory,
                           ground_truth_directory=None,
                           ground_truth_output_directory=output_directory)

        self.save_format = save_format
        self.operations = []

        # print("Initialised with %s image(s) found in selected directory." % len(self.augmentor_images))
        # print("Output directory set to %s." % self.output_directory)

    def _populate(self, source_directory, output_directory, ground_truth_directory, ground_truth_output_directory):
        """
        Private method for populating member variables with AugmentorImage
        objects for each of the images found in the source directory 
        specified by the user. It also populates a number of fields such as
        the :attr:`output_directory` member variable, used later when saving
        images to disk.
        
        This method is used by :func:`__init__`. 
        
        :param source_directory: The directory to scan for images.
        :param output_directory: The directory to set for saving files.
         Defaults to a directory named output relative to 
         :attr:`source_directory`.
        :param ground_truth_directory: A directory containing ground truth 
         files for the associated images in the :attr:`source_directory` 
         directory.
        :param ground_truth_output_directory: A path to a directory to store
         the output of the operations on the ground truth data set.
        :type source_directory: String
        :type output_directory: String
        :type ground_truth_directory: String
        :type ground_truth_output_directory: String
        :return: None
        """

        ####
        # NOTE.
        # A lot of this functionality, such as checking for paths that exist
        # and that they are writable, etc., will eventually be moved to the
        # AugmentorImage class in the ImageUtilities module.
        ####

        # Check if the source directory for the original images to augment exists at all
        if not os.path.exists(source_directory):
            raise IOError("The source directory you specified does not exist.")

        # If a ground truth directory is being specified we will check here if the path exists at all.
        if ground_truth_directory:
            if not os.path.exists(ground_truth_directory):
                raise IOError("The ground truth source directory you specified does not exist.")

        # Get the absolute path of the output directory specified.
        abs_output_directory = os.path.join(source_directory, output_directory)

        # Check if the output directory exists, if it does not attempt to create it
        # and raise an exception if this fails.
        if not os.path.exists(abs_output_directory):
            try:
                os.makedirs(abs_output_directory)
            except IOError:
                print("Insufficient rights to read or write output directory (%s)" % abs_output_directory)

        # Use the ImageUtilities class's scan_directory function to find images in the
        # source_directory folder.
        self.image_list = scan_directory(source_directory)

        # Create AugmentorImage objects for each of the images in the source directory.
        for image_path in scan_directory(source_directory):
            single_augmentor_image = AugmentorImage(image_path=image_path,
                                                    output_directory=abs_output_directory)
            if ground_truth_directory:
                single_augmentor_image.ground_truth = os.path.join(os.path.abspath(ground_truth_directory),
                                                                   os.path.basename(image_path))
            self.augmentor_images.append(single_augmentor_image)

        # Check the images, read their dimensions, and remove them if they cannot be read
        # TODO: Do not throw an error here, just remove the image and continue.
        for augmentor_image in self.augmentor_images:
            try:
                with Image.open(augmentor_image.image_path) as opened_image:
                    self.distinct_dimensions.add(opened_image.size)
                    self.distinct_formats.add(opened_image.format)
            except IOError:
                print("There is a problem with image %s in your source directory. "
                      "It is unreadable and will not be included when augmenting."
                      % os.path.basename(augmentor_image.image_path))
                self.augmentor_images.remove(augmentor_image)

        # Finally, we will print some informational messages.
        print("Initialised with %s image(s) found in selected directory." % len(self.augmentor_images))
        print("Output directory set to %s." % abs_output_directory)

    def _execute(self, augmentor_image, save_to_disk=True):
        """
        Private method. Used to pass an image through the current pipeline,
        and return the augmented image.
        
        The returned image can then either be saved to disk or simply passed
        back to the user. Currently this is fixed to True, as Augmentor
        has only been implemented to save to disk at present.

        :param augmentor_image: The image to pass through the pipeline.
        :param save_to_disk: Whether to save the image to disk. Currently
         fixed to true.
        :type augmentor_image: :class:`ImageUtilities.AugmentorImage`
        :type save_to_disk: Boolean
        :return: The augmented image.
        """
        self.image_counter += 1  # TODO: See if I can remove this...

        if augmentor_image.image_path is not None:
            try:
                image = Image.open(augmentor_image.image_path)
            except OSError:
                image = imread(augmentor_image.image_path)
                if image.shape[-1] not in (1, 3):
                    raise ValueError('Image should contain 1 or 3 channels')
                image = Image.fromarray(image)

        else:
            image = augmentor_image.image_PIL

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)

        if save_to_disk:
            file_name = str(uuid.uuid4()) + "." + self.save_format
            try:
                # A strange error is forcing me to do this at the moment, but will fix later properly
                # TODO: Fix this!
                if image.mode != "RGB":
                    image = image.convert("RGB")
                # For testing this cab be commented out to create only one output image.
                # file_name = "test"
                image.save(os.path.join(augmentor_image.output_directory, file_name), self.save_format)
            except IOError:
                print("Error writing %s." % file_name)

        return image


    def sample(self, n):
        """
        Generate :attr:`n` number of samples from the current pipeline.

        This function samples from the pipeline, using the original images
        defined during instantiation. All images generated by the pipeline
        are by default stored in an ``output`` directory, relative to the
        path defined during the pipeline's instantiation.

        :param n: The number of new samples to produce.
        :type n: Integer
        :return: None
        """
        if len(self.augmentor_images) == 0:
            raise IndexError("There are no images in the pipeline. "
                             "Add a directory using add_directory(), "
                             "pointing it to a directory containing images.")

        if len(self.operations) == 0:
            raise IndexError("There are no operations associated with this pipeline.")

        sample_count = 1

        progress_bar = tqdm(total=n, desc="Executing Pipeline", unit=' Samples', leave=False)
        while sample_count <= n:
            for augmentor_image in self.augmentor_images:
                if sample_count <= n:
                    self._execute(augmentor_image)
                    file_name_to_print = os.path.basename(augmentor_image.image_path)
                    # This is just to shorten very long file names which obscure the progress bar.
                    if len(file_name_to_print) >= 30:
                        file_name_to_print = file_name_to_print[0:10] + "..." + \
                                             file_name_to_print[-10: len(file_name_to_print)]
                    progress_bar.set_description("Processing %s" % file_name_to_print)
                    progress_bar.update(1)
                sample_count += 1
        progress_bar.close()

    def apply_current_pipeline(self, image_path, save_to_disk=False):
        """
        .. warning::
         This function has been deprecated in favour of :func:`apply_from_path()`.

        Apply the current pipeline to a single image, returning the
        transformed image. By default, the transformed image is not saved 
        to disk, and is returned to the user.

        This method can be used to pass a single image through the
        pipeline, but will not save the transformed to disk by
        default. To save to disk, supply a :attr:`save_to_disk`
        argument set to True.

        :param image_path: The path to the image to pass through the current 
         pipeline.
        :param save_to_disk: Whether to save the image to disk. Defaults to
         False.
        :type image_path: String
        :type save_to_disk: Boolean
        :return: The transformed image.
        """

        warnings.warn("This function has been deprecated in favour of sample_with_path() and sample_with_array().",
                      DeprecationWarning)

        return self.apply_from_path(image_path=image_path, save_to_disk=save_to_disk)

    def apply_from_path(self, image_path, save_to_disk=False):
        # for some reason the method was mentioned, but not implemented yet. So I make one --@arsenyinfo
        return self._execute(AugmentorImage(os.path.abspath(image_path), output_directory=self.output_dir),
                             save_to_disk)

    def sample_with_path(self, image_path, save_to_disk=False):
        raise NotImplementedError("This method is currently not implemented.")

    def sample_with_array(self, image_array, save_to_disk=False):

        a = AugmentorImage(image_path=None, output_directory=None)
        a.image_PIL = Image.fromarray(image_array)

        return self._execute(a, save_to_disk)

    def sample_with_image(self, image, save_to_disk=False):
        raise NotImplementedError("This method is currently not implemented.")

    def image_generator(self, n):
        for i in range(n):
            yield self.sample(1)

    def add_operation(self, operation):
        """
        Add an operation directly to the pipeline. Can be used to add custom
        operations to a pipeline.

        To add custom operations to a pipeline, subclass from the
        Operation abstract base class, overload its methods, and insert the 
        new object into the pipeline using this method.

         .. seealso:: The :class:`.Operation` class.

        :param operation: An object of the operation you wish to add to the
         pipeline. Will accept custom operations written at run-time.
        :type operation: Operation
        :return: None
        """
        if isinstance(operation, Operation):
            self.operations.append(operation)
        else:
            raise TypeError("Must be of type Operation to be added to the pipeline.")

    def remove_operation(self, operation_index=-1):
        """
        Remove the operation specified by :attr:`operation_index`, if
        supplied, otherwise it will remove the latest operation added to the
        pipeline.

         .. seealso:: Use the :func:`status` function to find an operation's
          index.

        :param operation_index: The index of the operation to remove.
        :type operation_index: Integer
        :return: The removed operation. You can reinsert this at end of the
         pipeline using :func:`add_operation` if required.
        """

        # Python's own List exceptions can handle erroneous user input.
        self.operations.pop(operation_index)

    def add_further_directory(self, new_source_directory, new_output_directory="output"):
        """
        Add a further directory containing images you wish to scan for augmentation.
        
        :param new_source_directory: The directory to scan for images.
        :param new_output_directory: The directory to use for outputted, 
         augmented images.
        :type new_source_directory: String
        :type new_output_directory: String
        :return: None
        """
        if not os.path.exists(new_source_directory):
            raise IOError("The path does not appear to exist.")

        self._populate(source_directory=new_source_directory,
                       output_directory=new_output_directory,
                       ground_truth_directory=None,
                       ground_truth_output_directory=new_output_directory)

    def status(self):
        """
        Prints the status of the pipeline to the console. If you want to
        remove an operation, use the index shown and the 
        :func:`remove_operation` method.

         .. seealso:: The :func:`remove_operation` function.

         .. seealso:: The :func:`add_operation` function.

        The status includes the number of operations currently attached to
        the pipeline, each operation's parameters, the number of images in the
        pipeline, and a summary of the images' properties, such as their
        dimensions and formats.

        :return: None
        """
        # TODO: Return this as a dictionary of some kind and print from the dict if in console
        print("There are %s operation(s) in the current pipeline." % len(self.operations))

        if len(self.operations) != 0:
            operation_index = 0
            for operation in self.operations:
                print("Index %s:\n\tOperation %s (probability: %s):" % (operation_index, operation, operation.probability))
                for operation_attribute, operation_value in operation.__dict__.items():
                    print("\t\tAttribute: %s (%s)" % (operation_attribute, operation_value))
                operation_index += 1
            print()

        print("There are %s image(s) in the source directory." % len(self.augmentor_images))

        if len(self.augmentor_images) != 0:
            print("Dimensions:")
            for distinct_dimension in self.distinct_dimensions:
                print("\tWidth: %s Height: %s" % (distinct_dimension[0], distinct_dimension[1]))
            print("Formats:")
            for distinct_format in self.distinct_formats:
                print("\t %s" % distinct_format)

        print("\nYou can remove operations using the appropriate index and the remove_operation(index) function.")

    @staticmethod
    def set_seed(seed):
        """
        Set the seed of Python's internal random number generator.

        :param seed: The seed to use. Strings or other objects will be hashed.
        :type seed: Integer
        :return: None
        """
        random.seed(seed)

    # TODO: Add this feature ASAP
    def subtract_mean(self, probability=1):
        # For implementation example, see bottom of:
        # https://patrykchrabaszcz.github.io/Imagenet32/
        self.add_operation(Mean(probability=probability))

    def rotate90(self, probability):
        """
        Rotate an image by 90 degrees.

        The operation will rotate an image by 90 degrees, and will be
        performed with a probability of that specified by the
        :attr:`probability` parameter.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :type probability: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Rotate(probability=probability, rotation=90))

    def rotate180(self, probability):
        """
        Rotate an image by 180 degrees.

        The operation will rotate an image by 180 degrees, and will be
        performed with a probability of that specified by the
        :attr:`probability` parameter.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :type probability: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Rotate(probability=probability, rotation=180))

    def rotate270(self, probability):
        """
        Rotate an image by 270 degrees.

        The operation will rotate an image by 270 degrees, and will be
        performed with a probability of that specified by the
        :attr:`probability` parameter.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :type probability: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Rotate(probability=probability, rotation=270))

    def rotate_random_90(self, probability):
        """
        Rotate an image by either 90, 180, or 270 degrees, selected randomly.

        This function will rotate by either 90, 180, or 270 degrees. This is
        useful to avoid scenarios where images may be rotated back to their
        original positions (such as a :func:`rotate90` and a :func:`rotate270` 
        being performed directly afterwards. The random rotation is chosen 
        uniformly from 90, 180, or 270 degrees. The probability controls the 
        chance of the operation being performed at all, and does not affect 
        the rotation degree.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :type probability: Float
        :return:
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Rotate(probability=probability, rotation=-1))

    def rotate_alt(self, probability, rotation):
        # Placeholder - to be implemented.         
        pass

    def rotate(self, probability, max_left_rotation, max_right_rotation):
        """
        Rotate an image by an arbitrary amount.

        The operation will rotate an image by an random amount, within a range
        specified. The parameters :attr:`max_left_rotation` and
        :attr:`max_right_rotation` allow you to control this range. If you
        wish to rotate the images by an exact number of degrees, set both
        :attr:`max_left_rotation` and :attr:`max_right_rotation` to the same
        value.
        
        .. note:: This function will rotate **in place**, and crop the largest 
         possible rectangle from the rotated image.
         
        In practice, angles larger than 25 degrees result in images that 
        do not render correctly, therefore there is a limit of 25 degrees
        for this function.
         
        If this function returns images that are not rendered correctly, then
        you must reduce the :attr:`max_left_rotation` and
        :attr:`max_right_rotation` arguments!

        :param max_left_rotation: The maximum number of degrees the image can
         be rotated to the left.
        :param max_right_rotation: The maximum number of degrees the image can
         be rotated to the right.
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :type max_left_rotation: Integer
        :type max_right_rotation: Integer
        :type probability: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        if not 0 <= max_left_rotation <= 25:
            raise ValueError("The max_left_rotation argument must be between 0 and 25.")
        if not 0 <= max_right_rotation <= 25:
            raise ValueError("The max_right_rotation argument must be between 0 and 25.")
        else:
            self.add_operation(RotateRange(probability=probability, max_left_rotation=ceil(max_left_rotation),
                                           max_right_rotation=ceil(max_right_rotation)))

    def flip_top_bottom(self, probability):
        """
        Flip (mirror) the image along its vertical axis, i.e. from top to
        bottom.

        .. seealso:: The :func:`flip_left_right` function.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :type probability: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Flip(probability=probability, top_bottom_left_right="TOP_BOTTOM"))

    def flip_left_right(self, probability):
        """
        Flip (mirror) the image along its horizontal axis, i.e. from left to
        right.

        .. seealso:: The :func:`flip_top_bottom` function.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :type probability: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Flip(probability=probability, top_bottom_left_right="LEFT_RIGHT"))

    def flip_random(self, probability):
        """
        Flip (mirror) the image along **either** its horizontal or vertical
        axis.

        This function mirrors the image along either the horizontal axis or
        the vertical access. The axis is selected randomly.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :type probability: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Flip(probability=probability, top_bottom_left_right="RANDOM"))

    def random_distortion(self, probability, grid_width, grid_height, magnitude):
        """
        Performs a random, elastic distortion on an image.

        This function performs a randomised, elastic distortion controlled
        by the parameters specified. The grid width and height controls how
        fine the distortions are. Smaller sizes will result in larger, more
        pronounced, and less granular distortions. Larger numbers will result
        in finer, more granular distortions. The magnitude of the distortions
        can be controlled using magnitude. This can be random or fixed.
        
        *Good* values for parameters are between 2 and 10 for the grid
        width and height, with a magnitude of between 1 and 10. Using values
        outside of these approximate ranges may result in unpredictable 
        behaviour.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param grid_width: The number of rectangles in the grid's horizontal
         axis.
        :param grid_height: The number of rectangles in the grid's vertical
         axis.
        :param magnitude: The magnitude of the distortions.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Distort(probability=probability, grid_width=grid_width,
                                       grid_height=grid_height, magnitude=magnitude))

    def zoom(self, probability, min_factor, max_factor):
        """
        Zoom in to an image, while **maintaining its size**. The amount by
        which the image is zoomed is a randomly chosen value between 
        :attr:`min_factor` and :attr:`max_factor`.
        
        Typical values may be ``min_factor=1.1`` and ``max_factor=1.5``.
        
        To zoom by a constant amount, set :attr:`min_factor` and
        :attr:`max_factor` to the same value.
        
        .. seealso:: See :func:`zoom_random` for zooming into random areas 
         of the image.
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed. 
        :param min_factor: The minimum factor by which to zoom the image.
        :param max_factor: The maximum factor by which to zoom the image.
        :type probability: Float
        :type min_factor: Float
        :type max_factor: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif min_factor < 1:
            raise ValueError("The min_factor argument must be greater than 1.")
        else:
            self.add_operation(Zoom(probability=probability, min_factor=min_factor, max_factor=max_factor))

    def zoom_random(self, probability, percentage_area, randomise_percentage_area=False):
        """
        Zooms into an image at a random location within the image. 
        
        You can randomise the zoom level by setting the 
        :attr:`randomise_percentage_area` argument to true. 
        
        .. seealso:: See :func:`zoom` for zooming into the centre of images.

        :param probability: The probability that the function will execute
         when the image is passed through the pipeline.
        :param percentage_area: The area, as a percentage of the current
         image's area, to crop.
        :param randomise_percentage_area: If True, will use 
         :attr:`percentage_area` as an upper bound and randomise the crop from
         between 0 and :attr:`percentage_area`. 
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0.1 <= percentage_area < 1:
            raise ValueError("The percentage_area argument must be greater than 0.1 and less than 1.")
        elif not isinstance(randomise_percentage_area, bool):
            raise ValueError("The randomise_percentage_area argument must be True or False.")
        else:
            self.add_operation(ZoomRandom(probability=probability, percentage_area=percentage_area, randomise=randomise_percentage_area))

    def crop_by_size(self, probability, width, height, centre=True):
        """
        Crop an image according to a set of dimensions.
    
        Crop each image according to :attr:`width` and :attr:`height`, by
        default at the centre of each image, otherwise at a random location
        within the image.
    
        .. seealso:: See :func:`crop_random` to crop a random, non-centred
         area of the image.
    
        If the crop area exceeds the size of the image, this function will
        crop the entire area of the image.
    
        :param probability: The probability that the function will execute
         when the image is passed through the pipeline.
        :param width: The width of the desired crop.
        :param height: The height of the desired crop.
        :param centre: If **True**, crops from the centre of the image,
         otherwise crops at a random location within the image, maintaining
         the dimensions specified.
        :type probability: Float
        :type width: Integer
        :type height: Integer
        :type centre: Boolean
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif width <= 1:
            raise ValueError("The width argument must be greater than 1.")
        elif height <= 1:
            raise ValueError("The height argument must be greater than 1.")
        elif not isinstance(centre, bool):
            raise ValueError("The centre argument must be True or False.")
        else:
            self.add_operation(Crop(probability=probability, width=width, height=height, centre=centre))

    def crop_centre(self, probability, percentage_area, randomise_percentage_area=False):
        """
        Crops the centre of an image as a percentage of the image's area.
        
        :param probability: The probability that the function will execute
         when the image is passed through the pipeline.
        :param percentage_area: The area, as a percentage of the current
         image's area, to crop.
        :param randomise_percentage_area: If True, will use 
         :attr:`percentage_area` as an upper bound and randomise the crop from
         between 0 and :attr:`percentage_area`. 
        :type probability: Float
        :type percentage_area: Float
        :type randomise_percentage_area: Boolean
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0.1 <= percentage_area < 1:
            raise ValueError("The percentage_area argument must be greater than 0.1 and less than 1.")
        elif not isinstance(randomise_percentage_area, bool):
            raise ValueError("The randomise_percentage_area argument must be True or False.")
        else:
            self.add_operation(CropPercentage(probability=probability, percentage_area=percentage_area, centre=True,
                                              randomise_percentage_area=randomise_percentage_area))

    def crop_random(self, probability, percentage_area, randomise_percentage_area=False):
        """
        Crop a random area of an image, based on the percentage area to be
        returned.

        This function crops a random area from an image, based on the area you
        specify using :attr:`percentage_area`.

        :param probability: The probability that the function will execute
         when the image is passed through the pipeline.
        :param percentage_area: The area, as a percentage of the current
         image's area, to crop.
        :param randomise_percentage_area: If True, will use 
         :attr:`percentage_area` as an upper bound and randomise the crop from
         between 0 and :attr:`percentage_area`. 
        :type probability: Float
        :type percentage_area: Float
        :type randomise_percentage_area: Boolean
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0.1 <= percentage_area < 1:
            raise ValueError("The percentage_area argument must be greater than 0.1 and less than 1.")
        elif not isinstance(randomise_percentage_area, bool):
            raise ValueError("The randomise_percentage_area argument must be True or False.")
        else:
            self.add_operation(CropPercentage(probability=probability, percentage_area=percentage_area, centre=False,
                                              randomise_percentage_area=randomise_percentage_area))

    def histogram_equalisation(self, probability=1.0):
        """
        Apply histogram equalisation to the image.
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed. For histogram,
         equalisation it is recommended that the probability be set to 1.
        :type probability: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(HistogramEqualisation(probability=probability))

    def scale(self, probability, scale_factor):
        """
        Scale (enlarge) an image, while maintaining its aspect ratio. This
        returns an image with larger dimensions than the original image.
        
        Use :func:`resize` to resize an image to absolute pixel values. 
    
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param scale_factor: The factor to scale by, which must be greater
         than 1.0.
        :type probability: Float
        :type scale_factor: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif scale_factor <= 1.0:
            raise ValueError("The scale_factor argument must be greater than 1.")
        else:
            self.add_operation(Scale(probability=probability, scale_factor=scale_factor))

    def resize(self, probability, width, height, resample_filter="BICUBIC"):
        """
        Resize an image according to a set of dimensions specified by the
        user in pixels.
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed. For resizing,
         it is recommended that the probability be set to 1.
        :param width: The new width that the image should be resized to.
        :param height: The new height that the image should be resized to.
        :param resample_filter: The resampling filter to use. Must be one of
         BICUBIC, BILINEAR, ANTIALIAS, or NEAREST.
        :type probability: Float
        :type width: Integer
        :type height: Integer
        :type resample_filter: String
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not width > 1:
            raise ValueError("Width must be greater than 1.")
        elif not height > 1:
            raise ValueError("Height must be greater than 1.")
        elif resample_filter not in Pipeline._legal_filters:
            raise ValueError("The save_filter argument must be one of %s." % Pipeline._legal_filters)
        else:
            self.add_operation(Resize(probability=probability, width=width, height=height, resample_filter=resample_filter))

    def skew_left_right(self, probability, magnitude=1):
        """
        Skew an image by tilting it left or right by a random amount. The 
        magnitude of this skew can be set to a maximum using the 
        magnitude parameter. This can be either a scalar representing the
        maximum tilt, or vector representing a range.
        
        To see examples of the various skews, see :ref:`perspectiveskewing`.
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param magnitude: The maximum tilt, which must be value between 0.1  
         and 1.0, where 1 represents a tilt of 45 degrees.
        :type probability: Float
        :type magnitude: Float
        :return: None 
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability, skew_type="TILT_LEFT_RIGHT", magnitude=magnitude))

    def skew_top_bottom(self, probability, magnitude=1):
        """
        Skew an image by tilting it forwards or backwards by a random amount. 
        The magnitude of this skew can be set to a maximum using the 
        magnitude parameter. This can be either a scalar representing the
        maximum tilt, or vector representing a range.
        
        To see examples of the various skews, see :ref:`perspectiveskewing`.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param magnitude: The maximum tilt, which must be value between 0.1 
         and 1.0, where 1 represents a tilt of 45 degrees.
        :type probability: Float
        :type magnitude: Float
        :return: None 
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability,
                                    skew_type="TILT_TOP_BOTTOM",
                                    magnitude=magnitude))

    def skew_tilt(self, probability, magnitude=1):
        """
        Skew an image by tilting in a random direction, either forwards,
        backwards, left, or right, by a random amount. The magnitude of 
        this skew can be set to a maximum using the magnitude parameter.
        This can be either a scalar representing the maximum tilt, or 
        vector representing a range.
        
        To see examples of the various skews, see :ref:`perspectiveskewing`.
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed. 
        :param magnitude: The maximum tilt, which must be value between 0.1 
         and 1.0, where 1 represents a tilt of 45 degrees.
        :type probability: Float
        :type magnitude: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability,
                                    skew_type="TILT",
                                    magnitude=magnitude))

    def skew_corner(self, probability, magnitude=1):
        """
        Skew an image towards one corner, randomly by a random magnitude.
        
        To see examples of the various skews, see :ref:`perspectiveskewing`.
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed. 
        :param magnitude: The maximum skew, which must be value between 0.1 
         and 1.0.
        :return: 
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability,
                                    skew_type="CORNER",
                                    magnitude=magnitude))

    def skew(self, probability, magnitude=1):
        """
        Skew an image in a random direction, either left to right,
        top to bottom, or one of 8 corner directions. 
        
        To see examples of all the skew types, see :ref:`perspectiveskewing`.
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed. 
        :param magnitude: The maximum skew, which must be value between 0.1 
         and 1.0.
        :type probability: Float
        :type magnitude: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability,
                                    skew_type="RANDOM",
                                    magnitude=magnitude))

    def shear(self, probability, max_shear_left, max_shear_right):
        """
        Shear the image by a specified number of degrees.

        In practice, shear angles of more than 25 degrees can cause
        unpredictable behaviour. If you are observing images that are
        incorrectly rendered (e.g. they do not contain any information) 
        then reduce the shear angles.

        :param probability: The probability that the operation is performed. 
        :param max_shear_left: The max number of degrees to shear to the left.
         Cannot be larger than 25 degrees.
        :param max_shear_right: The max number of degrees to shear to the 
         right. Cannot be larger than 25 degrees.
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < max_shear_left <= 25:
            raise ValueError("The max_shear_left argument must be between 0 and 25.")
        elif not 0 < max_shear_right <= 25:
            raise ValueError("The max_shear_right argument must be between 0 and 25.")
        else:
            self.add_operation(Shear(probability=probability,
                                     max_shear_left=max_shear_left,
                                     max_shear_right=max_shear_right))

    def greyscale(self, probability):
        """
        Convert images to greyscale. For this operation, setting the 
        :attr:`probability` to 1.0 is recommended.
        
        .. seealso:: The :func:`black_and_white` function.
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed. For resizing,
         it is recommended that the probability be set to 1.
        :type probability: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Greyscale(probability=probability))

    def black_and_white(self, probability, threshold=128):
        """
        Convert images to black and white. In other words convert the image
        to use a 1-bit, binary palette. The threshold defaults to 128, 
        but can be controlled using the :attr:`threshold` parameter.
        
        .. seealso:: The :func:`greyscale` function.
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed. For resizing,
         it is recommended that the probability be set to 1.
        :param threshold: A value between 0 and 255 which controls the
         threshold point at which each pixel is converted to either black
         or white. Any values above this threshold are converted to white, and
         any values below this threshold are converted to black.
        :type probability: Float
        :type threshold: Integer
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 <= threshold <= 255:
            raise ValueError("The threshold must be between 0 and 255.")
        else:
            self.add_operation(BlackAndWhite(probability=probability, threshold=threshold))

    def invert(self, probability):
        """
        Invert an image. For this operation, setting the 
        :attr:`probability` to 1.0 is recommended.
        
        .. warning:: This function will cause errors if used on binary, 1-bit
         palette images (e.g. black and white).
        
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed. For resizing,
         it is recommended that the probability be set to 1.
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Invert(probability=probability))
