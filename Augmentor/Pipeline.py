
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *

from .Operations import *
from .ImageUtilities import scan_directory, scan, scan_dataframe, AugmentorImage

import os
import sys
import random
import uuid
import warnings
import numbers
import numpy as np

from tqdm import tqdm
from PIL import Image
from abc import abstractmethod

class Sequence(object):
    """Base object for fitting to a sequence of data, such as a dataset.

    Every `Sequence` must implements the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    # Notes

    `Sequence` are a safer way to do multiprocessing. This structure guarantees that the network will only train once
     on each sample per epoch which is not the case with generators.

    # Examples

    ```python
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np

        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.

        class CIFAR10Sequence(Sequence):

            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size

            def __len__(self):
                return np.ceil(len(self.x) / float(self.batch_size))

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

                return np.array([
                    resize(imread(file_name), (200, 200))
                       for file_name in batch_x]), np.array(batch_y)
    ```
    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass

    def __iter__(self):
        """Create an infinite generator that iterate over the Sequence."""
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item


class Iterator(Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


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

    def __init__(self, source_directory=None, output_directory="output", save_format=None):
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
         saved to the disk. Default is the directory **output** relative to
         the path where the original image set was specified. If it does not
         exist it will be created.
        :param save_format: The file format to use when saving newly created,
         augmented images. Default is JPEG. Legal options are BMP, PNG, and
         GIF.
        :return: A :class:`Pipeline` object.
        """
        random.seed()

        # TODO: Allow a single image to be added when initialising.
        # Initialise some variables for the Pipeline object.
        self.image_counter = 0
        self.augmentor_images = []
        self.distinct_dimensions = set()
        self.distinct_formats = set()
        self.save_format = save_format
        self.operations = []
        self.class_labels = []
        self.process_ground_truth_images = False

        # Now we populate some fields, which we may need to do again later if another
        # directory is added, so we place it all in a function of its own.
        if source_directory is not None:
            self._populate(source_directory=source_directory,
                           output_directory=output_directory,
                           ground_truth_directory=None,
                           ground_truth_output_directory=output_directory)

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

        # Check if the source directory for the original images to augment exists at all
        if not os.path.exists(source_directory):
            raise IOError("The source directory you specified does not exist.")

        # If a ground truth directory is being specified we will check here if the path exists at all.
        if ground_truth_directory:
            if not os.path.exists(ground_truth_directory):
                raise IOError("The ground truth source directory you specified does not exist.")

        # Get absolute path for output
        abs_output_directory = os.path.join(source_directory, output_directory)

        # Scan the directory that user supplied.
        self.augmentor_images, self.class_labels = scan(source_directory, abs_output_directory)

        self._check_images(abs_output_directory)

    def _check_images(self, abs_output_directory):
        """
        Private method. Used to check and get the dimensions of all of the images
        :param abs_output_directory: the absolute path of the output directory
        :return:
        """
        # Make output directory/directories
        if len(set(self.class_labels)) <= 1:  # Fixed bad bug by adding set() function here.
            if not os.path.exists(abs_output_directory):
                try:
                    os.makedirs(abs_output_directory)
                except IOError:
                    print("Insufficient rights to read or write output directory (%s)" % abs_output_directory)
        else:
            for class_label in self.class_labels:
                if not os.path.exists(os.path.join(abs_output_directory, str(class_label[0]))):
                    try:
                        os.makedirs(os.path.join(abs_output_directory, str(class_label[0])))
                    except IOError:
                        print("Insufficient rights to read or write output directory (%s)" % abs_output_directory)
        # Check the images, read their dimensions, and remove them if they cannot be read
        # TODO: Do not throw an error here, just remove the image and continue.
        for augmentor_image in self.augmentor_images:
            try:
                with Image.open(augmentor_image.image_path) as opened_image:
                    self.distinct_dimensions.add(opened_image.size)
                    self.distinct_formats.add(opened_image.format)
            except IOError as e:
                print("There is a problem with image %s in your source directory: %s" % (augmentor_image.image_path, e.message))
                self.augmentor_images.remove(augmentor_image)

        sys.stdout.write("Initialised with %s image(s) found.\n" % len(self.augmentor_images))
        sys.stdout.write("Output directory set to %s." % abs_output_directory)

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
        # self.image_counter += 1  # TODO: See if I can remove this...

        images = []

        if augmentor_image.image_path is not None:
            images.append(Image.open(augmentor_image.image_path))

        if augmentor_image.ground_truth is not None:
            if isinstance(augmentor_image.ground_truth, list):
                for image in augmentor_image.ground_truth:
                    images.append(Image.open(image))
            else:
                images.append(Image.open(augmentor_image.ground_truth))

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        if save_to_disk:
            file_name = str(uuid.uuid4())
            try:
                # TODO: Add a 'coerce' parameter to force conversion to RGB for PNGA->JPEG saving.
                # if image.mode != "RGB":
                #     image = image.convert("RGB")
                for i in range(len(images)):
                    if i == 0:
                        save_name = augmentor_image.class_label + "_original_" + os.path.basename(augmentor_image.image_path) + "_" + file_name \
                                    + "." + (self.save_format if self.save_format else augmentor_image.file_format)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
                    else:
                        save_name = "_groundtruth_(" + str(i) + ")_" + augmentor_image.class_label + "_" + os.path.basename(augmentor_image.image_path) + "_" + file_name \
                                    + "." + (self.save_format if self.save_format else augmentor_image.file_format)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
            except IOError as e:
                print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
                print("You can change the save format using the set_save_format(save_format) function.")
                print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")

        # Currently we return only the first image if it is a list
        # for the generator functions.  This will be fixed in a future
        # version.
        return images[0]

    def _execute_with_array(self, image):
        """
        Private method used to execute a pipeline on array or matrix data.
        :param image: The image to pass through the pipeline.
        :type image: Array like object.
        :return: The augmented image.
        """

        pil_image = [Image.fromarray(image)]

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                pil_image = operation.perform_operation(pil_image)

        numpy_array = np.asarray(pil_image[0])

        return numpy_array

    def set_save_format(self, save_format):
        """
        Set the save format for the pipeline. Pass the value
        :attr:`save_format="auto"` to allow Augmentor to choose
        the correct save format based on each individual image's
        file extension.
        If :attr:`save_format` is set to, for example,
        :attr:`save_format="JPEG"` or :attr:`save_format="JPG"`,
        Augmentor will attempt to save the files using the
        JPEG format, which may result in errors if the file cannot
        be saved in this format, such as trying to save PNG images
        with an alpha channel as JPEG.
        :param save_format: The save format to save the images
         when writing to disk.
        :return: None
        """

        if save_format == "auto":
            self.save_format = None
        else:
            self.save_format = save_format

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

    def sample_with_array(self, image_array, save_to_disk=False):
        """
        Generate images using a single image in array-like format.
        .. seealso::
         See :func:`keras_image_generator_without_replacement()` for
        :param image_array: The image to pass through the pipeline.
        :param save_to_disk: Whether to save to disk or not (default).
        :return:
        """
        a = AugmentorImage(image_path=None, output_directory=None)
        a.image_PIL = Image.fromarray(image_array)

        return self._execute(a, save_to_disk)

    @staticmethod
    def categorical_labels(numerical_labels):
        """
        Return categorical labels for an array of 0-based numerical labels.
        :param numerical_labels: The numerical labels.
        :type numerical_labels: Array-like list.
        :return: The categorical labels.
        """
        # class_labels_np = np.array([x.class_label_int for x in numerical_labels])
        class_labels_np = np.array(numerical_labels)
        one_hot_encoding = np.zeros((class_labels_np.size, class_labels_np.max() + 1))
        one_hot_encoding[np.arange(class_labels_np.size), class_labels_np] = 1
        one_hot_encoding = one_hot_encoding.astype(np.uint)

        return one_hot_encoding

    def image_generator(self):
        while True:
            im_index = random.randint(0, len(self.augmentor_images)-1)  # Fix for issue 52.
            yield self._execute(self.augmentor_images[im_index], save_to_disk=False), \
                self.augmentor_images[im_index].class_label_int

    def keras_generator(self, batch_size, scaled=True, image_data_format="channels_last", with_indexes=False):
        """
        Returns an image generator that will sample from the current pipeline
        indefinitely, as long as it is called.
        .. warning::
         This function returns images from the current pipeline
         **with replacement**.
        You must configure the generator to provide data in the same
        format that Keras is configured for. You can use the functions
        :func:`keras.backend.image_data_format()` and
        :func:`keras.backend.set_image_data_format()` to get and set
        Keras' image format at runtime.
        .. code-block:: python
            >>> from keras import backend as K
            >>> K.image_data_format()
            'channels_first'
            >>> K.set_image_data_format('channels_last')
            >>> K.image_data_format()
            'channels_last'
        By default, Augmentor uses ``'channels_last'``.
        :param batch_size: The number of images to return per batch.
        :type batch_size: Integer
        :param scaled: True (default) if pixels are to be converted
         to float32 values between 0 and 1, or False if pixels should be
         integer values between 0-255.
        :type scaled: Boolean
        :param image_data_format: Either ``'channels_last'`` (default) or
         ``'channels_first'``.
        :type image_data_format: String
        :return: An image generator.
        """

        if image_data_format not in ["channels_first", "channels_last"]:
            warnings.warn("To work with Keras, must be one of channels_first or channels_last.")

        iter = Iterator (len (self.augmentor_images), batch_size, True, 42)
        self.indexes_gen = self.iter._flow_index ()

        while True:

            # Randomly select 25 images for augmentation and yield the
            # augmented images.
            # X = np.array([])
            # y = np.array([])
            # The correct thing to do here is to pre-allocate
            # batch = np.ndarray((batch_size, 28, 28, 1))

            X = []
            y = []
            Z = []
            for batch_indexes in self.indexes_gen:
                break
            for random_image_index in batch_indexes:
                # Pre-allocate
                # batch[i:i+28]

                # Select random image, get image array and label
                Z.append(random_image_index)
                numpy_array = np.asarray(self._execute(self.augmentor_images[random_image_index], save_to_disk=False))
                label = self.augmentor_images[random_image_index].categorical_label

                # Reshape
                w = numpy_array.shape[0]
                h = numpy_array.shape[1]

                if np.ndim(numpy_array) == 2:
                    l = 1
                else:
                    l = np.shape(numpy_array)[2]

                if image_data_format == "channels_last":
                    numpy_array = numpy_array.reshape(w, h, l)
                elif image_data_format == "channels_first":
                    numpy_array = numpy_array.reshape(l, w, h)

                X.append(numpy_array)
                y.append(label)

            X = np.asarray(X)
            y = np.asarray(y)

            if scaled:
                X = X.astype('float32')
                X /= 255
            if with_indexes:
                yield (X, y, Z)
            else:
                yield (X, y)



    def keras_generator_from_array(self, images, labels, batch_size, scaled=True, image_data_format="channels_last"):
        """
        Returns an image generator that will sample from the current pipeline
        indefinitely, as long as it is called.
        .. warning::
         This function returns images from :attr:`images`
         **with replacement**.
        You must configure the generator to provide data in the same
        format that Keras is configured for. You can use the functions
        :func:`keras.backend.image_data_format()` and
        :func:`keras.backend.set_image_data_format()` to get and set
        Keras' image format at runtime.
        .. code-block:: python
            >>> from keras import backend as K
            >>> K.image_data_format()
            'channels_first'
            >>> K.set_image_data_format('channels_last')
            >>> K.image_data_format()
            'channels_last'
        By default, Augmentor uses ``'channels_last'``.
        :param images: The images to augment using the current pipeline.
        :type images: Array-like matrix. For greyscale images they can be
         in the form ``(l, x, y)`` or ``(l, x, y, 1)``, where
         :attr:`l` is the number of images, :attr:`x` is the image width
         and :attr:`y` is the image height. For RGB/A images, the matrix
         should be in the form ``(l, x, y, n)``, where :attr:`n` is the
         number of layers, e.g. 3 for RGB or 4 for RGBA and CMYK.
        :param labels: The label associated with each image in :attr:`images`.
        :type labels: List.
        :param batch_size: The number of images to return per batch.
        :type batch_size: Integer
        :param scaled: True (default) if pixels are to be converted
         to float32 values between 0 and 1, or False if pixels should be
         integer values between 0-255.
        :type scaled: Boolean
        :param image_data_format: Either ``'channels_last'`` (default) or
         ``'channels_first'``. When ``'channels_last'`` is specified the
         returned batch is in the form ``(batch_size, x, y, num_channels)``,
         while for ``'channels_last'`` the batch is returned in the form
         ``(batch_size, num_channels, x, y)``.
        :param image_data_format: String
        :return: An image generator.
        """

        # Here, we will expect an matrix in the shape (l, x, y)
        # where l is the number of images

        # Check if the labels and images align
        if len(images) != len(labels):
            raise IndexError("The number of images does not match the number of labels.")

        while True:

            X = []
            y = []

            for i in range(batch_size):

                random_image_index = random.randint(0, len(images)-1)

                # Before passing the image we must format it in a shape that
                # Pillow can understand, that is either (w, h) for greyscale
                # or (w, h, num_channels) for RGB, RGBA, or CMYK images.
                # PIL expects greyscale or B&W images in the form (w, h)
                # and RGB(A) images images in the form (w, h, n) where n is
                # the number of channels, which is 3 or 4.
                # However, Keras often works with greyscale/B&W images in the
                # form (w, h, 1). We will convert all images to (w, h) if they
                #  are not RGB, otherwise we will use (w, h, n).
                if np.ndim(images) == 3:
                    l = 1
                else:
                    l = np.shape(images)[-1]

                w = images[random_image_index].shape[0]
                h = images[random_image_index].shape[1]

                if l == 1:
                    numpy_array = self._execute_with_array(np.reshape(images[random_image_index], (w, h)))
                else:
                    numpy_array = self._execute_with_array(np.reshape(images[random_image_index], (w, h, l)))

                if image_data_format == "channels_first":
                    numpy_array = numpy_array.reshape(l, w, h)
                elif image_data_format == "channels_last":
                    numpy_array = numpy_array.reshape(w, h, l)

                X.append(numpy_array)
                y.append(labels[random_image_index])

            X = np.asarray(X)
            y = np.asarray(y)

            if scaled:
                X = X.astype('float32')
                X /= 255

            yield(X, y)

    def torch_transform(self):
        """
        Returns the pipeline as a function that can be used with torchvision.
        .. code-block:: python
            >>> import Augmentor
            >>> import torchvision
            >>> p = Augmentor.Pipeline()
            >>> p.rotate(probability=0.7, max_left_rotate=10, max_right_rotate=10)
            >>> p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
            >>> transforms = torchvision.transforms.Compose([
            >>>     p.torch_transform(),
            >>>     torchvision.transforms.ToTensor(),
            >>> ])
        :return: The pipeline as a function.
        """
        def _transform(image):
            for operation in self.operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    image = [image]
                    image = operation.perform_operation(image)

            return image

        return _transform

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
        print("Operations: %s" % len(self.operations))

        if len(self.operations) != 0:
            operation_index = 0
            for operation in self.operations:
                print("\t%s: %s (" % (operation_index, operation), end="")
                for operation_attribute, operation_value in operation.__dict__.items():
                    print("%s=%s " % (operation_attribute, operation_value), end="")
                print(")")
                operation_index += 1

        print("Images: %s" % len(self.augmentor_images))

        label_pairs = sorted(set([x.label_pair for x in self.augmentor_images]))

        print("Classes: %s" % len(label_pairs))

        for label_pair in label_pairs:
            print ("\tClass index: %s Class label: %s " % (label_pair[0], label_pair[1]))

        if len(self.augmentor_images) != 0:
            print("Dimensions: %s" % len(self.distinct_dimensions))
            for distinct_dimension in self.distinct_dimensions:
                print("\tWidth: %s Height: %s" % (distinct_dimension[0], distinct_dimension[1]))
            print("Formats: %s" % len(self.distinct_formats))
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

    # TODO: Implement
    # def subtract_mean(self, probability=1):
    #    # For implementation example, see bottom of:
    #    # https://patrykchrabaszcz.github.io/Imagenet32/
    #    self.add_operation(Mean(probability=probability))

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
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Rotate(probability=probability, rotation=-1))

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
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param max_left_rotation: The maximum number of degrees the image can
         be rotated to the left.
        :param max_right_rotation: The maximum number of degrees the image can
         be rotated to the right.
        :type probability: Float
        :type max_left_rotation: Integer
        :type max_right_rotation: Integer
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

    def rotate_without_crop(self, probability, max_left_rotation, max_right_rotation, expand=False):
        """
        Rotate an image without automatically cropping.
        The :attr:`expand` parameter controls whether the image is enlarged
        to contain the new rotated images, or if the image size is maintained
        Defaults to :attr:`false` so that images maintain their dimensions
        when using this function.
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param max_left_rotation: The maximum number of degrees the image can
         be rotated to the left.
        :param max_right_rotation: The maximum number of degrees the image can
         be rotated to the right.
        :type probability: Float
        :type max_left_rotation: Integer
        :type max_right_rotation: Integer
        :param expand: Controls whether the image's size should be
         increased to accommodate the rotation. Defaults to :attr:`false`
         so that images maintain their original dimensions after rotation.
        :return: None
        """
        self.add_operation(RotateStandard(probability=probability, max_left_rotation=ceil(max_left_rotation),
                                          max_right_rotation=ceil(max_right_rotation), expand=expand))

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

    def gaussian_distortion(self, probability, grid_width, grid_height, magnitude, corner, method, mex=0.5, mey=0.5,
                            sdx=0.05, sdy=0.05):
        """
        Performs a random, elastic gaussian distortion on an image.
        This function performs a randomised, elastic gaussian distortion controlled
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
        :return: None
        For values :attr:`mex`, :attr:`mey`, :attr:`sdx`, and :attr:`sdy` the
        surface is based on the normal distribution:
        .. math::
         e^{- \Big( \\frac{(x-\\text{mex})^2}{\\text{sdx}} + \\frac{(y-\\text{mey})^2}{\\text{sdy}} \Big) }
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(GaussianDistortion(probability=probability, grid_width=grid_width,
                                                  grid_height=grid_height,
                                                  magnitude=magnitude, corner=corner,
                                                  method=method,  mex=mex,
                                                  mey=mey, sdx=sdx, sdy=sdy))

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
        elif min_factor <= 0:
            raise ValueError("The min_factor argument must be greater than 0.")
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

    def random_erasing(self, probability, rectangle_area):
        """
        Work in progress. This operation performs a Random Erasing operation,
        as described in
        `https://arxiv.org/abs/1708.04896 <https://arxiv.org/abs/1708.04896>`_
        by Zhong et al.
        Its purpose is to make models robust to occlusion, by randomly
        replacing rectangular regions with random pixel values.
        For greyscale images the random pixels values will also be greyscale,
        and for RGB images the random pixels values will be in RGB.
        This operation is subject to change, the original work describes
        several ways of filling the random regions, including a random
        solid colour or greyscale value. Currently this operations uses
        the method which yielded the best results in the tests performed
        by Zhong et al.
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param rectangle_area: The percentage area of the image to occlude
         with the random rectangle, between 0.1 and 1.
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0.1 < rectangle_area <= 1:
            raise ValueError("The rectangle_area must be between 0.1 and 1.")
        else:
            self.add_operation(RandomErasing(probability=probability, rectangle_area=rectangle_area))

    def ground_truth(self, ground_truth_directory):
        """
        Specifies a directory containing corresponding images that
        constitute respective ground truth images for the images
        in the current pipeline.
        This function will search the directory specified by
        :attr:`ground_truth_directory` and will associate each ground truth
        image with the images in the pipeline by file name.
        Therefore, an image titled ``cat321.jpg`` will match with the
        image ``cat321.jpg`` in the :attr:`ground_truth_directory`.
        The function respects each image's label, therefore the image
        named ``cat321.jpg`` with the label ``cat`` will match the image
        ``cat321.jpg`` in the subdirectory ``cat`` relative to
        :attr:`ground_truth_directory`.
        Typically used to specify a set of ground truth or gold standard
        images that should be augmented alongside the original images
        of a dataset, such as image masks or semantic segmentation ground
        truth images.
        A number of such data sets are openly available, see for example
        `https://arxiv.org/pdf/1704.06857.pdf <https://arxiv.org/pdf/1704.06857.pdf>`_
        (Garcia-Garcia et al., 2017).
        :param ground_truth_directory: A directory containing the
         ground truth images that correspond to the images in the
         current pipeline.
        :type ground_truth_directory: String
        :return: None.
        """

        num_of_ground_truth_images_added = 0

        # Progress bar
        progress_bar = tqdm(total=len(self.augmentor_images), desc="Processing", unit=' Images', leave=False)

        if len(self.class_labels) == 1:
            for augmentor_image_idx in range(len(self.augmentor_images)):
                ground_truth_image = os.path.join(ground_truth_directory,
                                                  self.augmentor_images[augmentor_image_idx].image_file_name)
                if os.path.isfile(ground_truth_image):
                    self.augmentor_images[augmentor_image_idx].ground_truth = ground_truth_image
                    num_of_ground_truth_images_added += 1
        else:
            for i in range(len(self.class_labels)):
                for augmentor_image_idx in range(len(self.augmentor_images)):
                    ground_truth_image = os.path.join(ground_truth_directory,
                                                      self.augmentor_images[augmentor_image_idx].class_label,
                                                      self.augmentor_images[augmentor_image_idx].image_file_name)
                    if os.path.isfile(ground_truth_image):
                        if self.augmentor_images[augmentor_image_idx].class_label == self.class_labels[i][0]:
                            # Check files are the same size. There may be a better way to do this.
                            original_image_dimensions = \
                                Image.open(self.augmentor_images[augmentor_image_idx].image_path).size
                            ground_image_dimensions = Image.open(ground_truth_image).size
                            if original_image_dimensions == ground_image_dimensions:
                                self.augmentor_images[augmentor_image_idx].ground_truth = ground_truth_image
                                num_of_ground_truth_images_added += 1
                                progress_bar.update(1)

        progress_bar.close()

        # May not be required after all, check later.
        if num_of_ground_truth_images_added != 0:
            self.process_ground_truth_images = True

        print("%s ground truth image(s) found." % num_of_ground_truth_images_added)

    def get_ground_truth_paths(self):
        """
        Returns a list of image and ground truth image path pairs. Used for
        verification purposes to ensure the ground truth images match to the
        images containing in the pipeline.
        :return: A list of tuples containing the image path and ground truth
         path pairs.
        """
        paths = []

        for augmentor_image in self.augmentor_images:
            print("Image path: %s\nGround truth path: %s\n---\n" % (augmentor_image.image_path, augmentor_image.ground_truth))
            paths.append((augmentor_image.image_path, augmentor_image.ground_truth))

        return paths

class DataFramePipeline(Pipeline):
    def __init__(self, source_dataframe, image_col, category_col, output_directory="output", save_format=None):
        """
        Create a new Pipeline object pointing to dataframe containing the paths
        to your original image dataset.
        Create a new Pipeline object, using the :attr:`source_dataframe`
        and the columns :attr:`image_col` for the path of the image and
        :attr:`category_col` for the name of the cateogry
        :param source_dataframe: A Pandas DataFrame where the images are located
        :param output_directory: Specifies where augmented images should be
         saved to the disk. Default is the absolute path
        :param save_format: The file format to use when saving newly created,
         augmented images. Default is JPEG. Legal options are BMP, PNG, and
         GIF.
        :return: A :class:`Pipeline` object.
        """
        super(DataFramePipeline, self).__init__(source_directory = None,
                                                output_directory=output_directory,
                                                save_format=save_format)
        self._populate(source_dataframe,
                  image_col,
                  category_col,
                  output_directory,
                  save_format)

    def _populate(self,
                  source_dataframe,
                  image_col,
                  category_col,
                  output_directory,
                  save_format):
        # Assume we have an absolute path for the output
        # Scan the directory that user supplied.
        self.augmentor_images, self.class_labels = scan_dataframe(source_dataframe,
                                                                   image_col,
                                                                   category_col,
                                                                  output_directory)

        self._check_images(output_directory)


