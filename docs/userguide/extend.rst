.. _extendingaugmentor:

Extending Augmentor
===================

Extending Augmentor to add new functionality is quite simple, and is performed in two steps:

1) Create a custom class which subclasses from the :class:`.Operation` base class, and
2) Add an object of your new class to the pipeline using the :func:`~Augmentor.Pipeline.Pipeline.add_operation` function.

This allows you to add custom functionality and extend Augmentor at run-time. Of course, if you have written an operation that may be of benefit to the community, you can make a pull request on the GitHub repository.

The following sections describe extending Augmentor in two steps. Step 1 involves creating a new :class:`.Operation` subclass, and step 2 involves using an object of your new custom operation in a pipeline.

Step 1: Create a New Operation Subclass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a custom operation and extend Augmentor:

1) You create a new class that inherits from the :class:`.Operation` base class.
2) You must overload the :func:`~Augmentor.Operations.Operation.perform_operation` method belonging to the superclass.
3) You must call the superclass's :func:`__init__` constructor.
4) You must return an object of type :class:`PIL.Image`.

For example, to add a new operation called ``FoldImage``, you would write this code:

.. code-block:: python

    # Create your new operation by inheriting from the Operation superclass:
    class FoldImage(Operation):
        # Here you can accept as many custom parameters as required:
        def __init__(self, probability, num_of_folds):
            # Call the superclass's constructor (meaning you must
            # supply a probability value):
            Operation.__init__(self, probability)
            # Set your custom operation's member variables here as required:
            self.num_of_folds = num_of_folds

        # Your class must implement the perform_operation method:
        def perform_operation(self, image):
            # Start of code to perform custom image operation.
            for fold in range(self.num_of_folds):
                pass
            # End of code to perform custom image operation.

            # Return the image so that it can further processed in the pipeline:
            return image

You have seen that you need to implement the :func:`~Augmentor.Operations.Operation.perform_operation` function and you must call the superclass's constructor which requires a :attr:`probability` value to be set. Ensure you return a PIL Image as a return value.

If you wish to make these changes permanent, place your code in the :mod:`~Augmentor.Operations` **module**.

.. hint::

    You can also overload the superclass's :func:`~Augmentor.Operations.Operation.__str__` function to return a custom string for the object's description text. This is useful for some methods that display information about the operation, such as the :func:`~Augmentor.Pipeline.Pipeline.status` method.

Step 2: Add an Object to the Pipeline Manually
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have a new operation which is of type :class:`.Operation`, you can add an object of your new operation to an existing pipeline.

.. code-block:: python

    # Instantiate a new object of your custom operation
    fold = Fold(probability = 0.75, num_of_folds = 4)

    # Add this to the current pipeline
    p.add_operation(fold)

    # Executed the pipeline as normal, and your custom operation will be executed
    p.sample(1000)

As you can see, adding custom operations at run-time is possible by subclassing the :class:`.Operation` class and adding an object of this class to the pipeline manually using the :func:`~Augmentor.Pipeline.Pipeline.add_operation` function.

Using non-PIL Image Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Images can be converted to their raw formats for custom operations, for example by using NumPy:

.. code-block:: python

    import numpy

    # Custom class declaration

    def perform_operation(image):

        image_array = numpy.array(image).astype('uint8')

        # Perform your custom operations here

        image = PIL.Image.fromarray(image_array)

        return image
