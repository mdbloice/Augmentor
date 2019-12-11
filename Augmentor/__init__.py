"""
The Augmentor image augmentation library.

Augmentor is a software package for augmenting image data. It provides a number of utilities that aid augmentation \
in a automated manner. The aim of the package is to make augmentation for machine learning tasks less prone to \
error, more reproducible, more efficient, and easier to perform.

.. moduleauthor:: Marcus D. Bloice <marcus.bloice@medunigraz.at>
   :platform: Windows, Linux, Macintosh
   :synopsis: An image augmentation library for Machine Learning.

"""

from .Pipeline import Pipeline
from .NumpyPipeline import NumpyPipeline
from .Pipeline import Pipeline, DataFramePipeline, DataPipeline

__author__ = """Marcus D. Bloice"""
__email__ = 'marcus.bloice@medunigraz.at'
__version__ = '0.2.7'

__all__ = ['Pipeline', 'DataFramePipeline', 'DataPipeline']
