from Augmentor import os


class ImageDetails(object):
    def __init__(self, image, filename):
        self.image = image
        self.fullpath = filename
        self.filename = os.path.basename(os.path.splitext(filename)[0])
        self.extension = os.path.splitext(filename)[1]
        self.directory = os.path.dirname(filename)
        (self.dimensions) = image.size
