from Augmentor import glob
from Augmentor import ImageDetails
from Augmentor import ImageFormat
from Augmentor import Image
from Augmentor import GithubFlavoredMarkdownTable
from Augmentor import gcd


class ImageSource(object):
    def __init__(self, root_path=None, image=None):

        self.root_path = root_path
        self.list_of_images = []
        self.file_extension = ('jpg', 'jpeg', 'png', 'bmp')
        if image is not None:
            self.list_of_images.append(ImageDetails.ImageDetails(image, image.filename))
        else:
            self.scan(root_path)
        self.dimensions = {}
        self.image_format = ImageFormat.ImageFormat()

    def scan(self, path_to_scan):
        # for root, subdirs, list_of_files in os.walk(path_to_scan):
        list_of_files = []
        for type in self.file_extension:
            list_of_files.extend(glob.glob(path_to_scan + '/**/*.' + type, recursive=True))

        for file in list_of_files:
            image = Image.open(file)
            self.list_of_images.append(ImageDetails.ImageDetails(image, file))

    def setup_summary(self):
        for image in self.list_of_images:

            if image.dimensions in self.dimensions:
                self.dimensions[image.dimensions] += 1
            else:
                self.dimensions[image.dimensions] = 1

    def summary(self):
        self.setup_summary()
        file_types = self.process_filetypes()
        table_data = [
            ['ImageSource Summary:', ''],
            ['Image count', str(len(self.list_of_images))],
            ['Dimensions', self.process_dimensions()],
            ['Aspect ratio(s)', self.process_aspect_ratio()],
            ['File type(s)', file_types[1]],
            ['File extension(s)', file_types[0]],
            ['File format(s)', self.process_file_formats()[0]],
        ]
        table = GithubFlavoredMarkdownTable(table_data)
        print(table.table)

    def process_dimensions(self):
        result = ""
        for dimension in sorted(self.dimensions)[:2]:
            result += str(dimension[0]) + 'x' + str(dimension[1]) + '(' + str(self.dimensions.get(dimension)) + ')\n'

        if len(self.dimensions) > 2:
            result += 'and more...\n'

        return result[:-1]

    def process_file_formats(self):
        file_formats = {}
        for image in self.list_of_images:
            if image.image.mode not in file_formats:
                file_formats[image.image.mode] = self.image_format.get_format_printable(image.image)

        return [str(', '.join(str(file_format) for file_format in file_formats.values())), str(len(file_formats))]
        #return 'lol'

    def process_filetypes(self):
        file_types = []
        for image in self.list_of_images:
            if image.extension not in file_types:
                file_types.append(image.extension)

        return [str(', '.join(str(type[1:]) for type in file_types)), str(len(file_types))]

    def process_aspect_ratio(self):
        aspect_ratios = []
        for width, height in self.dimensions.keys():
            ratio = gcd(width, height)
            if [width // ratio, height // ratio] not in aspect_ratios:
                aspect_ratios.append([width // ratio, height // ratio])

        result = '\n'.join(str(ratio[0]) + ':' + str(ratio[1]) for ratio in aspect_ratios[:2])
        if len(aspect_ratios) > 2:
            result += '\nand more...'

        return result