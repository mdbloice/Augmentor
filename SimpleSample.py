from Augmentor import Pipeline
from Augmentor import Image


## set the path to the folder with images to manipulate
pathToImages = 'Path/To/Image/Folder'
## create pipeline
pipe = Pipeline.Pipeline(image_path=pathToImages)
## add functions to pipeline
pipe.addFlipY()
pipe.addFlipX(1)
## add another function to pipeline
pipe.addCrop(45, 45)

## print a summary
pipe.summary()
pipe.image_source.summary()

## execute the functions in pipline
pipe.execute()

## all images are created in target folder

## Example of how to hand over a image to the pipeline, details on creating an image with pillow see: http://pillow.readthedocs.org/en/3.1.x/reference/Image.html
im = Image.open("Path/to/an/image")
pipe = Pipeline.Pipeline(image=im)
pipe.addFlipY()
pipe.summary()
pipe.image_source.summary()
pipe.execute()

