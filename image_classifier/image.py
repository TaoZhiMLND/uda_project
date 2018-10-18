from PIL import Image

import numpy as np

def process_image(image):
    image = Image.open(image)
    width, height = image.size
    
    # resize
    new_size = ()
    
    if width >= height:
        new_size = (width * 256 / height, 256)
    else:
        new_size = (256, height * 256 / width)
    image.thumbnail(new_size)
    
    # crop
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image) / 255
    
    for i in range(224):
        for j in range(224):
            np_image[i][j][0] = (np_image[i][j][0] - 0.485) / 0.229
            np_image[i][j][1] = (np_image[i][j][1] - 0.456) / 0.224
            np_image[i][j][2] = (np_image[i][j][2] - 0.406) / 0.225
    
    np_image = np_image.transpose((2, 0, 1))
    return np_image