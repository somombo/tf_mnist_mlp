from matplotlib import pyplot as plt

import numpy as np
import math


def disp(images, IMAGE_COLOR = 'bwr'):  
    def _disp_one(image):
        n = math.sqrt(image.shape[0])
        # print(image.shape)
        pixels = np.reshape(image, (math.ceil(n),math.floor(n)))
        img = plt.imshow(pixels, cmap=IMAGE_COLOR, interpolation='nearest')

        plt.axis('off')  
        
    fig = plt.figure()
    
    m = math.sqrt(images.shape[0])
    # print(images.shape)
    for i, img in enumerate(images): 
        a=fig.add_subplot(math.ceil(m),math.floor(m),i+1)
        _disp_one(img)

    plt.show()


# disp(data['train'].images[0:4,:])    