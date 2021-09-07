import cv2
import numpy as np
from unet import UnetSegmenter

class Preprocessor:
    def __init__(self, unet_model_file, unet_weights_file):
        self.segmenter = UnetSegmenter(unet_model_file, unet_weights_file)

    def lung_region_normalization(self, img): 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization locally
        clahe = cv2.createCLAHE(2.0, (16,16))
        gray = clahe.apply(gray)

        # Extract the lung region using the segmentation mask
        mask = self.segmenter.get_mask(img, threshold=0.75).reshape(gray.shape)
        extracted = mask * gray
 
        # Get the boundary of the lung region
        y, x = np.where(mask != 0)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # Extract the lung region and histogram equalize locally
        output = extracted[y_min:y_max, x_min:x_max]
        output = clahe.apply(output)

        return output

#img = cv2.imread('../images/test1.png')
#img = cv2.resize(img, (256, 256))

#preprocessor = Preprocessor('../checkpoints/model.h5', '../checkpoints/model.weights.hdf5')
#output = preprocessor.lung_region_normalization(img)
#output = (output * 255.0).astype(np.uint8)
#cv2.imwrite('output.png', output)
