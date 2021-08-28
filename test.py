import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from segmentation import lung_boundary_detection
from argparse import ArgumentParser

matplotlib.use('TkAgg')
parser = ArgumentParser()
parser.add_argument('--input', required=True, type=str, help='Path of input image')
args = vars(parser.parse_args())

img = cv2.imread(args['input'])
output = lung_boundary_detection(img, preprocessing='clahe')

fig, ax = plt.subplots(1,2, figsize=(15, 8))
ax[0].imshow(img)
ax[0].set_title('Original Image')

ax[1].imshow(output)
ax[1].set_title('Segmentation Output')

plt.show()
