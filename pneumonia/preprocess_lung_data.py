import os
import cv2
import glob
import numpy as np
from preprocess import Preprocessor

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Path to the input data with lung x-ray images')
parser.add_argument('--output', type=str, required=True, help='Path to the output data')
args = vars(parser.parse_args())

preprocessor = Preprocessor('../checkpoints/model.h5', '../checkpoints/model.weights.hdf5')
img_files = glob.glob(f'{args["input"]}/*.jpeg')
num_files = len(img_files)

for i, img_file in enumerate(img_files):
    filename = img_file.split('/')[-1]
    output_file = os.path.join(args['output'], filename)

    img = cv2.imread(img_file)
    img = cv2.resize(img, (256, 256))

    output = preprocessor.lung_region_normalization(img)
    output = (output * 255.0).astype(np.uint8)

    cv2.imwrite(output_file, output)

    print(f'[INFO] File #[{i+1}/{num_files}] processed ... ')
