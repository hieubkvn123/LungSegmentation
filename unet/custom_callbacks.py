import os
import cv2
import imageio
import numpy as np
import tensorflow as tf

class GifCreator:
    def __init__(self, test_file, output_dir='gifs', fps=15):
        self.test_file = test_file
        self.image = cv2.imread(self.test_file, cv2.COLOR_BGR2RGB)
        self.image = (self.image - 127.5)/127.5
        self.output_dir = output_dir
        self.fps = fps
        self.counter = 0

        if(not os.path.exists(self.output_dir)):
            print('Creating GIF output directory ', self.output_dir)
            os.mkdir(self.output_dir)
    
    def log_output(self, model):
        self.counter += 1
        output = model.predict(np.array([self.image]))
        output = output[0]

        output_file = f'{self.output_dir}/output_{self.counter}.png'
        cv2.imwrite(output_file, output)
