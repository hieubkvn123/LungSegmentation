import os
import cv2
import glob
import numpy as np
import tensorflow as tf

class GifCreator:
    def __init__(self, test_file, output_dir='gifs', fps=15):
        self.stop_training=False
        self.test_file = test_file
        self.image = cv2.imread(self.test_file, cv2.COLOR_BGR2RGB)
        if(len(self.image.shape) < 3):
            img = self.image
            self.image = np.zeros((self.image.shape[0], self.image.shape[1], 3))
            self.image[:, :, 0] = img
            self.image[:, :, 1] = img
            self.image[:, :, 2] = img

        self.image = cv2.resize(self.image, (256, 256))
        self.image = (self.image - 127.5)/127.5
        self.output_dir = output_dir
        self.fps = fps
        self.counter = 0

        if(not os.path.exists(self.output_dir)):
            print('Creating GIF output directory ', self.output_dir)
            os.mkdir(self.output_dir)
        else:
            # Clear the output directory
            for _file in glob.glob(f'{self.output_dir}/*.png'):
                os.remove(_file)

            print('Gif output directory cleared')

    def log_output(self, model):
        if(self.model is not None):
            self.counter += 1
            output = self.model.predict(np.array([self.image]))
            output = output[0]
            output = output * 255.0
            output = output.astype(np.uint8)

            output_file = f'{self.output_dir}/{self.counter}.png'
            cv2.imwrite(output_file, output)

    def reset_state(self, state):
        self.model = state['model']

    def __call__(self):
        self.log_output(self.model)

class EarlyStopping:
    def __init__(self, patience, monitor='mean_val_loss'):
        self.stop_training = False
        self.patience = patience
        self.monitor = monitor
        self.losses = []

    @staticmethod
    def _overfitting(losses, patience):
        head = losses[-patience:]

        return sorted(head) == head

    def reset_state(self, state):
        if(self.monitor not in state):
            raise Exception(f'"{self.monitor}" monitor is not in state dict ...')

        self.losses.append(state[self.monitor])
        
    def __call__(self):
        if(EarlyStopping._overfitting(self.losses, self.patience)):
            self.stop_training = True

