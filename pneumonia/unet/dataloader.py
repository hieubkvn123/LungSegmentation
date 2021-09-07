import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

### Tensorflow dependencies ###
import tensorflow as tf

class DataLoader:
    def __init__(self, data_dir, batch_size, val_ratio):
        '''
            DataLoader constructor

            Arguments :
                data_dir : Data directory with images, masks sub directories
                batch_size : Number of instances per batch
                val_ration : Split ratio between test - overall dataset
        '''
        self.data_dir = data_dir 
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')

        self.batch_size = batch_size
        self.val_ratio = val_ratio

        # Pair the images and the mask
        counter = 0
        self.image_files = []
        self.mask_files = []

        for mask in glob.glob(f'{self.mask_dir}/*.png'):
            filename = mask.split('/')[-1]
            
            respective_img = f'{self.img_dir}/{filename}'
            if(respective_img in glob.glob(f'{self.img_dir}/*.png')):
                self.mask_files.append(mask)
                self.image_files.append(respective_img)
                
        print(f'Number of matching mask-image pairs : {len(self.mask_files)}')

    @staticmethod
    def map_fn(img, size=256):
        img = tf.image.resize(img, [size, size])
        img = tf.clip_by_value(img, 0, 255)
        #img = img / 127.5 - 1.0
        
        mean = tf.math.reduce_mean(img)
        std = tf.math.reduce_std(img)
        img = (img - mean) / std

        return img

    @staticmethod
    def parse_fn(img_file, mask_file):
        img = tf.io.read_file(img_file)
        img = tf.image.decode_png(img, 3)
        img = DataLoader.map_fn(img)

        mask = tf.io.read_file(mask_file)
        mask = tf.image.decode_png(mask, 1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask = tf.image.resize(mask, [256, 256])
        mask = mask / 255.0

        return img, mask

    def get_train_val_datasets(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.image_files, self.mask_files, test_size=self.val_ratio)
        self.train_steps = len(X_train) // self.batch_size + 1
        self.val_steps = len(X_test) // self.batch_size + 1

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.map(DataLoader.parse_fn)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.take(self.train_steps)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        val_dataset = val_dataset.map(DataLoader.parse_fn)
        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.take(self.val_steps)

        return train_dataset, val_dataset

