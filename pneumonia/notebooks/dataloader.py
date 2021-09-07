import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from preprocess import Preprocessor
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

### Tensorflow dependencies ###
import tensorflow as tf

class DataLoader:
    def __init__(self, train_dir, batch_size, val_ratio):
        '''
            DataLoader constructor

            Arguments :
                train_dir : Data directory with images, masks sub directories
                batch_size : Number of instances per batch
                val_ratio : Split ratio between test - overall dataset
                unet_model : Path containing the U-net structure file
                unet_weights : Path containing the U-net weights
        '''
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.val_ratio = val_ratio

        train_normal_dir = f'{self.train_dir}/NORMAL'
        train_abnormal_dir = f'{self.train_dir}/PNEUMONIA'

        train_normal_img_files = glob.glob(f'{train_normal_dir}/*.jpeg')
        train_abnormal_img_files = glob.glob(f'{train_abnormal_dir}/*.jpeg')
    
        self.train_img_files = train_normal_img_files + train_abnormal_img_files
        self.train_classes = np.concatenate([np.ones((len(train_normal_img_files),)), np.zeros((len(train_abnormal_img_files),))])

    @staticmethod
    def map_fn(img, size=256):
        img = tf.image.resize(img, [size, size])
        img = tf.clip_by_value(img, 0, 255)
        
        return img

    @staticmethod
    def parse_fn(img_file, label):
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img, 3)
        img = DataLoader.map_fn(img)

        label = tf.cast(tf.convert_to_tensor(label), tf.int32)
        label = tf.one_hot(label, depth=2)

        return img, label

    def get_train_val_datasets(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.train_img_files, self.train_classes, test_size=self.val_ratio, random_state=4001)
        self.train_steps = len(X_train) // self.batch_size + 1
        self.val_steps = len(X_train) // self.batch_size + 1

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

