import os
import glob
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from dataloader import DataLoader
from custom_callbacks import GifCreator, EarlyStopping
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Path to the data directory with subdirectories "images" and "masks"')
parser.add_argument('--epochs', type=int, required=False, default=100, help='Number of training iterations')
parser.add_argument('--batch-size', type=int, required=False, default=16, help='Number of instances per batch')
parser.add_argument('--val-ratio', type=float, required=False, default=0.2, help='Ratio of validation/total dataset')
parser.add_argument('--lr', type=float, required=False, default=0.00005, help='Learning rate')
parser.add_argument('--save-path', type=str, required=False, default='../checkpoints', help='Path at which model and weights are saved')
args = vars(parser.parse_args())

acc = CategoricalAccuracy()
@tf.function
def train_step(model, opt, batch):
    with tf.GradientTape() as tape:
        bce = BinaryCrossentropy(from_logits=False)
        images, labels = batch

        predictions = model(images, training=True)
        loss = bce(labels, predictions)
        accuracy = acc(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, accuracy

@tf.function
def val_step(model, batch):
    bce = BinaryCrossentropy(from_logits=False)
    images, labels = batch
    
    predictions = model(images, training=False)
    loss = bce(labels, predictions)
    accuracy = acc(labels, predictions)

    return loss, accuracy

def build_vgg_model():
    backbone = tf.keras.applications.vgg19.VGG19(include_top=False, weights=None, input_shape=(256, 256, 3), pooling='avg')
    backbone.trainable = True

    ### Add regularizer to backbone model ###
    reg = regularizers.l1(2e-3)

    for layer in backbone.layers:
        if(hasattr(layer, 'kernel_regularizer')):
            setattr(layer, 'kernel_regularizer', reg)
    
        if(hasattr(layer, 'kernel_initializer')):
            setattr(layer, 'kernel_initializer', 'he_normal')

    ### Add head ###
    inputs = Input(shape=(256, 256, 3))
    outputs = backbone(inputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2, activation='softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='Lung_Diagnosis_Network')
    return model

def train(model, train_dataset, val_dataset, save_path='checkpoints', 
        epochs=100, lr=0.0001, steps_per_epoch=50, val_steps=10, save_steps=5, callbacks=None):
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)
        print('Checkpoint path created')

    if(not os.path.exists(f'{save_path}/vgg.h5')):
        model.save(f'{save_path}/vgg.h5')

    optimizer = Adam(lr=lr, amsgrad=True)

    train_losses = []
    val_losses = []
    for i in range(epochs):
        print(f'Epoch #[{i+1}/{epochs}]')
        with tqdm.tqdm(total=steps_per_epoch) as pbar:
            for batch in train_dataset:
                train_loss, train_acc = train_step(model, optimizer, batch)
                train_loss = train_loss.numpy()
                train_acc = train_acc.numpy()

                train_losses.append(train_loss)
                pbar.set_postfix({'train_loss' : f'{train_loss:.4f}', 'train_acc' : f'{train_acc:.4f}'})
                pbar.update(1)
           
        with tqdm.tqdm(total=val_steps, colour='green') as pbar:
            for batch in val_dataset:
                val_loss, val_acc = val_step(model, batch)
                val_loss = val_loss.numpy()
                val_acc = val_acc.numpy()

                val_losses.append(val_loss)
                pbar.set_postfix({'val_loss' : f'{val_loss:.4f}', 'val_acc' : f'{val_acc:.4f}'})
                pbar.update(1)

        # Compute mean losses
        mean_train_loss = np.array(train_losses).mean()
        mean_val_loss = np.array(val_losses).mean()

        # Save models
        if((i + 1) % save_steps == 0):
            print('Checkpointing weights ...')
            model.save_weights(f'{save_path}/vgg.weights.hdf5')
            
        # Callbacks
        if(callbacks is not None):
            for callback in callbacks:
                callback.reset_state({
                    'model' : model,
                    'mean_train_loss' : mean_train_loss,
                    'mean_val_loss' : mean_val_loss
                })
                callback()

                if(callback.stop_training):
                    break

model = build_vgg_model() 
data_dir = args['data'] 
batch_size = args['batch_size']
test_ratio = args['val_ratio']
epochs = args['epochs'] 
lr = args['lr']
save_path = args['save_path']

# Create data loader
loader = DataLoader(data_dir, batch_size, test_ratio)
train_ds, val_ds = loader.get_train_val_datasets()
steps_per_epoch, val_steps = loader.train_steps, loader.val_steps

# Create callbacks
early_stop = EarlyStopping(monitor='mean_val_loss', patience=2)

train(model, train_ds, val_ds, epochs=epochs, lr=lr, save_path=save_path, 
        steps_per_epoch=steps_per_epoch, val_steps=val_steps, callbacks=[early_stop])

