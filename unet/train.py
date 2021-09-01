import os
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from dataloader import DataLoader
from models import build_unet_model
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Path to the data directory with subdirectories "images" and "masks"')
parser.add_argument('--epochs', type=int, required=False, default=100, help='Number of training iterations')
parser.add_argument('--batch-size', type=int, required=False, default=16, help='Number of instances per batch')
parser.add_argument('--val-ratio', type=float, required=False, default=0.2, help='Ratio of validation/total dataset')
parser.add_argument('--lr', type=float, required=False, default=0.0005, help='Learning rate')
parser.add_argument('--save-path', type=str, required=False, default='../checkpoints', help='Path at which model and weights are saved')
args = vars(parser.parse_args())

@tf.function
def train_step(model, opt, batch):
    with tf.GradientTape() as tape:
        bce = BinaryCrossentropy(from_logits=True)
        images, masks = batch

        predicted_masks = model(images, training=True)
        loss = bce(masks, predicted_masks)

        gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

@tf.function
def val_step(model, batch):
    bce = BinaryCrossentropy(from_logits=True)
    images, masks = batch
    
    predictions = model(images, training=False)
    loss = bce(masks, predictions)

    return loss

def train(model, train_dataset, val_dataset, save_path='checkpoints', epochs=100, lr=0.0001, steps_per_epoch=50, val_steps=10, save_steps=5):
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)
        print('Checkpoint path created')

        model.save(f'{save_path}/model.h5')

    optimizer = Adam(lr=lr, amsgrad=True)

    for i in range(epochs):
        with tqdm.tqdm(total=steps_per_epoch) as pbar:
            for j in range(steps_per_epoch):
                batch = next(iter(train_dataset))
                train_loss = train_step(model, optimizer, batch)
                train_loss = train_loss.numpy()

                pbar.set_postfix({'train_loss' : f'{train_loss:.4f}'})
                pbar.update(1)
            
        with tqdm.tqdm(total=val_steps, colour='green') as pbar:
            for j in range(val_steps):
                batch = next(iter(val_dataset))
                val_loss = val_step(model, batch)
                val_loss = val_loss.numpy()

                pbar.set_postfix({'val_loss' : f'{val_loss:.4f}'})
                pbar.update(1)

        if((i + 1) % save_steps == 0):
            print('Checkpointing weights ...')
            model.save_weights(f'{save_path}/model.weights.hdf5')
            
model = build_unet_model()
data_dir = args['data'] 
batch_size = args['batch_size']
test_ratio = args['val_ratio']
epochs = args['epochs'] 
lr = args['lr']
save_path = args['save_path']

loader = DataLoader(data_dir, batch_size, test_ratio)
train_ds, val_ds = loader.get_train_val_datasets()
steps_per_epoch, val_steps = loader.train_steps, loader.val_steps

train(model, train_ds, val_ds, epochs=epochs, lr=lr, save_path=save_path, 
        steps_per_epoch=steps_per_epoch, val_steps=val_steps)

