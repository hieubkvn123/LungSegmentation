import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from dataloader import DataLoader
from models import build_unet_model

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
            
        with tqdm.tqdm(total=validation_steps, colour='green') as pbar:
            for j in range(validation_steps):
                batch = next(iter(val_dataset))
                val_loss = val_step(model, batch)
                val_loss = val_loss.numpy()

                pbar.set_postfix({'val_loss' : f'{val_loss:.4f}'})
                pbar.update(1)

        if((i + 1) % save_steps):
            print('Checkpointing weights ...')
            model.save_weights(f'{save_path}/model.weights.hdf5')
            
model = build_unet_model()
data_dir = '../data/LungSegments'
batch_size = 16
test_ratio = 0.2
epochs = 100
lr=0.001

loader = DataLoader(data_dir, batch_size, test_ratio)
train_ds, val_ds = loader.get_train_val_datasets()
steps_per_epoch, val_steps = loader.train_steps, loader.val_steps

train(model, train_ds, val_ds, epochs=epochs, lr=lr, 
        steps_per_epoch=steps_per_epoch, val_steps=val_steps)

