import os
import glob
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from dataloader import DataLoader
from models import EMA_Unet # build_unet_model
from custom_callbacks import GifCreator, EarlyStopping, InfoLogger
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Path to the data directory with subdirectories "images" and "masks"')
parser.add_argument('--u-data', type=str, required=False, default=None, help='Path to the unlabelled dataset')
parser.add_argument('--momentum', type=str, required=False, default=0.999, help='EMA momentum')
parser.add_argument('--epochs', type=int, required=False, default=100, help='Number of training iterations')
parser.add_argument('--batch-size', type=int, required=False, default=8, help='Number of instances per batch')
parser.add_argument('--u-batch-size', type=int, required=False, default=16, help='Number of instances per batch in unsupervised training')
parser.add_argument('--val-ratio', type=float, required=False, default=0.2, help='Ratio of validation/total dataset')
parser.add_argument('--lr', type=float, required=False, default=0.00005, help='Learning rate')
parser.add_argument('--save-path', type=str, required=False, default='../checkpoints', help='Path at which model and weights are saved')
parser.add_argument('--log-dir', type=str, required=False, default='../checkpoints', help='Base directory to store training logs')
args = vars(parser.parse_args())

@tf.function
def train_step(model, opt, batch):
    with tf.GradientTape() as tape:
        bce = BinaryCrossentropy(from_logits=False)
        images, masks = batch

        predicted_masks = model(images, training=True)
        loss = bce(masks, predicted_masks)

        gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

@tf.function
def train_step_unsupervised(ema_model, opt, batch, alpha=0.05):
    with tf.GradientTape() as tape:
        bce = BinaryCrossentropy(from_logits=False)
        strong_aug, weak_aug = batch

        pseudo_label = ema_model.teacher(weak_aug, training=False)
        predicted_masks = ema_model.student(strong_aug, training=True)
        loss = alpha * bce(pseudo_label, predicted_masks)

        gradients = tape.gradient(loss, ema_model.student.trainable_variables)
    opt.apply_gradients(zip(gradients, ema_model.student.trainable_variables))

    return loss

@tf.function
def val_step(model, batch):
    bce = BinaryCrossentropy(from_logits=False)
    images, masks = batch
    
    predictions = model(images, training=False)
    loss = bce(masks, predictions)

    return loss

def train(model, train_dataset, val_dataset, u_dataset=None, save_path='checkpoints', 
        epochs=100, lr=0.0001, steps_per_epoch=50, unsupervised_steps=50, val_steps=10, save_steps=5, callbacks=None):
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)
        print('Checkpoint path created')
        model.student.save(f'{save_path}/ema_model_student.h5')
        model.teacher.save(f'{save_path}/ema_model_teacher.h5')

    optimizer = Adam(lr=lr, amsgrad=True)

    train_losses = []
    val_losses = []
    for i in range(epochs):
        print(f'Epoch #[{i+1}/{epochs}]')

        ### Run through train supervised data directory ###
        with tqdm.tqdm(total=steps_per_epoch) as pbar:
            for batch in train_dataset:
                train_loss = train_step(model.student, optimizer, batch)
                train_loss = train_loss.numpy()

                train_losses.append(train_loss)
                pbar.set_postfix({'train_loss' : f'{train_loss:.4f}'})
                pbar.update(1)

        # EMA update after supervised learning
        model.update_ema_params(i+1)

        ### Run through unsupervised data directory ###
        if(u_dataset is not None):
            with tqdm.tqdm(total=unsupervised_steps, colour='red') as pbar:
                for batch in u_dataset:
                    unsupervised_loss = train_step_unsupervised(model, optimizer, batch)
                    unsupervised_loss = unsupervised_loss.numpy()

                    pbar.set_postfix({'unsupervised_loss' : f'{unsupervised_loss:.4f}'})
                    pbar.update(1)

        # EMA update after unsupervised learning
        model.update_ema_params(i+1)
           
        ### Run through val supervised data directory ###
        with tqdm.tqdm(total=val_steps, colour='green') as pbar:
            for batch in val_dataset:
                val_loss = val_step(model.student, batch)
                val_loss = val_loss.numpy()

                val_losses.append(val_loss)
                pbar.set_postfix({'val_loss' : f'{val_loss:.4f}'})
                pbar.update(1)

        # Compute mean losses
        mean_train_loss = np.array(train_losses).mean()
        mean_val_loss = np.array(val_losses).mean()

        # Save models
        if((i + 1) % save_steps == 0):
            print('Checkpointing weights ...')
            model.student.save_weights(f'{save_path}/ema_model_student.weights.hdf5')
            model.teacher.save_weights(f'{save_path}/ema_model_teacher.weights.hdf5')
            
        # Callbacks
        stop_training = False
        if(callbacks is not None):
            for callback in callbacks:
                callback.reset_state({
                    'model' : model.student,
                    'mean_train_loss' : mean_train_loss,
                    'mean_val_loss' : mean_val_loss
                })
                callback()

                if(callback.stop_training):
                    stop_training = True

        if(stop_training) : break

data_dir = args['data'] 
momentum = args['momentum']
u_data_dir = args['u_data']
batch_size = args['batch_size']
u_batch_size = args['u_batch_size']
test_ratio = args['val_ratio']
epochs = args['epochs'] 
lr = args['lr']
save_path = args['save_path']
model = EMA_Unet(momentum=momentum)

# Create data loader
loader = DataLoader(data_dir, u_data_dir, batch_size, u_batch_size, test_ratio)
train_ds, val_ds = loader.get_train_val_datasets()
u_ds = loader.get_unlabelled_dataset()
steps_per_epoch, val_steps = loader.train_steps, loader.val_steps
unsupervised_steps = loader.unsupervised_steps

# Create callbacks
test_file = np.random.choice(glob.glob('../data/LungSegments/images/*.png'))
gif_creator = GifCreator(test_file)
early_stop = EarlyStopping(monitor='mean_val_loss', patience=2)
info_logger = InfoLogger(args['log_dir'], 'lung-segmentation')

train(model, train_ds, val_ds, u_dataset=u_ds,epochs=epochs, lr=lr, save_path=save_path, 
        steps_per_epoch=steps_per_epoch, unsupervised_steps=unsupervised_steps,val_steps=val_steps, 
        callbacks=[gif_creator, early_stop, info_logger])

