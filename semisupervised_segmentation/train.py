import os
import glob
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

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
parser.add_argument('--u-batch-size', type=int, required=False, default=8, help='Number of instances per batch in unsupervised training')
parser.add_argument('--val-ratio', type=float, required=False, default=0.2, help='Ratio of validation/total dataset')
parser.add_argument('--lr', type=float, required=False, default=0.00005, help='Learning rate')
parser.add_argument('--save-path', type=str, required=False, default='../checkpoints', help='Path at which model and weights are saved')
parser.add_argument('--log-dir', type=str, required=False, default='../checkpoints', help='Base directory to store training logs')
args = vars(parser.parse_args())

@tf.function
def train_step(ema_model, opt, batch, step=1):
    # Compute consistency weight
    consistency = 100.0
    consistency_rampup = 5

    alpha = 0.01 # consistency * EMA_Unet.sigmoid_rampup(step, consistency_rampup)
    
    bce = BinaryCrossentropy(from_logits=False)
    mse = MeanSquaredError()
    
    with tf.GradientTape() as tape:
        weak_aug, strong_aug, masks = batch

        # 1. Calculate the classification loss
        predicted_masks, logits_weak = ema_model.student(weak_aug, training=True)
        cls_loss = bce(masks, predicted_masks)

        # 2. Calculate the consistency loss
        predicted_masks, logits_strong = ema_model.student(strong_aug, training=True)
        consistency_loss = alpha * mse(logits_weak, logits_strong)

        # 3. Overall loss
        loss = cls_loss + alpha * consistency_loss

        gradients = tape.gradient(loss, ema_model.student.trainable_variables)

    opt.apply_gradients(zip(gradients, ema_model.student.trainable_variables))

    return loss

@tf.function
def train_step_unsupervised(ema_model, opt, batch, step, alpha=0.005):
    # Compute consistency weight
    consistency = 100.0
    consistency_rampup = 5

    alpha = 0.01 # consistency * EMA_Unet.sigmoid_rampup(step, consistency_rampup)

    with tf.GradientTape() as tape:
        mse = MeanSquaredError()
        strong_aug, weak_aug = batch

        pseudo_label, logits_weak = ema_model.student(weak_aug, training=True)
        predicted_mask, logits_strong = ema_model.student(strong_aug, training=True)
        loss = alpha * mse(logits_weak, logits_strong)

        gradients = tape.gradient(loss, ema_model.student.trainable_variables)
    opt.apply_gradients(zip(gradients, ema_model.student.trainable_variables))

    return loss

@tf.function
def val_step(model, batch):
    bce = BinaryCrossentropy(from_logits=False)
    images, _, masks = batch
    
    predictions, logits = model(images, training=False)
    loss = bce(masks, predictions)

    return loss

def train(model, train_dataset, val_dataset, u_dataset=None, save_path='checkpoints', ema=False, 
        epochs=100, lr=0.0001, steps_per_epoch=50, unsupervised_steps=50, val_steps=10, save_steps=5, callbacks=None):
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)
        print('Checkpoint path created')
        model.student.save(f'{save_path}/ema_model_student.h5')
        model.teacher.save(f'{save_path}/ema_model_teacher.h5')

    optimizer = Adam(lr=lr, amsgrad=True)

    train_losses = []
    val_losses = []
    global_step = 0
    for i in range(epochs):
        print(f'Epoch #[{i+1}/{epochs}]')

        ### Run through train supervised data directory ###
        with tqdm.tqdm(total=steps_per_epoch) as pbar:
            for batch in train_dataset:
                train_loss = train_step(model, optimizer, batch, step=i+1)
                global_step += 1
                train_loss = train_loss.numpy()

                train_losses.append(train_loss)
                pbar.set_postfix({'train_loss' : f'{train_loss:.4f}'})
                pbar.update(1)

                # EMA update after supervised learning
                # model.update_ema_params(global_step, ema=ema)

        ### Run through unsupervised data directory ###
        if(u_dataset is not None):
            with tqdm.tqdm(total=unsupervised_steps, colour='red') as pbar:
                for batch in u_dataset:
                    unsupervised_loss = train_step_unsupervised(model, optimizer, batch, i+1)
                    global_step += 1
                    unsupervised_loss = unsupervised_loss.numpy()

                    pbar.set_postfix({'unsupervised_loss' : f'{unsupervised_loss:.4f}'})
                    pbar.update(1)

                    # EMA update after unsupervised learning
                    # model.update_ema_params(global_step, ema=ema)
           
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

ema = False # Whether to use Mean Teacher or not
data_dir = args['data'] 
momentum = args['momentum']
u_data_dir = args['u_data']
batch_size = args['batch_size']
u_batch_size = args['u_batch_size']
test_ratio = args['val_ratio']
epochs = args['epochs'] 
lr = args['lr']
save_path = args['save_path']
model = EMA_Unet(momentum=momentum, batchnorm=True)

# Create data loader
loader = DataLoader(data_dir, u_data_dir, batch_size, u_batch_size, test_ratio)
train_ds, val_ds = loader.get_train_val_datasets()
u_ds = loader.get_unlabelled_dataset()
steps_per_epoch, val_steps = loader.train_steps, loader.val_steps
unsupervised_steps = loader.unsupervised_steps

# Create callbacks
test_file = np.random.choice(glob.glob('../data/LungSegments/images/*.png'))
gif_creator = GifCreator(test_file)
early_stop = EarlyStopping(monitor='mean_val_loss', patience=5)
info_logger = InfoLogger(args['log_dir'], 'lung-segmentation')

train(model, train_ds, val_ds, u_dataset=u_ds, ema=ema, epochs=epochs, lr=lr, save_path=save_path, 
        steps_per_epoch=steps_per_epoch, unsupervised_steps=unsupervised_steps,val_steps=val_steps, 
        callbacks=[gif_creator, early_stop, info_logger])

