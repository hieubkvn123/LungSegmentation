import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from dataloader import DataLoader
from models import build_unet_model

data_dir = '../data/LungSegments'
batch_size = 16
test_ratio = 0.2

loader = DataLoader(data_dir, batch_size, test_ratio)
train_ds, val_ds = loader.get_train_val_datasets()

images, masks = next(iter(train_ds))
print(np.unique(masks))
