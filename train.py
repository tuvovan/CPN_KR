import os

from tensorflow.keras.utils import Progbar
from tensorflow.keras import Input, Model
from dataprocessing.dataset import *
from model.model import *
from model.utils import *

import model.resnet_backbone as backbone
import model.config as cfg
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def define_model(input_shape):
    input = Input(shape=input_shape)
    backbone = get_backbone_resnet50(input_shape)
    global_fms, global_out = GlobalNet(backbone=backbone)(input)
    refine_out = RefineNet()(global_fms)
    model = Model(inputs=input, outputs=[global_out, refine_out])
    return model


@tf.function
def train_step(images, targets, valids):
    with tf.GradientTape() as tape:
        predicted = model(images, training=True)
        loss_value = compute_loss(predicted, targets, valids)
        # grads = tf.gradients(loss_value, model.trainable_variables)
    grads = tape.gradient(loss_value, model.trainable_weights)
    return loss_value, grads


@tf.function
def val_step(model, images, targets, valids):
    val_logits = model(images, training=False)
    # Update val metrics
    val_loss = compute_loss(val_logits, targets, valids)
    return val_loss


input_shape = (256, 192, 3)

model = define_model(input_shape)
model.summary()

# Define data
train_aug = iaa.Sequential(
    [
        iaa.Flipud(0.3),
        # `Sometimes()` applies a function randomly to the inputs with
        # a given probability (0.3, in this case).
        iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.7, 1.35))),
    ]
)

test_aug = iaa.Sequential(
    [iaa.Fliplr(0.0)])

train_df = pd.read_csv(
    'coco/annotations/person_keypoints_train2017_.csv')
val_df = pd.read_csv('coco/annotations/person_keypoints_val2017.csv')

train_keys = list(train_df.path)
validation_keys = list(val_df.path)

train_dataset = KeyPointsDataset(
    train_keys, train_aug, train_df, BATCH_SIZE=cfg.BATCH_SIZE)
validation_dataset = KeyPointsDataset(
    validation_keys, test_aug, val_df, train=False, BATCH_SIZE=cfg.BATCH_SIZE)

print(f"Total batches in training set: {len(train_dataset)}")
print(f"Total batches in validation set: {len(validation_dataset)}")


epochs = cfg.EPOCHS
optimizer = tf.keras.optimizers.SGD(
    learning_rate=cfg.LEARNING_RATE, clipnorm=cfg.GRADIENT_CLIP_NORM)
min_loss = cfg.min_loss
metrics_names = ['train_loss', 'val_loss']

for epoch in range(epochs):
    print("\nEpoch {}/{}".format(epoch+1, epochs))

    pb = Progbar(len(train_dataset) * cfg.BATCH_SIZE,
                 stateful_metrics=metrics_names)
    train_dataset = KeyPointsDataset(
        train_keys, train_aug, train_df, BATCH_SIZE=cfg.BATCH_SIZE)
    validation_dataset = KeyPointsDataset(
        validation_keys, test_aug, val_df, train=False, BATCH_SIZE=cfg.BATCH_SIZE)

    # Iterate over the batches of the dataset.
    train_loss = 0
    for step, (images, keypoints, targets, valids) in enumerate(train_dataset):

        loss_value, grads = train_step(images, targets, valids)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_loss += loss_value
        show_loss = train_loss / (step+1)
        values = [('train_loss', show_loss)]
        pb.update(step*cfg.BATCH_SIZE, values=values)
    # Run a validation loop at the end of each epoch.
    val_loss = 0
    for i, (x_batch_val, y_batch_val, label_batch_val, valid_batch_val) in enumerate(validation_dataset):
        val_loss += val_step(model, x_batch_val,
                             label_batch_val, valid_batch_val)
    val_loss /= (i+1)
    values = [('train_loss', show_loss), ('val_loss', val_loss)]

    pb.update(len(train_dataset) * cfg.BATCH_SIZE, values=values)
    if val_loss < min_loss:
        print("Saving model to {}".format(cfg.save_path))
        model.save_weights(cfg.save_path)
        min_loss = val_loss

    train_dataset.on_epoch_end()
    validation_dataset.on_epoch_end()
