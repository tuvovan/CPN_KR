from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
from matplotlib import pyplot as plt
from ast import literal_eval
from tensorflow.keras import utils as KU
from model.utils import *
from PIL import Image

import imgaug.augmenters as iaa
import tensorflow as tf
import pandas as pd
import numpy as np
import model.config as cfg
import random
import os

# Utility for reading an image and for getting its annotations.


def get_data(name, df, train=False):
    IMG_DIR = "coco/images/train2017_crop" if train else "coco/images/val2017_crop"
    keypoints = df.loc[df['path'] == name, 'keypoints'].iloc[0]
    k_vis = df.loc[df['path'] == name, 'vis'].iloc[0].replace(' ', ',')
    keypoints = literal_eval(keypoints)
    k_vis = literal_eval(k_vis)
    keypoints = np.array(keypoints).reshape(-1, 2)
    data = {"keypoints": keypoints}
    img_data = plt.imread(os.path.join(IMG_DIR, name))
    # If the image is RGBA convert it to RGB.
    if img_data.shape[-1] == 4:
        img_data = img_data.astype(np.uint8)
        img_data = Image.fromarray(img_data)
        img_data = np.array(img_data.convert("RGB"))
    elif len(img_data.shape) != 3:
        img_data = img_data.astype(np.uint8)
        img_data = np.stack((img_data,)*3, axis=-1)
        img_data = Image.fromarray(img_data)
        img_data = np.array(img_data.convert("RGB"))
    data["img_data"] = np.array(img_data, dtype=np.float)
    data["k_vis"] = np.array(k_vis, dtype=np.float)

    return data


class KeyPointsDataset(KU.Sequence):
    def __init__(self, image_keys, aug, df, BATCH_SIZE=32, train=True):
        self.image_keys = image_keys
        self.aug = aug
        self.BATCH_SIZE = BATCH_SIZE
        self.train = train
        self.df = df
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_keys) // self.BATCH_SIZE

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_keys))
        if self.train:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.BATCH_SIZE: (index + 1) * self.BATCH_SIZE]
        image_keys_temp = [self.image_keys[k] for k in indexes]
        (images, keypoints, targets, valids) = self.__data_generation(image_keys_temp)

        # return (images, keypoints, targets, valids)
        return (images, keypoints, targets, valids)

    def __data_generation(self, image_keys_temp):
        batch_images = np.empty(
            (self.BATCH_SIZE, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3), dtype=np.float32)
        batch_keypoints = np.empty(
            (self.BATCH_SIZE, 1, 1, cfg.NUM_KEYPOINTS), dtype=np.float32)

        targets = []
        valids = []

        for i, key in enumerate(image_keys_temp):
            data = get_data(key, self.df, self.train)
            current_keypoint = np.array(data["keypoints"])[:, :2]
            k_vis = np.array(data["k_vis"])
            kps = []

            target15 = np.zeros(
                (cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1], cfg.NR_SKELETON))
            target11 = np.zeros(
                (cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1], cfg.NR_SKELETON))
            target9 = np.zeros(
                (cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1], cfg.NR_SKELETON))
            target7 = np.zeros(
                (cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1], cfg.NR_SKELETON))
            idx = np.where(k_vis == 0)
            current_keypoint[idx, :2] = -1000000

            # To apply our data augmentation pipeline, we first need to
            # form Keypoint objects with the original coordinates.
            for j in range(0, len(current_keypoint)):
                kps.append(
                    Keypoint(x=current_keypoint[j][0], y=current_keypoint[j][1]))

            # We then project the original image and its keypoint coordinates.
            current_image = data["img_data"]
            current_image = current_image - cfg.pixel_means
            current_image = current_image / 255
            kps_obj = KeypointsOnImage(kps, shape=current_image.shape)

            # Apply the augmentation pipeline
            (new_image, new_kps_obj) = self.aug(
                image=current_image, keypoints=kps_obj)

            batch_images[i, ] = new_image

            # Parse the coordinates from the new keypoint object.
            kp_temp = []
            kp_temp_out = []
            for keypoint in new_kps_obj:
                kp_temp.append(np.nan_to_num(keypoint.x / cfg.IMG_WIDTH))
                kp_temp.append(np.nan_to_num(keypoint.y / cfg.IMG_HEIGHT))

                kp_temp_out.append(
                    min(np.nan_to_num(keypoint.x) // 4, cfg.OUTPUT_SHAPE[1]-1))
                kp_temp_out.append(
                    min(np.nan_to_num(keypoint.y) // 4, cfg.OUTPUT_SHAPE[0]-1))

            kp_temp = np.array(kp_temp).reshape(-1, 2)
            kp_temp_out = np.array(kp_temp_out).reshape(-1, 2)

            for k in range(cfg.NR_SKELETON):
                if k_vis[k] > 0:  # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
                    if kp_temp_out[k][0] < 0. or kp_temp_out[k][1] < 0.:
                        continue
                    target15[:, :, k] = generate_heatmap(
                        target15[:, :, k], kp_temp_out[k], cfg.gk15)
                    target11[:, :, k] = generate_heatmap(
                        target11[:, :, k], kp_temp_out[k], cfg.gk11)
                    target9[:, :, k] = generate_heatmap(
                        target9[:, :, k], kp_temp_out[k], cfg.gk9)
                    target7[:, :, k] = generate_heatmap(
                        target7[:, :, k], kp_temp_out[k], cfg.gk7)

            target = [np.array(target15), np.array(target11),
                      np.array(target9), np.array(target7)]
            valid = k_vis

            batch_keypoints[i, ] = np.array(kp_temp).reshape(1, 1, 17 * 2)
            targets.append(target)
            valids.append(valid)

        batch_images = batch_images.astype(np.float32)
        batch_keypoints = batch_keypoints.astype(np.float32)
        targets = np.array(targets).astype(np.float32)
        valids = np.array(valids).astype(np.float32)
        return batch_images, batch_keypoints, targets, valids


def visualize_keypoints(images, keypoints):
    fig, axes = plt.subplots(nrows=len(images), ncols=2, figsize=(16, 12))
    [ax.axis("off") for ax in np.ravel(axes)]

    for (ax_orig, ax_all), image, current_keypoint in zip(axes, images, keypoints):
        ax_orig.imshow(image)
        ax_all.imshow(image)

        # If the keypoints were formed by `imgaug` then the coordinates need
        # to be iterated differently.
        if isinstance(current_keypoint, KeypointsOnImage):
            for idx, kp in enumerate(current_keypoint.keypoints):
                ax_all.scatter(
                    [kp.x], [kp.y], marker="x", s=50, linewidths=5
                )
        else:
            current_keypoint = np.array(current_keypoint)
            # Since the last entry is the visibility flag, we discard it.
            current_keypoint = current_keypoint[:, :2]
            for idx, (x, y) in enumerate(current_keypoint):
                if x < 0 or y < 0:
                    continue
                ax_all.scatter([x], [y],
                               marker="x", s=50, linewidths=5)

    plt.tight_layout(pad=2.0)
    plt.savefig('test_augment.png')


if __name__ == '__main__':
    train_aug = iaa.Sequential(
        [
            iaa.Flipud(0.3),
            # `Sometimes()` applies a function randomly to the inputs with
            # a given probability (0.3, in this case).
            iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7))),
        ]
    )

    test_aug = iaa.Sequential(
        [iaa.Fliplr(0.0)])

    train_df = pd.read_csv(
        'coco/annotations/person_keypoints_train2017.csv')
    val_df = pd.read_csv('coco/annotations/person_keypoints_val2017.csv')

    train_keys = list(train_df.path)
    validation_keys = list(val_df.path)

    train_dataset = KeyPointsDataset(train_keys, train_aug, train_df)
    validation_dataset = KeyPointsDataset(
        validation_keys, test_aug, train=False, df=val_df)

    print(f"Total batches in training set: {len(train_dataset)}")
    print(f"Total batches in validation set: {len(validation_dataset)}")

    sample_images, sample_keypoints, sample_targets, sample_valids = next(
        iter(train_dataset))

    i = random.randint(0, cfg.BATCH_SIZE)
    sample_keypoints = sample_keypoints[i:i+2].numpy().reshape(-1, 17, 2)
    sample_keypoints[:, :, 0] = sample_keypoints[:, :, 0] * cfg.IMG_WIDTH
    sample_keypoints[:, :, 1] = sample_keypoints[:, :, 1] * cfg.IMG_HEIGHT

    visualize_keypoints(np.uint8(sample_images[i:i+2]), sample_keypoints)
