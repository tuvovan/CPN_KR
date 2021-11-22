# IMPORTS
# from ipynb.fs.full.defs import * #importing defs for download_unzip function
import pandas as pd
from pycocotools.coco import COCO
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import multiprocessing
import json
from PIL import Image


'''
FIXED PARAMETERS
'''

# DIMENSIONS OF RESIZED IMAGE (INPUT FOR THE MODEL)
IMG_WIDTH = 192
IMG_HEIGHT = 256
TRAIN_VAL = 0

# PARAMETERS OF THE KEYPOINTS OF THE DATASET
N_DIM = 3  # x,y,visibility
N_KEYPOINTS = 17  # it is set later in the code just in case
K_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
           'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']  # it is set later in the code just in case

# PATHS
TRAIN_ANNOT_PATH = '../coco/annotations/person_keypoints_train2017.json'  # annotations
VAL_ANNOT_PATH = 'coco/annotations/person_keypoints_val2017.json'  # annotations
PATH_TRAIN = 'coco/images/train2017'  # images
PATH_VAL = 'coco/images/val2017'  # images


def get_img(torval, img_id):
    # load image
    # the image names are 12 digits and the first gaps are filled with 0's
    img_name = '000000000000'
    img_name = img_name[0:len(img_name)-len(img_id)] + img_id + '.jpg'
    path = PATH_VAL if torval else PATH_TRAIN
    og_img = Image.open(path + '/' + img_name)
    return og_img


def write_img(torval, i, crop_image, img_id):
    # load image
    # the image names are 12 digits and the first gaps are filled with 0's
    img_name = '000000000000'
    img_name = img_name[0:len(img_name)-len(img_id)] + img_id + '.jpg'
    img_name = img_name.replace('.jpg', '_{}.jpg'.format(i))
    path = PATH_VAL if torval else PATH_TRAIN
    path = val_crop_path if torval else train_crop_path
    crop_image.save(path + '/' + img_name)

# some of the samples of the dataset have keypoints drawn out of the bounding box and I propose the following solution
# making the bounding box as big as necessary, if it is possible, to include the keypoints out of the area, as this part of the image can be interesting for the training or validation of the model


def check_keypoints_in_bbox(bbox, keypoints, k_vis, og_img):
    # attributes of the bounding box and original image
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    img_w, img_h = og_img.size

    # calculate min and max values of x and y positions of the keypoints
    x_min = 99999
    x_max = -1
    y_min = 99999
    y_max = -1
    for i in range(len(keypoints)):
        if k_vis[i] > 0:
            x_temp = keypoints[i][0]
            y_temp = keypoints[i][1]
            if x_temp < x_min:
                x_min = x_temp
            if x_temp > x_max:
                x_max = x_temp
            if y_temp < y_min:
                y_min = y_temp
            if y_temp > y_max:
                y_max = y_temp
    x_min = x_min-10
    x_max = x_max+10
    y_min = y_min-10
    y_max = y_max+10

    if x_min < bbox_x:
        if x_min < 0:
            x_min = 0
    else:
        x_min = bbox_x

    if x_max > bbox_x+bbox_w:
        if x_max > img_w:
            x_max = img_w
    else:
        x_max = bbox_x+bbox_w

    if y_min < bbox_y:
        if y_min < 0:
            y_min = 0
    else:
        y_min = bbox_y

    if y_max > bbox_y+bbox_h:
        if y_max > img_h:
            y_max = img_h
    else:
        y_max = bbox_y+bbox_h

    bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
    return bbox


def get_keypoints_vis(k_list):
    k_array = np.asarray(k_list)
    k_array3d = np.reshape(k_array, (17, 3))
    keypoints = k_array3d[:, :2]
    k_vis = k_array3d[:, 2]
    return keypoints, k_vis


# CROP IMAGE IN THE BOUNDING BOX AND RESIZE TO INPUT IMAGE SIZE

# torval -> 0 if training and 1 if validation dataset
def crop_resize_img(torval, og_img, bbox):
    # attributes of the bounding box
    bbox_x, bbox_y, bbox_w, bbox_h = bbox

    # resize image part of the bounding box
    res_img = og_img.resize((IMG_WIDTH, IMG_HEIGHT), box=(
        bbox_x, bbox_y, bbox_x+bbox_w, bbox_y+bbox_h))

    return res_img


# RESCALE KEYPOINTS TO MATCH RESIZED IMAGE
# crop -> rest to the keypoint position the bounding box position as the keypoint should start from there
# rescale -> as the image has been resized, the position of the keypoint must be adapted taking into account the scale from the original image to the resized one
def rescale_keypoints(keypoints, bbox):
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    k_array = np.asarray(keypoints)
    k_array3d = np.reshape(k_array, (N_KEYPOINTS, N_DIM))
    keypoints = k_array3d[:, :2]
    box_start_pos = np.asarray([bbox_x, bbox_y])
    box_size = np.asarray([bbox_w, bbox_h])
    res_size = np.asarray([IMG_WIDTH, IMG_HEIGHT])
    keypoints = np.round((keypoints-box_start_pos) *
                         (res_size/box_size)).astype(int)

    # if the original value was 0, then it will be converted to negative, so it should be reconverted to 0
    keypoints[keypoints < 0] = 0

    return keypoints


def get_meta(coco):
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']

        yield [img_id, img_file_name, w, h, anns]


def convert_to_df(coco):
    images_data = []
    persons_data = []
    c = 0
    for img_id, img_fname, w, h, meta in tqdm(get_meta(coco)):
        images_data.append({
            'image_id': int(img_id),
            'path': img_fname,
            'width': int(w),
            'height': int(h)
        })
        for i, m in enumerate(meta):
            keypoints, k_vis = get_keypoints_vis(m["keypoints"])
            if m['iscrowd'] == 1 or m["num_keypoints"] < 1:
                continue
            og_img = get_img(TRAIN_VAL, str(img_id))
            bbox = check_keypoints_in_bbox(
                m['bbox'], keypoints, k_vis, og_img)
            res_img = crop_resize_img(TRAIN_VAL, og_img, bbox)
            write_img(TRAIN_VAL, i, res_img, str(img_id))
            keypoints = rescale_keypoints(m['keypoints'], m['bbox'])
            keypoints = list(np.squeeze(np.array(keypoints).reshape(34, -1)))
            persons_data.append({
                'path': img_fname.replace('.jpg', '_{}.jpg'.format(i)),
                # 'is_crowd': m['iscrowd'],
                # 'bbox': bbox,
                # 'area': m['area'],
                # 'num_keypoints': m['num_keypoints'],
                'keypoints': keypoints,
                'vis': k_vis
            })

    images_df = pd.DataFrame(images_data)
    images_df.set_index('image_id', inplace=True)

    persons_df = pd.DataFrame(persons_data)
    persons_df.set_index('path', inplace=True)

    return images_df, persons_df


train_annot_path = 'coco/annotations/person_keypoints_train2017.json'
train_img_path = 'coco/images/train2017/'
train_crop_path = 'coco/images/train2017_crop/'

val_annot_path = 'coco/annotations/person_keypoints_val2017.json'
val_img_path = 'coco/images/val2017/'
val_crop_path = 'coco/images/val2017_crop/'

TRAIN_VAL = 0
train_coco = COCO(train_annot_path)
images_df, persons_df = convert_to_df(train_coco)
persons_df.to_csv(train_annot_path.replace('json', 'csv'))

TRAIN_VAL = 1
val_coco = COCO(val_annot_path)
images_df, persons_df = convert_to_df(val_coco)
persons_df.to_csv(val_annot_path.replace('json', 'csv'))

