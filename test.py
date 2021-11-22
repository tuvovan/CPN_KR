import os
from glob import glob

from model.model import *
from model.utils import *
import model.config as cfg
from dataprocessing.dataset import *

from tensorflow.keras import Input, Model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def define_model(input_shape):
    input = Input(shape=input_shape)
    backbone = get_backbone_resnet50(input_shape)
    global_fms, global_out = GlobalNet(backbone=backbone)(input)
    refine_out = RefineNet()(global_fms)
    model = Model(inputs=input, outputs=[global_out, refine_out])
    return model


def preprocess_images(paths):
    image = []
    original = []
    detail = []
    for path in paths:
        img_orig = plt.imread(path)
        img = cv2.resize(img_orig, (192, 256), interpolation=cv2.INTER_CUBIC)
        img = (np.array(img)-cfg.pixel_means) / 255.0
        image.append(img)
        h, w, _ = img_orig.shape
        original.append(img_orig)
        detail.append([h, w])

    return np.array(image), np.array(original), np.array(detail)


def deprocess_images(image):
    image = image * 255.
    image += cfg.pixel_means
    return np.uint8(image)


input_shape = (256, 192, 3)

model = define_model(input_shape)
# model.load_weights(cfg.save_path_val.replace("keypoint_model_2", "keypoint_model_3"))
model.load_weights(cfg.save_path_train)

# Check its architecture
model.summary()

img_paths = glob("test_images/*.jpeg")
images, original, details = preprocess_images(img_paths)
predicted = run_test_augment(model, images)
predicted = np.array(predicted).reshape(-1, 17, 2)
predicted[:, :, 0] = predicted[:, :, 0] * 4
predicted[:, :, 1] = predicted[:, :, 1] * 4
images = deprocess_images(images)
resized_keypoints = postprocess_keypoints(keypoints=predicted, details=details)
# visualize_keypoints(original, resized_keypoints)
for i, im, kp in zip(img_paths, original, resized_keypoints):
    draw_skeleton(im, kp, name='{}__rs.png'.format(i))
