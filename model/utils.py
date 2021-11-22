import tensorflow as tf
import model.config as cfg
import cv2
import numpy as np

from tqdm import tqdm

def compute_loss(predicted, labels, valids):
    global_outputs, refine_outputs = predicted[0], predicted[1]
    target_1, target_2, target_3, target_4 = labels[:,0,:,:], labels[:,1,:,:], labels[:,2,:,:], labels[:,3,:,:]
    global_label_1 = target_1 * tf.cast(tf.greater(tf.reshape(valids, (-1, 1, 1, cfg.NR_SKELETON)), 1.1), tf.float32)
    global_label_2 = target_2 * tf.cast(tf.greater(tf.reshape(valids, (-1, 1, 1, cfg.NR_SKELETON)), 1.1), tf.float32)
    global_label_3 = target_3 * tf.cast(tf.greater(tf.reshape(valids, (-1, 1, 1, cfg.NR_SKELETON)), 1.1), tf.float32)
    global_label_4 = target_4 * tf.cast(tf.greater(tf.reshape(valids, (-1, 1, 1, cfg.NR_SKELETON)), 1.1), tf.float32)

    global_loss = 0.
    global_output_1, global_output_2, global_output_3, global_output_4 = global_outputs[0], global_outputs[1], global_outputs[2], global_outputs[3]

    global_loss += tf.reduce_mean(tf.square(global_output_1 -
                                    global_label_1)) / cfg.BATCH_SIZE

    global_loss += tf.reduce_mean(tf.square(global_output_2 -
                                    global_label_2)) / cfg.BATCH_SIZE

    global_loss += tf.reduce_mean(tf.square(global_output_3 -
                                    global_label_3)) / cfg.BATCH_SIZE

    global_loss += tf.reduce_mean(tf.square(global_output_4 -
                                    global_label_4)) / cfg.BATCH_SIZE

    global_loss /= 2.

    refine_loss = tf.reduce_mean(tf.square(refine_outputs - target_4), (1,2)) * tf.cast((tf.greater(valids, 0.1)), tf.float32) / cfg.BATCH_SIZE
    refine_loss = ohkm(refine_loss, 8)

    total_loss = global_loss + refine_loss
    return total_loss


def ohkm(loss, top_k):
    ohkm_loss = 0.
    values, topk_idx = tf.nn.top_k(loss, k=top_k, sorted=False)
    values = tf.reduce_mean(values, axis=1)
    ohkm_loss = tf.reduce_mean(values, axis=0)
    return ohkm_loss

def draw_skeleton(aa, kp, show_skeleton_labels=False, name=None):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 
    'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 
    'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

    for i, j in skeleton:
        if kp[i-1][0] >= 0 and kp[i-1][1] >= 0 and kp[j-1][0] >= 0 and kp[j-1][1] >= 0:
            cv2.line(aa, tuple(kp[i-1][:2]), tuple(kp[j-1][:2]), (0,255,0), 5)
    for j in range(len(kp)):
        if kp[j][0] >= 0 and kp[j][1] >= 0:

            cv2.circle(aa, tuple(kp[j][:2]), 2, tuple((255,0,0)), 5)

            if show_skeleton_labels and kp[j][2] > 0.1:
                cv2.putText(aa, kp_names[j], tuple(kp[j][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
    aa = cv2.cvtColor(aa, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, aa)

def postprocess_keypoints(keypoints, details):
    resized_keypoints = []

    for keypoint, detail in zip(keypoints, details):
        resized_keypoint = np.zeros((17,2))
        h, w = detail
        resized_keypoint[:, 1], resized_keypoint[:, 0] = (
            h/256)*keypoint[:, 1], (w/192)*keypoint[:, 0]

        resized_keypoints.append(resized_keypoint)
    return np.array(resized_keypoints).astype(int)

def generate_heatmap(heatmap, pt, sigma):
    heatmap[int(pt[1]),int(pt[0])] = 1
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    am = np.amax(heatmap)
    heatmap /= am / 255.
    # heatmap /= am
    return heatmap

def lrfn(epoch):
    warmup_epochs = 10
    lr_max = 0.002
    lr_start = 0.0005
    total_epochs = cfg.EPOCHS
    num_cycles = 50
    if epoch < warmup_epochs:
        lr = (lr_max - lr_start) / warmup_epochs * epoch + lr_start
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = lr_max * (0.5 * (1.0 + np.cos(np.pi * ((num_cycles * progress) % 1.0))))
    return lr


def run_test_augment(model, inputs):
    print('testing...')
    full_result = []
    for i, input_var in tqdm(enumerate(inputs)):
        if cfg.flip == True:
            flip_inputs = np.zeros_like(inputs)
            for i, finp in enumerate(flip_inputs):
                finp = cv2.flip(finp, 1)
                flip_inputs[i] = finp
            flip_input_var = flip_inputs

        # compute output
        _, refine_output = model(inputs.astype(np.float32))
        score_map = refine_output
        score_map = score_map.numpy()

        if cfg.flip == True:
            _, flip_output = model(flip_input_var)
            flip_score_map = flip_output
            flip_score_map = flip_score_map.numpy()

            for i, fscore in enumerate(flip_score_map):
                # fscore = fscore.transpose((1,2,0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore)
                for (q, w) in cfg.symmetry:
                    fscore[q], fscore[w] = fscore[w], fscore[q] 
                fscore = np.array(fscore)
                score_map[i] += fscore
                score_map[i] /= 2

    for b in range(inputs.shape[0]):
        single_result_dict = {}
        single_result = []
        
        single_map = score_map[b]
        r0 = single_map.copy()
        r0 /= 255
        r0 += 0.5
        v_score = np.zeros(17)
        for p in range(17): 
            single_map[:,:,p] /= np.amax(single_map[:,:,p])
            border = 10
            dr = np.zeros((cfg.OUTPUT_SHAPE[0] + 2*border, cfg.OUTPUT_SHAPE[1]+2*border))
            dr[border:-border, border:-border] = single_map[:,:,p].copy()
            dr = cv2.GaussianBlur(dr, (9, 9), 0)
            lb = dr.argmax()
            y, x = np.unravel_index(lb, dr.shape)

            y -= border
            x -= border
            x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
            y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))

            v_score[p] = float(r0[int(round(y)+1e-10), int(round(x)+1e-10), p])                
            single_result.append(x)
            single_result.append(y)
            # single_result.append(1)   
        if len(single_result) != 0:
            single_result_dict['category_id'] = 1
            single_result_dict['keypoints'] = single_result
            full_result.append(single_result)
    return np.array(full_result)

def run_test(model, inputs):
    print('testing...')
    full_result = []

    # compute output
    _, refine_output = model(inputs.astype(np.float32))
    score_map = refine_output
    score_map = score_map.numpy()

    for b in range(inputs.shape[0]):
        single_result_dict = {}
        single_result = []
        
        single_map = score_map[b]
        r0 = single_map.copy()
        r0 /= 255
        r0 += 0.5
        v_score = np.zeros(17)
        for p in range(17): 
            single_map[:,:,p] /= np.amax(single_map[:,:,p])
            border = 10
            dr = np.zeros((cfg.OUTPUT_SHAPE[0] + 2*border, cfg.OUTPUT_SHAPE[1]+2*border))
            dr[border:-border, border:-border] = single_map[:,:,p].copy()
            dr = cv2.GaussianBlur(dr, (9, 9), 0)
            lb = dr.argmax()
            y, x = np.unravel_index(lb, dr.shape)
            dr[y, x] = 0
            lb = dr.argmax()
            py, px = np.unravel_index(lb, dr.shape)
            y -= border
            x -= border
            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.25
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
            x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
            y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))
            # resy = float((4 * y + 2) / cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
            # resx = float((4 * x + 2) / cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
            v_score[p] = float(r0[int(round(y)+1e-10), int(round(x)+1e-10), p])                
            single_result.append(x)
            single_result.append(y)
            # single_result.append(1)   
        if len(single_result) != 0:
            single_result_dict['category_id'] = 1
            single_result_dict['keypoints'] = single_result
            full_result.append(single_result)
    return np.array(full_result)


if __name__ == '__main__':
    global_outputs = [tf.random.uniform(shape=(cfg.BATCH_SIZE, 4, *cfg.OUTPUT_SHAPE, cfg.NR_SKELETON)),
                      tf.random.uniform(shape=(cfg.BATCH_SIZE, *cfg.OUTPUT_SHAPE, cfg.NR_SKELETON))]

    labels = tf.random.uniform(shape=(cfg.BATCH_SIZE, 4, *cfg.OUTPUT_SHAPE, cfg.NR_SKELETON))
    valids = tf.random.uniform(shape=(cfg.BATCH_SIZE, cfg.NR_SKELETON))

    loss = compute_loss(global_outputs, labels, valids)
