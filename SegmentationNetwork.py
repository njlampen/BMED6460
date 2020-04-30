# ####################################################################################################################
#
# Code taken from the submissions of the MU-Lux-CZ and UVA-NL teams at the cell tracking challenge:
# http://celltrackingchallenge.net/participants/MU-Lux-CZ/
# http://celltrackingchallenge.net/participants/UVA-NL/
# 
# Notes from UVA-NL:    
#   The input and output were adjusted so that the initial segmentation is unaffected by the additional tracker.
#   Also a few if statement and variable names were changed to adjust to the new input feeding.
#
#
# ####################################################################################################################

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from skimage import io
import skimage
import tensorflow as tf
from scipy.ndimage.morphology import binary_fill_holes
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 5GB of memory on the GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

BATCH_SIZE = 2
MARKER_THRESHOLD = 240

# [marker_threshold, cell_mask_thershold]
THRESHOLDS = {
    'DIC-C2DH-HeLa': [240, 216],
    'Fluo-N2DH-SIM+': [240, 229],
    'PhC-C2DL-PSC': [240, 156],
    'BF-C2DL-HSC' : [240, 100]}


def median_normalization(image):
    image_ = image / 255 + (.5 - np.median(image / 255))
    return np.maximum(np.minimum(image_, 1.), .0)


def hist_equalization(image):
    return cv2.equalizeHist(image) / 255


def get_normal_fce(normalization):
    if normalization == 'HE':
        return hist_equalization
    if normalization == 'MEDIAN':
        return median_normalization
    else:
        assert False


def remove_uneven_illumination(img, blur_kernel_size=501):
    """
    uses LPF to remove uneven illumination
    """

    img_f = img.astype(np.float32)
    img_mean = np.mean(img_f)

    img_blur = cv2.GaussianBlur(img_f, (blur_kernel_size, blur_kernel_size), 0)
    result = np.maximum(np.minimum((img_f - img_blur) + img_mean, 255), 0).astype(np.int32)

    return result


def remove_edge_cells(label_img, border=20):
    edge_indexes = get_edge_indexes(label_img, border=border)
    return remove_indexed_cells(label_img, edge_indexes)


def get_edge_indexes(label_img, border=20):
    mask = np.ones(label_img.shape)
    mi, ni = mask.shape
    mask[border:mi - border, border:ni - border] = 0
    border_cells = mask * label_img
    indexes = (np.unique(border_cells))

    result = []

    # get only cells with center inside the mask
    for index in indexes:
        cell_size = sum(sum(label_img == index))
        gap_size = sum(sum(border_cells == index))
        if cell_size * 0.5 < gap_size:
            result.append(index)

    return result


def remove_indexed_cells(label_img, indexes):
    mask = np.ones(label_img.shape)
    for i in indexes:
        mask -= (label_img == i)
    return label_img * mask


def get_image_size(path):
    """
    returns size of the given image
    """
    names = os.listdir(path)
    name = names[0]
    o = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
    return o.shape[0:2]


def get_new_value(mi, divisor=16):
    if mi % divisor == 0:
        return mi
    else:
        return mi + (divisor - mi % divisor)


def read_image(path):
    if 'Fluo' in path:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if 'Fluo-N2DL-HeLa' in path:
            img = (img / 255).astype(np.uint8)
        if 'Fluo-N2DH-SIM+' in path:
            img = np.minimum(img, 255).astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=1)
    union = K.sum(y_true,axis=1)+K.sum(y_pred,axis=1)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=1)
    union = K.sum(y_true, axis=1) + K.sum(y_pred, axis=1)
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


# read images
def load_images(path, cut=False, new_mi=0, new_ni=0, normalization='HE', uneven_illumination=False):
    names = os.listdir(path)
    names.sort()

    mi, ni = get_image_size(path)

    dm = (mi % 16) // 2
    mi16 = mi - mi % 16
    dn = (ni % 16) // 2
    ni16 = ni - ni % 16

    total = len(names)
    normalization_fce = get_normal_fce(normalization)

    image = np.empty((total, mi, ni, 1), dtype=np.float32)

    for i, name in enumerate(names):

        o = read_image(os.path.join(path, name))

        if o is None:
            print('image {} was not loaded'.format(name))

        if uneven_illumination:
            o = np.minimum(o, 255).astype(np.uint8)
            o = remove_uneven_illumination(o)

        image_ = normalization_fce(o)

        image_ = image_.reshape((1, mi, ni, 1)) - .5
        image[i, :, :, :] = image_

    if cut:
        image = image[:, dm:mi16 + dm, dn:ni16 + dn, :]
    if new_ni > 0 and new_ni > 0:
        image2 = np.zeros((total, new_mi, new_ni, 1), dtype=np.float32)
        image2[:, :mi, :ni, :] = image
        image = image2

    print('loaded images from directory {} to shape {}'.format(path, image.shape))
    return image


def create_model(model_path, mi=512, ni=512, loss='mse'):
    input_img = Input(shape=(mi, ni, 1))

    # network definition
    c1e = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1e)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2e = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2e)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3e = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3e)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    c4e = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4e)
    p4 = MaxPooling2D((2, 2), padding='same')(c4)

    c5e = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5e)

    u4 = UpSampling2D((2, 2), interpolation='bilinear')(c5)
    a4 = Concatenate(axis=3)([u4, c4])
    c6e = Conv2D(256, (3, 3), activation='relu', padding='same')(a4)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6e)

    u3 = UpSampling2D((2, 2), interpolation='bilinear')(c6)
    a3 = Concatenate(axis=3)([u3, c3])
    c7e = Conv2D(128, (3, 3), activation='relu', padding='same')(a3)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7e)

    u2 = UpSampling2D((2, 2), interpolation='bilinear')(c7)
    a2 = Concatenate(axis=3)([u2, c2])
    c8e = Conv2D(64, (3, 3), activation='relu', padding='same')(a2)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8e)

    u1 = UpSampling2D((2, 2), interpolation='bilinear')(c8)
    a1 = Concatenate(axis=3)([u1, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(a1)

    c10 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    markers = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    cell_mask = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    output = Concatenate(axis=3)([markers, cell_mask])

    model = Model(input_img, output)
    model.compile(optimizer=Adam(lr=0.0001), loss=loss)
    
    Early_Stop = EarlyStopping(monitor='val_loss', min_delta=-.00001, patience=50, verbose=0, mode='auto')
    #checkpoint = MultiGPUCheckpointCallback(modelfilepath,model,monitor='val_loss',verbose=1,save_weights_only=False,save_best_only=True,mode='auto')
    print(model.summary())

    print('Model was created')

    model.load_weights(model_path) # by_name=True
    #model.load_weights(model_path)

    return model


# postprocess markers
def postprocess_markers(m, threshold=240, erosion_size=12, circular=False, step=4):
    # threshold
    m = m.astype(np.uint8)
    _, new_m = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)

    # distance transform | only for circular objects
    if circular:
        dist_m = (cv2.distanceTransform(new_m, cv2.DIST_L2, 5) * 5).astype(np.uint8)
        new_m = hmax(dist_m, step=step).astype(np.uint8)

    # filling gaps
    hol = binary_fill_holes(new_m).astype(np.uint8)

    # morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    clos = cv2.morphologyEx(hol, cv2.MORPH_OPEN, kernel)

    # label connected components
    idx, res = cv2.connectedComponents(clos)

    return idx, res


# postprocess markers
def postprocess_markers2(img, threshold=240, erosion_size=12, circular=False, step=4):
    # distance transform | only for circular objects
    if circular:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        markers = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        new_m = ((hconvex(markers, step) > 0) & (img > threshold)).astype(np.uint8)
    else:

        # threshold
        m = img.astype(np.uint8)
        _, new_m = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)

        # filling gaps
        hol = binary_fill_holes(new_m * 255).astype(np.uint8)

        # morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        new_m = cv2.morphologyEx(hol, cv2.MORPH_OPEN, kernel)

    # label connected components
    idx, res = cv2.connectedComponents(new_m)

    return idx, res


def hmax2(img, h=50):
    h_img = img.astype(np.uint16) + h

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rec0 = img
    rec1 = np.minimum(cv2.dilate(rec0, kernel), h_img)

    # reconstruction
    for i in range(255):

        rec1 = np.minimum(cv2.dilate(rec0, kernel), h_img)
        if np.sum(rec0 - rec1) == 0:
            break
        rec0 = rec1

    # retype to uint8
    hmax_result = np.maximum(np.minimum((rec1 - h), 255), 0).astype(np.uint8)

    return hmax_result


def hconvex(img, h=5):
    return img - hmax2(img, h)


def hmax(ml, step=50):
    """
    old version of H-MAX transform
    not really correct
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ml = cv2.blur(ml, (3, 3))

    rec1 = np.maximum(ml.astype(np.int32) - step, 0).astype(np.uint8)

    for i in range(255):
        rec0 = rec1
        rec1 = np.minimum(cv2.dilate(rec0, kernel), ml.astype(np.uint8))
        if np.sum(rec0 - rec1) == 0:
            break

    return ml - rec1 > 0


# postprocess cell mask
def postprocess_cell_mask(b, threshold=230):
    # tresholding
    b = b.astype(np.uint8)
    bt = cv2.inRange(b, threshold, 255)

    return bt


def threshold_and_store(predictions,
                        input_images,
                        res_path,
                        thr_markers=240,
                        thr_cell_mask=230,
                        viz=True,
                        circular=False,
                        erosion_size=12,
                        step=4,
                        border=0):

    print(predictions.shape)
    print(input_images.shape)
    viz_path = res_path.replace('_RES', '_VIZ')

    for i in range(predictions.shape[0]):

        m = predictions[i, :, :, 1] * 255
        c = predictions[i, :, :, 3] * 255
        o = (input_images[i, :, :, 0] + .5) * 255
        o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)     

        # postprocess the result of prediction
        idx, markers = postprocess_markers2(m, threshold=thr_markers, erosion_size=erosion_size, circular=circular,
                                            step=step)
        cell_mask = postprocess_cell_mask(c, threshold=thr_cell_mask)

        # correct border
        cell_mask = np.maximum(cell_mask, markers)
        
        labels = watershed(-c, markers, mask=cell_mask)
        labels = remove_edge_cells(labels, border)

        # labels = markers

        labels_rgb = cv2.applyColorMap(labels.astype(np.uint8) * 15, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(o_rgb.astype(np.uint8), 0.5, labels_rgb, 0.5, 0)
        
        # store result
        cv2.imwrite('{}/mask{:03d}.tif'.format(res_path, i), labels.astype(np.uint16))

        viz_m = np.absolute(m - (markers > 0) * 64)
        
        #if viz:
        m_rgb = cv2.cvtColor(viz_m.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        c_rgb = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        result = np.concatenate((m_rgb, c_rgb, overlay), 1)
        cv2.imwrite('{}/res{:03d}.tif'.format(viz_path, i), result)
        cv2.imwrite('{}/markers{:03d}.tif'.format(viz_path, i), markers.astype(np.uint8) * 16)
        
def predict_dataset(name, sequence, model_path, output_path, viz=False, init=False):
    """
    reads images from the path and converts them to the np array
    """

    erosion_size = 1
    if 'DIC-C2DH-HeLa' in name:
        erosion_size = 8
        NORMALIZATION = 'HE'
        MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['DIC-C2DH-HeLa']
        UNEVEN_ILLUMINATION = False
        CIRCULAR = False
        STEP = 0
        BORDER = 15

    elif 'Fluo-N2DH-SIM+' in name:
        erosion_size = 15
        NORMALIZATION = 'HE'
        MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['Fluo-N2DH-SIM+']
        UNEVEN_ILLUMINATION = False
        CIRCULAR = True
        STEP = 5
        BORDER = 1

    elif 'PhC-C2DL-PSC' in name:
        erosion_size = 1
        NORMALIZATION = 'MEDIAN'
        MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['PhC-C2DL-PSC']
        UNEVEN_ILLUMINATION = True
        CIRCULAR = True
        STEP = 3
        BORDER = 12
    elif 'BF-C2DL-HSC' in name:
        erosion_size = 1
        NORMALIZATION = 'HE'
        MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['BF-C2DL-HSC']
        UNEVEN_ILLUMINATION = True
        CIRCULAR = True
        STEP = 3
        BORDER = 15
    else:
        print('unknown dataset')
        return

    model_init_path = model_path
    store_path = output_path

    # get c_mask_threshold value
    if init:
        assert False

    if not os.path.isdir(store_path):
        os.mkdir(store_path)
        print('directory {} was created'.format(store_path))

    img_path = os.path.join('..', name, sequence)
    if not os.path.isdir(img_path):
        print('given name of dataset or the sequence is not valid')
        exit()

    mi, ni = get_image_size(img_path)
    new_mi = get_new_value(mi)
    new_ni = get_new_value(ni)

    print(mi, ni)
    print(new_mi, new_ni)

    model = create_model(model_init_path, new_mi, new_ni)

    input_img = load_images(img_path,
                            new_mi=new_mi,
                            new_ni=new_ni,
                            normalization=NORMALIZATION,
                            uneven_illumination=UNEVEN_ILLUMINATION)

    pred_img = model.predict(input_img, batch_size=BATCH_SIZE)
    print('pred shape: {}'.format(pred_img.shape))

    org_img = load_images(img_path)
    pred_img = pred_img[:, :mi, :ni, :]

    threshold_and_store(pred_img,
                        org_img,
                        store_path,
                        thr_markers=MARKER_THRESHOLD,
                        thr_cell_mask=C_MASK_THRESHOLD,
                        viz=viz,
                        circular=CIRCULAR,
                        erosion_size=erosion_size,
                        step=STEP,
                        border=BORDER)

def predict_dataset_2(path, output_path, threshold=0.15):

    # check if path exists
    if not os.path.isdir(path):
        print('input path is not a valid path')
        return

    names = os.listdir(path)
    names = [name for name in names if '.tif' in name and 'mask' in name]
    names.sort()

    img = cv2.imread(os.path.join(path, names[0]), cv2.IMREAD_ANYDEPTH)
    mi, ni = img.shape
    print('Relabelling the segmentation masks.')
    records = {}

    old = np.zeros((mi, ni))
    index = 1

    for i, name in enumerate(names):
        result = np.zeros((mi, ni), np.uint16)

        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_ANYDEPTH)
        labels = np.unique(img)[1:]
        parent_cells = []

        for label in labels:
            mask = (img == label) * 1

            mask_size = np.sum(mask)
            overlap = mask * old
            candidates = np.unique(overlap)[1:]

            max_score = 0
            max_candidate = 0

            for candidate in candidates:
                score = np.sum(overlap == candidate * 1) / mask_size
                if score > max_score:
                    max_score = score
                    max_candidate = candidate

            if max_score < threshold:
                # no parent cell detected, create new track

                records[index] = [i, i, 0]
                result = result + mask * index
                index += 1
            else:

                if max_candidate not in parent_cells:
                    # prolonging track
                    records[max_candidate][1] = i
                    result = result + mask * max_candidate

                else:
                    # split operations
                    # if have not been done yet, modify original record
                    if records[max_candidate][1] == i:
                        records[max_candidate][1] = i - 1
                        # find mask with max_candidate label in the result and rewrite it to index
                        m_mask = (result == max_candidate) * 1
                        result = result - m_mask * max_candidate + m_mask * index
                        
                        records[index] = [i, i, max_candidate.astype(np.uint16)]
                        index += 1

                    # create new record with parent cell max_candidate
                    records[index] = [i, i, max_candidate.astype(np.uint16)]
                    result = result + mask * index
                    index += 1

                # update of used parent cells
                parent_cells.append(max_candidate)
        # store result
        cv2.imwrite(os.path.join(output_path, name), result.astype(np.uint16))
        old = result

    # store tracking
    print('Generating the tracking file.')
    with open(os.path.join(output_path, 'res_track.txt'), "w") as file:
        for key in records.keys():
            file.write('{} {} {} {}\n'.format(key, records[key][0], records[key][1], records[key][2]))