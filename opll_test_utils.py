
import numpy as np                     # numeric python lib

import tensorlayer as tl
import shutil
import SimpleITK as sitk
# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import nibabel
from tensorflow.python.ops import array_ops
from skimage.feature import greycomatrix, greycoprops
from sklearn.feature_extraction import image
from skimage.exposure import rescale_intensity
import os
from itertools import product
from scipy.signal import  argrelextrema
import cv2
# from imgaug import augmenters as iaa
import random

# Feature extraction

def rgb2gray(rgb):
    '''
    Change rgb to grayscale image
    '''
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_ND_bounding_box(input, margin):
    '''
    get bounding box excluding background
    :return: min, max indexes, and width, length, height of bounding box
    '''
    input_shape = input.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indices = np.nonzero(input)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indices[i].min())
        idx_max.append(indices[i].max())
    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    width = idx_max[0] - idx_min[0]
    length = idx_max[1] - idx_min[1]
    height = idx_max[2] - idx_min[2]
    return idx_min, idx_max, width, length, height

def crop_volume_with_bounding_box(volume, min_idx, max_idx):
    output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    return output

def get_area(label):
    indices = np.nonzero(label)
    area = len(indices[0])
    return area

def get_volume(label):
    indices = np.nonzero(label)
    vol = len(indices[0])
    return vol

def get_glcm(input):
    '''
    get gray level co-occurrence matrix and then it's features
    '''
    glcm = greycomatrix(input, [1], [0], 256, normed=True)
    cont = greycoprops(glcm, 'contrast')
    diss = greycoprops(glcm, 'dissimilarity')
    homo = greycoprops(glcm, 'homogeneity')
    eng = greycoprops(glcm, 'energy')
    corr = greycoprops(glcm, 'correlation')
    ASM = greycoprops(glcm, 'ASM')
    return [cont, diss, homo, eng, corr, ASM]

def save_array_as_nifty_volume(data, filename, reference_name = None):
    '''
    :param data: np array
    :param filename: path + filename want to save
    '''
    if data.ndim == 3:
        transposed_data = np.transpose(data, [2,1,0])
    if data.ndim == 2:
        transposed_data = np.transpose(data, [1,0])
    img = sitk.GetImageFromArray(transposed_data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

def save_array_as_nifty_volume2(data, filename):
    '''
    :param data: np array
    :param filename: path + filename want to save
    '''
    img = nibabel.Nifti1Image(data, affine=np.eye(4))
    nibabel.save(img, filename)

def get_array_as_nifty_volume(data):
    nii = nibabel.Nifti1Image(data, affine=np.eye(4))
    return nii

def add_values_to_key(target_list):
    '''
    marge several dictionaries with same key in list to one dictionaries
    :return:
    '''
    subdict = {}
    for i in range(len(target_list)):
        for k,v in target_list[i].items():
            subdict.setdefault(k, [])
            subdict[k].append(v)
    for k,v in subdict.items():
        subdict[k] = tuple(subdict[k])
    return subdict

def drop_col_contains_sth(dataframe,sth):
    '''
    drop columns contains certain strings(sth) in the dataframe
    :param sth:  str 'sth'
    :return:
    '''
    dropped_df = dataframe[dataframe.columns.drop(list(dataframe.filter(regex=sth)))]
    return dropped_df

def get_path_list(data_path):
    '''
    :return: all folder(or file list)
    '''
    id_list = []
    for path in data_path:
        path_list = tl.files.load_folder_list(path)
        id_list += [os.path.join(path, os.path.basename(p), os.path.basename(p)) for p in path_list]
    return id_list

# Loading

def make_folder(data_path):
    '''
    make folders for each file in the data path.
    Directory names are same as file name (w/o ext)
    '''
    path_list = os.listdir(data_path)
    for i in range(len(path_list)):
        filename = os.path.splitext(path_list[i])
        abs_path = os.path.join(data_path,filename[0])
        tl.files.exists_or_mkdir(abs_path)

def make_folder_from_root(data_path):
    '''
    From mother file (data_path), make folders for each file in the sub directories.
    Directory names are same as file name (w/o ext)
    '''
    path_list = tl.files.load_folder_list(data_path)
    for i in range(len(path_list)):
        make_folder(path_list[i])

def move_file(src,dst):
    shutil.move(src, dst)

def copy_file(src,dst):
    # shutil.copy2('/src/file.ext', '/dst/dir')  # target filename is /dst/dir/file.ext
    shutil.copy2(src,dst)

def copy_files(data_path):
    dir_list = tl.files.load_folder_list(data_path)
    print(dir_list)
    for i in range(len(dir_list)):
        path_list = os.listdir(dir_list[i])
        for idx in range(len(path_list) // 2):
            filename = path_list[1::2]
            print(filename[idx])
            dir_name = os.path.splitext(filename[idx])
            print(dir_name)
            copy_file(dir_list[i] + '\\' + filename[idx], dir_list[i]  + '\\' + dir_name[0])

def move_files(data_path):
    dir_list = tl.files.load_folder_list(data_path)
    print(dir_list)
    for i in range(len(dir_list)):
        path_list = os.listdir(dir_list[i])
        for idx in range(len(path_list) // 2):
            filename = path_list[1::2]
            print(filename[idx])
            dir_name = os.path.splitext(filename[idx])
            print(dir_name)
            move_file(dir_list[i] + '\\' + filename[idx], dir_list[i]  + '\\' + dir_name[0])

def make_dir_and_move(data_path):
    '''
    make directories and move files to dir with same names
    '''
    dir_list = tl.files.load_folder_list(data_path)
    print(dir_list)
    for i in range(len(dir_list)):
        make_folder(dir_list[i])
        path_list = os.listdir(dir_list[i])
        for idx in range(len(path_list) // 2):
            filename = path_list[1::2]
            print(filename[idx])
            dir_name = os.path.splitext(filename[idx])
            print(dir_name)
            move_file(dir_list[i] + '\\' + filename[idx], dir_list[i]  + '\\' + dir_name[0])

def cal_hm_landmark(arr, max_percent = 99.8, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
    if arr.ndim > 1:
        arr = arr.ravel()
    arr_hist_sd, arr_edges_sd = np.histogram(arr, bins = range(np.max(arr)+2))
    arr_mean = int(np.mean(arr))
    black_peak = np.argmax(arr_hist_sd[:arr_mean])
    white_peak = arr_mean + np.argmax(arr_hist_sd[arr_mean:])
    # valley = arr_hist_sd[int(black_peak):int(white_peak)]
    # consider only points over 10
    # over_cutline = np.where(valley > 10)
    # find local minmums among 500 points
    # local_mins = argrelextrema(valley[over_cutline], np.less, order=250)
    # take first local minimum
    # local_min = over_cutline[0][local_mins[0][0]]
    # local_min = argrelextrema(valley, np.less, order=250)[0][np.where(arr_hist_sd[argrelextrema(valley, np.less, order=250)] > 10)[0][0]]

    threshold_dict = {}     # 'fuzzy', 'mean', 'median', 'valley'
    threshold_dict['fuzzy'] = int((black_peak + white_peak) / 2)
    threshold_dict['mean'] = arr_mean
    threshold_dict['median'] = int(np.median(arr))
    # threshold_dict['valley'] = black_peak + local_min

    pc1 = threshold_dict[threshold]
    pc2 = np.percentile(arr, max_percent)
    ioi = arr[np.where((arr>=pc1) * (arr<=pc2))]
    landmark_list = [np.percentile(ioi, i * (100/n_divide) ) for i in range(n_divide) if not i == 0]
    landmark_list = [pc1] + landmark_list + [pc2]

    if standard:
        std_scale = (scale / pc2)
        landmark_list = [landmark * std_scale for landmark in landmark_list]

    return [int(landmark) for landmark in landmark_list]

def get_mean_lm(input, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
    landmarks_list = []
    for i in input:
        print('i.shape : ', i.shape)
        landmark_list = cal_hm_landmark(i,threshold=threshold, n_divide = n_divide, standard=standard, scale=scale)
        landmarks_list.append(landmark_list)
    return landmarks_list, np.mean(landmarks_list, axis=0).astype(int)

def hm_rescale(arr, input_landmark_list, standard_landmark_list):
    arr_shape = arr.shape
    if arr.ndim > 1:
        arr = arr.ravel()
    arr_copy = np.zeros_like(arr)

    scale_idx = np.where((arr < input_landmark_list[0]))

    # 0 ~ pc1 rescale
    arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
                                            in_range=(input_landmark_list[0] - 1, input_landmark_list[0]),
                                            out_range=(standard_landmark_list[0] - 1, standard_landmark_list[0]))
    # pc1 ~ m25 ~ m50 ~ m75 ~ pc2 rescale
    for idx in range(len(input_landmark_list) - 1):

        scale_idx = np.where((arr >= input_landmark_list[idx]) * (arr < input_landmark_list[idx+1]))
        arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
                                                in_range=(input_landmark_list[idx], input_landmark_list[idx+1]),
                                                out_range=(standard_landmark_list[idx], standard_landmark_list[idx+1]))
    # pc2 ~ max rescale
    scale_idx = np.where((arr >= input_landmark_list[-1]))
    arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
                                                in_range=(input_landmark_list[-1], input_landmark_list[-1] + 1),
                                                out_range=(standard_landmark_list[-1], standard_landmark_list[-1] + 1))



    arr_copy = np.clip(arr_copy, a_min=standard_landmark_list[0], a_max=standard_landmark_list[-1])

    return arr_copy.reshape(arr_shape)

def batch_histogram_match(input, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
    print('type(input) : ', type(input))
    output = input
    print('input.shape : ', input.shape)
    print('>>> before g m l')
    landmarks_list, stadard_list = get_mean_lm(input,threshold=threshold, n_divide = n_divide, standard=standard, scale=scale)
    print('>>> passed get_mean_lm')
    for idx in range(len(input)):
        batch_match = hm_rescale(input[idx], landmarks_list[idx],stadard_list)
        output[idx] = batch_match
    output = tf.convert_to_tensor(output)
    return output

def std_list_loop(threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
    def _cond(i,input):
        return tf.less(i,input.shape[0])
    def _body(i,input):
        landmark_list = cal_hm_landmark(input[i],threshold=threshold, n_divide = n_divide, standard=standard, scale=scale)
    pass



#############################################################################################################################
#                                                    Layer Functions                                                        #
#############################################################################################################################

initializer = tf.contrib.layers.variance_scaling_initializer()
regularizer = None #tf.contrib.layers.l2_regularizer(0.0001)

def conv2D(name, inputs, filters, kernel_size, strides, padding='valid', dilation_rate=(1,1)):
    conv2D = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                              padding=padding, use_bias=True, dilation_rate=dilation_rate,
                              kernel_initializer=initializer, kernel_regularizer=regularizer, name=name)
    return conv2D

def s_conv2D(name, inputs, filters, kernel_size, strides, padding='valid'):
    s_conv2D = tf.layers.separable_conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=True,
                                          depthwise_initializer=initializer, depthwise_regularizer=regularizer, pointwise_initializer=initializer,
                                          pointwise_regularizer=regularizer, name=name)
    return s_conv2D

def deconv2D(name, inputs, filter_shape, output_shape, strides, padding='valid'):
    W = tf.get_variable(name+'W', filter_shape, initializer=initializer,regularizer=regularizer)
    shape = tf.shape(inputs)
    batch_size = shape[0]
    output_shape2 = [batch_size, output_shape[1], output_shape[2], output_shape[3]]
    layer = tf.nn.conv2d_transpose(inputs, filter=W, output_shape=output_shape2, strides=strides, padding=padding)
    return layer

def re_conv2D(name, inputs, output_shape):
    """
    https://distill.pub/2016/deconv-checkerboard/
    """
    resize_layer = tf.image.resize_nearest_neighbor(images=inputs, size=[output_shape[1], output_shape[2]], name=name+'_resizing')
    # padding_layer = tf.pad(resize_layer)
    # conv_layer = conv2D(padding_layer)
    conv_layer = conv2D(name=name+'_conv', inputs=resize_layer, filters=output_shape[3], kernel_size=[3, 3], strides=[1, 1], padding='same')
    return conv_layer

def GlobalAveragePooling2D(input, n_class, name, keep_dims=False):
    """
    replace Fully Connected Layer.
    https://www.facebook.com/groups/smartbean/permalink/1708560322490187/
    https://github.com/AndersonJo/global-average-pooling/blob/master/global-average-pooling.ipynb
    """
    kernel_size_h = input.get_shape().as_list()[1]
    kernel_size_w = input.get_shape().as_list()[2]
    gap_filter = tf.get_variable(name='gap_filter', shape=[1, 1, input.get_shape()[-1], n_class], dtype=tf.float32, initializer=initializer)
    layer = tf.nn.conv2d(input, filter=gap_filter, strides=[1, 1, 1, 1], padding='SAME', name=name)
    print('kernel_size shape : ', kernel_size_h)
    print('kernel_size shape : ', kernel_size_w)
    print('layer shape : ', layer)
    layer = tf.nn.avg_pool(layer, ksize=[1, kernel_size_h, kernel_size_w, 1], strides=[1, 1, 1, 1], padding='VALID')
    if not keep_dims:
        layer = tf.reduce_mean(layer, axis=[1, 2])
    return layer

def maxpool(name,inputs, pool_size, strides, padding='valid'):
    MP_layer = tf.layers.max_pooling2d(inputs, pool_size, strides, padding,name=name)
    return MP_layer

def averagepool(name,inputs, pool_size, strides, padding='valid'):
    AP_layer = tf.layers.average_pooling2d(inputs, pool_size, strides, padding,name=name)
    return AP_layer

def maxout(name,inputs, num_units):
    # num_units must multiple of axis
    MO_layer = tf.contrib.layers.maxout(inputs, num_units,name=name)
    return MO_layer

def concat(name,inputs,axis):
    con_layer = tf.concat(inputs,axis,name=name)
    return con_layer

def dropout(name,inputs,drop_rate,training):
    DP_layer = tf.layers.dropout(inputs,drop_rate,training=training,name=name)
    return DP_layer

def add(*inputs,name):
    layer = tf.add(*inputs,name=name)
    return layer

def flatten(name,inputs):
    L1 = tf.layers.flatten(inputs,name=name)
    return L1

def fully_connected(name,inputs,units):
    L2 = tf.layers.dense(inputs,units,name=name,kernel_initializer=initializer,kernel_regularizer=regularizer)
    return L2

#############################################################################################################################
#                                                 Activation Functions                                                      #
#############################################################################################################################

def activation(name, inputs, type):
    if type == 'relu':
        active_layer = tf.nn.relu(inputs,name=name)
        return active_layer

    elif type == 'lrelu':
        alpha=0.2
        x = tf.maximum(inputs, alpha * inputs, name=name)
        return x

    elif type == 'elu':
        active_layer = tf.nn.elu(inputs,name=name)
        return active_layer

    elif type == 'prelu':
        alphas = tf.get_variable(name, inputs.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs - abs(inputs)) * 0.5
        return pos + neg

    elif type == 'selu':
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(inputs > 0.0, inputs, alpha * tf.exp(inputs) - alpha)

    elif type is None:
        return inputs

def Normalization(x, norm_type, is_train, name, G=2, esp=1e-5, channel_mode='NHWC', batch_size = 10, threshold = 'fuzzy',n_divide = 4,standard=False, scale=1,first=True):
    with tf.variable_scope('{}_norm'.format(norm_type)):

        if norm_type == 'None':
            output = x
            return output

        # Batch Normalization
        elif norm_type == 'batch':
            # output = tf.contrib.layers.batch_norm(x, center=True, scale=True, decay=0.999, is_training=is_train, updates_collections=None)
            output = tf.layers.batch_normalization(x, momentum=0.9, epsilon=0.0001, training=is_train, name='BN_'+name)
            return output

        # Group Normalization
        elif norm_type == 'group':
            with tf.name_scope('GN_'+name):
                if channel_mode == 'NHWC':

                    # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
                    x = tf.transpose(x, [0, 3, 1, 2])
                    N, C, H, W = x.get_shape().as_list()
                    G = min(G, C)
                    # N -> -1
                    x = tf.reshape(x, [-1, G, C // G, H, W])
                    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
                    x = (x - mean) / tf.sqrt(var + esp)
                    # per channel gamma and beta
                    gamma = tf.get_variable(name+'_GN_gamma', [C],
                                            initializer=tf.constant_initializer(1.0))
                    beta = tf.get_variable(name+'_GN_beta', [C],
                                           initializer=tf.constant_initializer(0.0))
                    gamma = tf.reshape(gamma, [1, C, 1, 1])
                    beta = tf.reshape(beta, [1, C, 1, 1])
                    # N -> -1
                    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
                    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
                    output = tf.transpose(output, [0, 2, 3, 1])
                    return output

                elif channel_mode == 'NCHW':
                    N, C, H, W = x.get_shape().as_list()
                    G = min(G, C)
                    # N -> -1
                    x = tf.reshape(x, [-1, G, C // G, H, W])
                    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
                    x = (x - mean) / tf.sqrt(var + esp)
                    # per channel gamma and beta
                    gamma = tf.get_variable(name + '_GN_gamma', [C],
                                            initializer=tf.constant_initializer(1.0))
                    beta = tf.get_variable(name + '_GN_beta', [C],
                                           initializer=tf.constant_initializer(0.0))
                    gamma = tf.reshape(gamma, [1, C, 1, 1])
                    beta = tf.reshape(beta, [1, C, 1, 1])
                    # N -> -1
                    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
                    return output

        elif norm_type == 'batch_match':
            output = batch_histogram_match_tensor(x, batch_size=batch_size, threshold = threshold, n_divide = n_divide, standard=standard, scale=scale,first=first)
            return output
        else:
            raise NotImplementedError

#############################################################################################################################
#                                                  Accuracy Functions                                                       #
#############################################################################################################################

def iou_coe(output, target, smooth=1e-5):
    # output : self.foreground_pred
    # target : self.foreground_truth
    # return : list of Batch IoU

    axis = [1, 2, 3]
    pre = tf.cast(output > 0.51, dtype=tf.float32)
    truth = tf.cast(target > 0.51, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = batch_iou

    return iou, inse, pre

def cal_result(pred, label, one_hot=False, e=1e-6):
    # convert one-hot labels to multiple labels
    if one_hot:
        _pred = np.argmax(pred, axis=-1)
        _label = np.argmax(label, axis=-1)

    else:
        _pred = pred
        _label = label

    _pred1 = _pred.flatten()
    _label1 = _label.flatten()
    _pred2 = _pred.reshape(np.shape(_pred)[0], -1)
    _label2 = _label.reshape(np.shape(_label)[0], -1)

    cm = confusion_matrix(_label1, _pred1, labels=[0, 1])
    TP = cm[1][1].astype(np.float32)
    FP = cm[0][1].astype(np.float32)
    FN = cm[1][0].astype(np.float32)
    TN = cm[0][0].astype(np.float32)

    # accuracy, sensitivity, specificity, mean iou, dice coefficient, hausdorff
    acc = (TP + TN + e) / (TP + FP + FN + TN + e)
    sens = (TP + e) / (TP + FN + e)
    spec = (TN + e) / (TN + FP + e)
    miou = (TP + e) / (FP + FN + TP + e)
    dice = (2 * TP + e) / (2 * TP + FP + FN + e)
    prec = (TP + e) /(TP + FP + e)
    hdorff = max(directed_hausdorff(_pred2, _label2)[0], directed_hausdorff(_label2, _pred2)[0])

    return [acc, sens, spec, prec, miou, dice, hdorff]

def convert_to_subregions(pred, label, convert_keys, one_hot=True):
    if one_hot:
        pred_arr = np.argmax(pred, axis=-1)
        label_arr = np.argmax(label, axis=-1)
    else:
        pred_arr = pred
        label_arr = label

    pred_list = []
    label_list = []

    _, pred_index = np.unique(pred_arr, return_inverse=True)
    _, label_index = np.unique(label_arr, return_inverse=True)

    for convert_key in convert_keys:
        key = np.array(convert_key)
        pred_list.append(key[pred_index].reshape(pred_arr.shape))
        label_list.append(key[label_index].reshape(label_arr.shape))

    return pred_list, label_list

#############################################################################################################################
#                                                    Loss Functions                                                         #
#############################################################################################################################

def huber_loss(output, target, delta=1.0):
    residual = tf.abs(output - target)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

# def focal_loss(output, target, smooth=1e-6):
#     focal = -tf.reduce_sum(tf.square(tf.ones_like(output) - output) * target * tf.log(target + smooth))
#     return focal

# def focal_loss(output, target, epsilon=1e-6):
#     ndim = len(output.get_shape())
#     output /= tf.reduce_sum(output, axis=(ndim - 1), keep_dims=True)
#     output = tf.clip_by_value(output, epsilon, 1 - epsilon)
#     focal = -tf.reduce_mean(tf.square(tf.ones_like(output)-output)*target*tf.log(output), axis=tuple(range(ndim-1)))
#     #focal = -tf.reduce_sum(tf.square(tf.ones_like(output)-output)*target*tf.log(output), axis=tuple(range(ndim-1)))
#
#     return tf.reduce_sum(focal)

def focal_loss(prediction_tensor, target_tensor, alpha=0.25, gamma=2, epsilon=1e-6):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent) / (192 * 160 * 10)

'''
?????? : label??? ?????? ?????? predict?????? ????????? ??? ????????? ????????? ????????? ?????? ?????? ????????????.
?????? : inse, l, r??? reduce_sum??? reduce_mean?????? ??????
1. pixel-wise??? ?????? ????????????
2. ??????????????? ?????? ????????? ?????????
3. ???????????? dice loss??? ?????????
4. ?????? ????????? ????????????

* ?????? ??????
1. ?????? ????????? ????????? ???????????? ??????????????? ????????? ???????????? ??????
2. ?????? ??? ????????? ????????? ????????? ?????? ??? ??? ????????? ????????????
'''

def dice_loss(output, target, axis=(1, 2, 3), smooth=1e-6):
    inse = tf.reduce_mean(output * target, axis=axis)
    l = tf.reduce_mean(output * output, axis=axis)
    r = tf.reduce_mean(target * target, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return 1-dice

def dice_loss_sum(output, target, axis=(1, 2, 3), smooth=1e-6):
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output * output, axis=axis)
    r = tf.reduce_sum(target * target, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return 1-dice

def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    # if len(ground_truth.shape) == len(prediction.shape):
    #     ground_truth = ground_truth[..., -1]
    one_hot = ground_truth

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        # weight_map_nclasses = tf.reshape(
        #     tf.tile(weight_map, [n_classes]), prediction.get_shape())
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, n_classes])
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        # ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        # intersect = tf.sparse_reduce_sum(one_hot * prediction,
        #                                  reduction_axes=[0])
        print('one_hot', one_hot)
        print('prediction', prediction)
        ref_vol = tf.reduce_sum(one_hot, [0])
        print('ref_vol', ref_vol)
        intersect = tf.reduce_sum(one_hot * prediction, [0])
        seg_vol = tf.reduce_sum(prediction, 0)
        print('seg_vol', seg_vol)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights,  tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return 1 - generalised_dice_score


def wasserstein_disagreement_map(
        prediction, ground_truth, weight_map=None, M=None):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened prediction and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.

    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    assert M is not None, "Distance matrix is required."
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    n_classes = prediction.shape[-1].value
    ground_truth.set_shape(prediction.shape)
    unstack_labels = tf.unstack(ground_truth, axis=-1)
    unstack_labels = tf.cast(unstack_labels, dtype=tf.float64)
    unstack_pred = tf.unstack(prediction, axis=-1)
    unstack_pred = tf.cast(unstack_pred, dtype=tf.float64)
    # print("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(unstack_pred[i], unstack_labels[j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map

def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

def generalised_wasserstein_dice_loss(prediction,
                                      ground_truth,
                                      weight_map=None):
    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in

        Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score
        for Imbalanced Multi-class Segmentation using Holistic
        Convolutional Networks.MICCAI 2017 (BrainLes)

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    M_tree = np.array([[0., 1., 1., 1., 1.],
                       [1., 0., 0.6, 0.2, 0.5],
                       [1., 0.6, 0., 0.6, 0.7],
                       [1., 0.2, 0.6, 0., 0.5],
                       [1., 0.5, 0.7, 0.5, 0.]], dtype=np.float64)

    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    prediction = tf.cast(prediction, tf.float32)
    n_classes = prediction.shape[-1].value
    # one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    one_hot = ground_truth

    # labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    # one_hot = tf.sparse_tensor_to_dense(one_hot)

    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree
    delta = wasserstein_disagreement_map(prediction, one_hot, M=M)
    print('delta', delta)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    print('all_error', all_error)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(one_hot, dtype=tf.float64)
    print('one_hot', one_hot)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
        axis=-1)
    print('true_pos', true_pos)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    print('true_pos2', true_pos)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)
    print('WGDL', WGDL)
    return tf.cast(WGDL, dtype=tf.float32)



def cross_entropy(output, target):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output))

def weighted_categorical_cross_entropy(output, target, weights, epsilon=1e-6):
    ndim = len(output.get_shape())
    ncategory = output.get_shape[-1]
    # scale predictions so class probabilities of each pixel sum to 1
    output /= tf.reduce_sum(output, axis=(ndim-1), keep_dims=True)
    output = tf.clip_by_value(output, epsilon, 1-epsilon)
    w = tf.constant(weights) * (ncategory / sum(weights))
    # first, average over all axis except classes
    cross_entropies = -tf.reduce_mean(target * tf.log(output), axis=tuple(range(ndim-1)))
    return tf.reduce_sum(w * cross_entropies)

def select_loss(mode, output, target, smooth=1e-6, weight=1, epsilon=1e-6, delta=1.0):
    if mode == 'dice':
        return dice_loss(output, target, smooth=smooth)
    elif mode == 'focal':
        return focal_loss(output, target, epsilon=epsilon)
    elif mode == 'cross_entropy':
        return cross_entropy(output, target)
    elif mode == 'dice_sum':
        return dice_loss_sum(output, target, smooth=smooth)
    elif mode == 'huber':
        return huber_loss(output, target, delta=delta)
    elif mode == 'weighted_cross_entropy':
        return weighted_categorical_cross_entropy(output, target, weights=weight)
    else:
        print("Not supported loss function. Select among dice, focal, cross_entropy, dice_sum, huber,weighted_cross_entropy")

#############################################################################################################################
#                                                      Optimizer                                                            #
#############################################################################################################################

def select_optimizer(mode, learning_rate, loss, global_step):
    if mode == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    elif mode == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    elif mode == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    else:
        print("Not supported optimizer. Select among adam, rmsprop, sgd")

#############################################################################################################################
#                                                  Sampling Functions                                                       #
#############################################################################################################################

def select_downsampling(name, down_conv, down_pool, channel_n, pool_size_h, pool_size_w, mode):
    if mode == 'neighbor':
        shape = [-1, pool_size_w, pool_size_h, channel_n]
        down_pool = re_conv2D(name + '_neighbor', down_conv, shape)

    elif mode == 'maxpool':
        down_pool = maxpool(name + '_maxpool', down_conv, [2,2], [2,2], 'SAME')

    elif mode == 'avgpool':
        down_pool = averagepool(name + '_avgpool', down_conv, [2,2], [2,2], 'SAME')

    return down_pool

def select_downsampling2(name, down_conv, channel_n, pool_size_h, pool_size_w, mode):
    if mode == 'neighbor':
        shape = [-1, pool_size_w, pool_size_w, channel_n]
        down_pool = re_conv2D(name + '_neighbor', down_conv, shape)

    elif mode == 'maxpool':
        down_pool = maxpool(name + '_maxpool', down_conv, [2,2], [2,2], 'SAME')

    elif mode == 'avgpool':
        down_pool = averagepool(name + '_avgpool', down_conv, [2,2], [2,2], 'SAME')

    return down_pool

def select_upsampling(name, up_conv, up_pool, channel_n, pool_size_h, pool_size_w, mode):
    shape = [-1, pool_size_w, pool_size_h, channel_n]

    if mode == 'resize':
      up_pool = re_conv2D(name + '_reconv', up_conv, shape)

    elif mode == 'transpose':
        up_pool = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2], shape, [1,2,2,1], 'SAME')
        up_pool = tf.reshape(up_pool, shape)

    elif mode == 'add':
        up_pool1 = re_conv2D(name + '_reconv', up_conv, shape)
        up_pool2 = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2], shape, [1,2,2,1], 'SAME')
        up_pool2 = tf.reshape(up_pool2, shape)
        up_pool = up_pool1 + up_pool2

    elif mode == 'concat':
        up_pool1 = re_conv2D(name + '_reconv', up_conv, shape)
        up_pool2 = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2], shape, [1, 2, 2, 1], 'SAME')
        up_pool2 = tf.reshape(up_pool2, shape)
        up_pool = concat(name + '_upsampling_concat', [up_pool1, up_pool2], axis=3)
        up_pool = conv2D(name + '_bottleneck', up_pool, channel_n, [1,1], [1,1], padding='SAME')

    return up_pool

def select_upsampling2(name, up_conv, channel_n, pool_size_h, pool_size_w, mode):
    shape = [-1, pool_size_h, pool_size_w, channel_n]

    if mode == 'resize':
      up_pool = re_conv2D(name + '_reconv', up_conv, shape)

    elif mode == 'transpose':
        up_pool = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2], shape, [1,2,2,1], 'SAME')
        up_pool = tf.reshape(up_pool, shape)

    elif mode == 'add':
        up_pool1 = re_conv2D(name + '_reconv', up_conv, shape)
        up_pool2 = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2], shape, [1,2,2,1], 'SAME')
        up_pool2 = tf.reshape(up_pool2, shape)
        up_pool = up_pool1 + up_pool2

    elif mode == 'concat':
        up_pool1 = re_conv2D(name + '_reconv', up_conv, shape)
        up_pool2 = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2], shape, [1, 2, 2, 1], 'SAME')
        up_pool2 = tf.reshape(up_pool2, shape)
        up_pool = concat(name + '_upsampling_concat', [up_pool1, up_pool2], axis=3)
        up_pool = conv2D(name + '_bottleneck', up_pool, channel_n, [1,1], [1,1], padding='SAME')

    elif mode == 'avgpool' :
        filter = tf.ones([2, 2, channel_n, channel_n * 2])  # [height, width, output_channels, in_channels]
        batch_size = tf.shape(up_conv)[0]
        shape2 = [batch_size, pool_size_h, pool_size_w, channel_n]
        up_pool = tf.nn.conv2d_transpose(up_conv, filter, shape2, [1, 2, 2, 1], 'SAME')

    return up_pool

#############################################################################################################################
#                                                    Save Functions                                                         #
#############################################################################################################################

def result_saver(path, data):
    with open(path, 'at') as f:
        f.write(data)
        f.write('\n')

def save_array_as_nifty_volume(data, filename):
   img = nibabel.Nifti1Image(data, affine=np.eye(4))
   nibabel.save(img, filename)

#############################################################################################################################
#                                                      CNN Models                                                           #
#############################################################################################################################

# Unet  (https://arxiv.org/abs/1505.04597)
def unet_down_block(inputs, conv_list, pool_list, channel_n, pool_size_h, pool_size_w, group_n, act_fn, norm_type, down_type, training, idx):
    conv_list[idx] = conv2D(str(idx) + '_downconv1', inputs, channel_n, [3, 3], [1, 1], padding='SAME')
    conv_list[idx] = Normalization(conv_list[idx], norm_type, training, str(idx) + '_downnorm1', G=group_n)
    conv_list[idx] = activation(str(idx) + '_downact1', conv_list[idx], act_fn)

    conv_list[idx] = conv2D(str(idx) + '_downconv2', conv_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
    conv_list[idx] = Normalization(conv_list[idx], norm_type, training, str(idx) + '_downnorm2', G=group_n)
    conv_list[idx] = activation(str(idx) + '_downact2', conv_list[idx], act_fn)
    print('down' + str(idx + 1) + 'conv', conv_list[idx])
    pool_list[idx] = select_downsampling(str(idx) + '_downsampling',
                                         conv_list[idx],
                                         pool_list[idx],
                                         channel_n,
                                         pool_size_h,
                                         pool_size_w,
                                         down_type)
    print('down' + str(idx + 1) + 'pool', pool_list[idx])

    if down_type == 'neighbor':
        conv_list[idx] = Normalization(conv_list[idx], norm_type, training,
                                             str(idx) + '_norm3', G=group_n)
        conv_list[idx] = activation(str(idx) + '_act3', conv_list[idx], act_fn)

    return pool_list[idx]

def unet_same_block(inputs, channel_n, group_n, act_fn, norm_type, training,batch_size, name='same', threshold = 'fuzzy',n_divide = 4,standard=False, scale=1):
    conv_list = conv2D(name + 'same_conv1', inputs, channel_n, [3, 3], [1, 1], padding='SAME')
    conv_list = Normalization(conv_list, norm_type, training, name + 'same_norm1', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale)
    conv_list = activation(name + 'same_act1', conv_list, act_fn)
    conv_list = conv2D(name + 'same_conv2', conv_list, channel_n, [1, 1], [1, 1], padding='SAME')
    conv_list = Normalization(conv_list, norm_type, training, name + 'same_norm2', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale)
    conv_list = activation(name + 'same_act2', conv_list, act_fn)

    return conv_list

def unet_up_block(inputs, downconv_list, upconv_list, pool_list, channel_n, group_n, act_fn, norm_type, training, idx,batch_size, name='up',threshold = 'fuzzy',n_divide = 4,standard=False, scale=1):
    pool_list[idx] = Normalization(inputs, norm_type, training, name + str(idx) + '_norm1', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale)
    pool_list[idx] = activation(name + str(idx) + '_upsampling_act', pool_list[idx], act_fn)
    pool_list[idx] = concat(name + str(idx) + '_upconcat', [pool_list[idx], downconv_list[idx]], axis=3)

    upconv_list[idx] = conv2D(name + str(idx) + '_upconv1', pool_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
    upconv_list[idx] = Normalization(upconv_list[idx], norm_type, training, name + str(idx) + '_upnorm1', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale)
    upconv_list[idx] = activation(name + str(idx) + '_upact1', upconv_list[idx], act_fn)

    upconv_list[idx] = conv2D(name + str(idx) + '_upconv2', upconv_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
    upconv_list[idx] = Normalization(upconv_list[idx], norm_type, training, name + str(idx) + '_upnorm2', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale)
    upconv_list[idx] = activation(name + str(idx) + '_upact2', upconv_list[idx], act_fn)
    print('up' + str(idx + 1) + 'conv', upconv_list[idx])

    return upconv_list[idx]

# resnet  (https://arxiv.org/abs/1512.03385)
def residual_block_v1(inputs, channel_n, group_n, act_fn, norm_type, training, idx, shortcut=True):
    # input
    il = conv2D(str(idx) + '_input', inputs, channel_n, [1, 1], [1, 1], padding='SAME')
    il = Normalization(il, norm_type, training, str(idx) + '_input_norm', G=group_n)
    il = activation(str(idx) + '_input_act', il, act_fn)

    # bottleneck1
    hl = conv2D(str(idx) + '_bottleneck1', inputs, int(channel_n/4), [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, str(idx) + '_bottleneck_norm1', G=group_n)
    hl = activation(str(idx) + '_bottleneck_act1', hl, act_fn)

    # conv
    hl = conv2D(str(idx) + '_conv', hl, int(channel_n / 4), [3, 3], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, str(idx) + '_conv_norm', G=group_n)
    hl = activation(str(idx) + '_conv_act', hl, act_fn)

    # bottleneck2
    hl = conv2D(str(idx) + '_bottleneck2', hl, channel_n, [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, str(idx) + '_bottleneck_norm2', G=group_n)
    hl = activation(str(idx) + '_bottleneck_act2', hl, act_fn)

    hl = il + hl if shortcut else hl

    return hl

def residual_block_v1_dr(name, inputs, channel_n, group_n, drop_rate, act_fn, norm_type, training, idx, batch_size, shortcut=True, threshold = 'fuzzy',n_divide = 4,standard=False, scale=1):
    # input
    il = conv2D(name + str(idx) + '_input', inputs, channel_n, [1, 1], [1, 1], padding='SAME')
    il = Normalization(il, norm_type, training, name + str(idx) + '_input_norm', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale,first=True)
    il = activation(name + str(idx) + '_input_act', il, act_fn)

    # bottleneck1
    hl = conv2D(name + str(idx) + '_bottleneck1', inputs, int(channel_n/4), [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale,first=False)
    hl = activation(name + str(idx) + '_bottleneck_act1', hl, act_fn)
    # hl = dropout(name + str(idx) + '_dropout1', hl, drop_rate, training)

    # conv
    hl = conv2D(name + str(idx) + '_conv', hl, int(channel_n / 4), [3, 3], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_conv_norm', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale,first=False)
    hl = activation(name + str(idx) + '_conv_act', hl, act_fn)
    hl = dropout(name + str(idx) + '_dropout2', hl, drop_rate, training)

    # bottleneck2
    hl = conv2D(name + str(idx) + '_bottleneck2', hl, channel_n, [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_bottleneck_norm2', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale,first=False)
    hl = activation(name + str(idx) + '_bottleneck_act2', hl, act_fn)
    # hl = dropout(name + str(idx) + '_dropout3', hl, drop_rate, training)

    hl = il + hl if shortcut else hl

    return hl

def residual_block_dw_dr(name, inputs, channel_n, width_mul, group_n, drop_rate, act_fn, norm_type, training, idx, rate=None, shortcut=True):
    # input
    il = conv2D(name + str(idx) + '_input', inputs, channel_n, [1, 1], [1, 1], padding='SAME')
    il = Normalization(il, norm_type, training, name + str(idx) + '_input_norm', G=group_n)
    il = activation(name + str(idx) + '_input_act', il, act_fn)

    # bottleneck1
    hl = conv2D(name + str(idx) + '_bottleneck1', inputs, int(channel_n/4), [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
    hl = activation(name + str(idx) + '_bottleneck_act1', hl, act_fn)
    hl = dropout(name + str(idx) + '_dropout1', hl, drop_rate, training)

    # depthwise
    depthwise_filter = tf.get_variable(name=name+'depthwise_filter' + str(idx),
                                       shape=[3, 3, int(channel_n/4), width_mul],
                                       dtype=tf.float32,
                                       initializer=initializer)  # [filter_height, filter_width, in_channels, channel_multiplier]
    hl = tf.nn.depthwise_conv2d(hl, depthwise_filter, [1, 1, 1, 1], 'SAME', rate=rate, name = name + str(idx) + '_depthwise')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_depthwise_norm', G=group_n)
    hl = activation(name + str(idx) + '_depthwise_act', hl, act_fn)
    hl = dropout(name + str(idx) + '_dropout2', hl, drop_rate, training)

    # bottleneck2
    hl = conv2D(name + str(idx) + '_bottleneck2', hl, channel_n, [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_bottleneck_norm2', G=group_n)
    hl = activation(name + str(idx) + '_bottleneck_act2', hl, act_fn)
    hl = dropout(name + str(idx) + '_dropout3', hl, drop_rate, training)

    hl = il + hl if shortcut else hl

    return hl

# densenet  (https://arxiv.org/abs/1608.06993)
def dense_layer(name, inputs, group_n, drop_rate, act_fn, norm_type, growth, training, idx):
    # bottleneck
    l = Normalization(inputs, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
    l = activation(name + str(idx) + '_bottleneck_act1', l, act_fn)
    l = conv2D(name + str(idx) + '_bottleneck1', l, 4 * growth, [1, 1], [1, 1], 'SAME')
    l = dropout(name + str(idx) + '_dropout1', l, drop_rate, training)

    # conv
    l = Normalization(l, norm_type, training, name + str(idx) + '_bottleneck_norm2', G=group_n)
    l = activation(name + str(idx) + '_bottleneck_act2', l, act_fn)
    l = conv2D(name + str(idx) + '_bottleneck2', l, growth, [3, 3], [1, 1], 'SAME')
    l = dropout(name + str(idx) + '_dropout2', l, drop_rate, training)

    return l

def transition_layer(name, inputs, group_n, act_fn, norm_type, theta, training, specific_n_channels=False, idx=0):
    if specific_n_channels:
        l = Normalization(inputs, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
        l = activation(name + str(idx) + '_bottleneck_act1', l, act_fn)
        l = conv2D(name + str(idx) + '_bottleneck1', l, specific_n_channels, [1, 1], [1, 1], 'SAME')

    else:
        n_channels = inputs.get_shape().as_list()[-1]

        l = Normalization(inputs, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
        l = activation(name + str(idx) + '_bottleneck_act1', l, act_fn)
        l = conv2D(name + str(idx) + '_bottleneck1', l, n_channels * theta, [1, 1], [1, 1], padding='SAME')

    return l

def dense_block_v1(name, inputs, group_n, drop_rate, act_fn, norm_type, growth, training, n_layer):
    hl = tf.identity(inputs)

    for idx in range(n_layer):
        l = dense_layer(name = name,
                        inputs = hl,
                        group_n = group_n,
                        drop_rate = drop_rate,
                        act_fn = act_fn,
                        norm_type = norm_type,
                        growth = growth,
                        training = training,
                        idx = idx)
        hl = tf.concat([hl, l], axis=3)

    return hl

# depthwise separable conv  (Mobilenet - https://arxiv.org/abs/1704.04861)
def depthwise_separable_convlayer(name, inputs, channel_n, width_mul, group_n, act_fn, norm_type, training, idx, rate=None):
    # depthwise
    depthwise_filter = tf.get_variable(name=name+'depthwise_filter' + str(idx),
                                       shape=[3, 3, inputs.get_shape()[-1], width_mul],
                                       dtype=tf.float32,
                                       initializer=initializer)
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter, [1, 1, 1, 1], 'SAME', rate=rate, name = name + str(idx) + '_depthwise')
    l = Normalization(l, norm_type, training, name + str(idx) + '_depthwise_norm', G=group_n)
    l = activation(name + str(idx) + '_depthwise_act1', l, act_fn)

    # pointwise
    l = conv2D(name + str(idx) + '_pointwise', l, channel_n, [1, 1], [1, 1], padding='SAME')
    l = Normalization(l, norm_type, training, name + str(idx) + '_pointwise_norm1', G=group_n)
    l = activation(name + str(idx) + '_pointwise_act1', l, act_fn)

    return l

# def depthwise_separable_convlayer_dr(name, inputs, channel_n, width_mul, group_n, drop_rate, act_fn, norm_type, training, idx,  rate=None):
#     # depthwise
#     depthwise_filter = tf.get_variable(name=name + 'depthwise_filter' + str(idx),
#                                        shape=[3, 3, inputs.get_shape()[-1], width_mul],
#                                        dtype=tf.float32,
#                                        initializer=initializer)
#     l = tf.nn.depthwise_conv2d(inputs, depthwise_filter, [1, 1, 1, 1], 'SAME', rate=rate,
#                                name=name + str(idx) + '_depthwise')
#     l = Normalization(l, norm_type, training, name + str(idx) + '_depthwise_norm', G=group_n)
#     l = activation(name + str(idx) + '_depthwise_act1', l, act_fn)
#     l = dropout(name + str(idx) + '_dropout2', l, drop_rate, training)
#
#     # pointwise
#     l = conv2D(name + str(idx) + '_pointwise', l, channel_n, [1, 1], [1, 1], padding='SAME')
#     l = Normalization(l, norm_type, training, name + str(idx) + '_pointwise_norm1', G=group_n)
#     l = activation(name + str(idx) + '_pointwise_act1', l, act_fn)
#     l = dropout(name + str(idx) + '_dropout2', l, drop_rate, training)
#
#     return l

def depthwise_separable_convlayer_dr(name, batch_size, inputs, channel_n, width_mul, group_n, drop_rate, act_fn, norm_type, training, idx,rate=None,threshold = 'fuzzy',n_divide = 4,standard=False, scale=1):
    # depthwise
    depthwise_filter = tf.get_variable(name=name + 'depthwise_filter' + str(idx),
                                       shape=[3, 3, inputs.get_shape()[-1], width_mul],
                                       dtype=tf.float32)
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter, [1, 1, 1, 1], 'SAME', rate=rate,
                               name=name + str(idx) + '_depthwise')
    l = Normalization(l, norm_type, training, name + str(idx) + '_depthwise_norm', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale,first=True)
    l = activation(name + str(idx) + '_depthwise_act1', l, act_fn)
    l = dropout(name + str(idx) + '_dropout2', l, drop_rate, training)

    # pointwise
    l = conv2D(name + str(idx) + '_pointwise', l, channel_n, [1, 1], [1, 1], padding='SAME')
    l = Normalization(l, norm_type, training, name + str(idx) + '_pointwise_norm1', G=group_n,batch_size=batch_size, threshold = threshold,n_divide = n_divide,standard=standard, scale=scale,first=False)
    l = activation(name + str(idx) + '_pointwise_act1', l, act_fn)
    l = dropout(name + str(idx) + '_dropout2', l, drop_rate, training)

    return l

# shufflenet  (https://arxiv.org/abs/1707.01083)

def channel_shuffle(name, inputs, group_n):
    with tf.variable_scope(name):
        # _, h, w, c = l.shape
        _, h, w, c = inputs.get_shape().as_list()
        _l = tf.reshape(inputs, [-1, h, w, group_n, c // group_n])
        _l = tf.transpose(_l, [0, 1, 2, 4, 3])
        l = tf.reshape(_l, [-1, h, w, c])

        return l

def group_conv2D(name, inputs, channel_n, kernel_size, group_n, stride, act_fn, norm_type, training, idx, use_act_fn=True, padding='SAME'):
    in_channel_per_group = inputs.get_shape().as_list()[-1] // group_n
    out_channel_per_group = channel_n // group_n
    grouped_channel_list = []

    for i in range(group_n):
        _l = conv2D(name = name + str(i),
                    inputs = inputs[:, :, :, i * in_channel_per_group: i * in_channel_per_group + in_channel_per_group],
                    filters = out_channel_per_group,
                    kernel_size = [kernel_size, kernel_size],
                    strides = [stride, stride],
                    padding = padding)
        grouped_channel_list.append(_l)

    _l = tf.concat(grouped_channel_list, axis=-1, name='concat_channel')
    _l = Normalization(_l, norm_type, training, name + str(idx) + 'norm', G=group_n)

    if use_act_fn:
        _l = activation(name + str(idx) + 'act', _l, act_fn)

    return _l

def shufflenet_unit(name, inputs, channel_n, group_n, stride, act_fn, norm_type, training, idx):
    if stride != 1:
        *_, c = inputs.get_shape().as_list()
        channel_n = channel_n - c # Residual ?????? ??? c??? + Group ?????? ??? = ?????? ?????????(num_filter) ??? ????????? ?????????

    # Residual part
    if stride != 1:
        residual_layer = averagepool('residual' + str(idx), inputs, [2,2], [2,2], 'SAME')
    else:
        residual_layer = tf.identity(inputs)

    # Group part
    depthwise_filter = tf.get_variable(name = name + 'depthwise_filter' + str(idx),
                                       shape = [3,3,inputs.get_shape()[-1], 1],
                                       dtype = tf.float32,
                                       initializer = tf.contrib.layers.variance_scaling_initializer())
    _group_layer = group_conv2D(name = name + '_group_conv1',
                                inputs = inputs,
                                channel_n = channel_n,
                                kernel_size = 1,
                                group_n = group_n,
                                stride = 1,
                                act_fn = act_fn,
                                norm_type = norm_type,
                                training = training,
                                idx = idx)

    _shuffled_group_layer = channel_shuffle(name = name + '_channel_shuffle',
                                            inputs = _group_layer,
                                            group_n = group_n)

    _depthwise_conv_layer = tf.nn.depthwise_conv2d(input = _shuffled_group_layer,
                                                   filter = depthwise_filter,
                                                   strides = [1,stride,stride,1],
                                                   padding = 'SAME',
                                                   name = name + str(idx) + '_depthwise')

    _depthwise_conv_layer = Normalization(x = _depthwise_conv_layer,
                                          norm_type = norm_type,
                                          is_train = training,
                                          name = name + str(idx) + 'depthwise_norm',
                                          G = group_n)

    final_group_layer = group_conv2D(name = name + '_group_conv2',
                                     inputs = _depthwise_conv_layer,
                                     channel_n = channel_n,
                                     kernel_size = 1,
                                     group_n = group_n,
                                     stride = 1,
                                     act_fn = act_fn,
                                     norm_type = norm_type,
                                     training = training,
                                     idx = idx,
                                     use_act_fn=False)

    # Concat part
    if stride != 1:
        layer = tf.concat([residual_layer, final_group_layer], axis=3)

    else:
        layer = residual_layer + final_group_layer

    final_layer = activation(name + str(idx) + '_shuffleunit_act', layer, act_fn)

    return final_layer

def shufflenet_stage(name, inputs, channel_n, group_n, act_fn, norm_type, training, repeat):
    l = shufflenet_unit(name = name,
                        inputs = inputs,
                        channel_n = channel_n,
                        group_n = group_n,
                        stride = 2,
                        act_fn = act_fn,
                        norm_type = norm_type,
                        training=training,
                        idx=0)

    for i in range(repeat):
        l = shufflenet_unit(name = name + str(i),
                            inputs = inputs,
                            channel_n = channel_n,
                            group_n = group_n,
                            stride = 1,
                            act_fn = act_fn,
                            norm_type = norm_type,
                            training=training,
                            idx=i+1)

    return l

# HENet  (https://arxiv.org/abs/1803.02742)
def he_hlayer(name, inputs, channel_n, group_m, group_n, act_fn, norm_type, training, idx):
    l = group_conv2D(name = name + str(idx) + '_1st_Gconv',
                     inputs = inputs,
                     channel_n = channel_n,
                     kernel_size = 1,
                     group_n = group_m,
                     stride = 1,
                     act_fn = act_fn,
                     norm_type = norm_type,
                     training = training,
                     idx = 0,
                     use_act_fn = False)

    l = channel_shuffle(name = name + str(idx) + '_channelshuffle',
                        inputs = l,
                        group_n = group_m)

    l = group_conv2D(name = name + str(idx) + '_2nd_Gconv',
                     inputs = l,
                     channel_n = channel_n,
                     kernel_size = 3,
                     group_n = group_n,
                     stride = 1,
                     act_fn = act_fn,
                     norm_type = norm_type,
                     training = training,
                     idx = 1,
                     use_act_fn = True)
    return l

def he_s1block(name, inputs, channel_n, group_m, group_n, act_fn, norm_type, training, repeat):
    X = tf.identity(inputs)
    l = inputs
    for i in range(repeat):
        HL = he_hlayer(name = name + '_hlayer{}'.format(i + 1),
                       inputs = l,
                       channel_n = channel_n,
                       group_m = group_m,
                       group_n = group_n,
                       act_fn = act_fn,
                       norm_type = norm_type,
                       training = training,
                       idx = i)
        X = X + HL
        l = tf.concat([HL, X], axis=3)

    return l

def he_s2block(name, inputs, channel_n, group_m, group_n, act_fn, norm_type, training, resize = True):
    l = group_conv2D(name = name + '_1st_s2_Gconv',
                     inputs = inputs,
                     channel_n = channel_n // 2,
                     kernel_size = 3,
                     group_n = group_m,
                     stride = 2 if resize else 1,
                     act_fn = act_fn,
                     norm_type = norm_type,
                     training = training,
                     idx = 0,
                     use_act_fn = False)

    l = channel_shuffle(name = name + '_s2_channelshuffle',
                        inputs = l,
                        group_n = group_m)

    l = group_conv2D(name = name + '_2nd_s2_Gconv',
                     inputs = l,
                     channel_n = channel_n,
                     kernel_size = 1,
                     group_n = group_n,
                     stride = 1,
                     act_fn = act_fn,
                     norm_type = norm_type,
                     training = training,
                     idx = 1,
                     use_act_fn = True)
    return l

def he_stage(name, inputs, channel_in, channel_out, group_m, group_n, act_fn, norm_type, training, repeat, resize=True, last_stage=False):
    if last_stage:
        _, h, _, _ = inputs.get_shape().as_list()

        l = group_conv2D(name = name + 'last_stage1',
                         inputs = inputs,
                         channel_n = channel_in,
                         kernel_size = h,
                         group_n = group_m,
                         stride = 1,
                         act_fn = act_fn,
                         norm_type = norm_type,
                         training = training,
                         idx = 0,
                         use_act_fn = False,
                         padding = 'VALID')

        l = channel_shuffle(name = name + '_last_channelshuffle',
                            inputs = l,
                            group_n = group_m)

        l = group_conv2D(name = name + 'last_stage2',
                         inputs = l,
                         channel_n = channel_out,
                         kernel_size = 1,
                         group_n = group_n,
                         stride = 1,
                         act_fn = act_fn,
                         norm_type = norm_type,
                         training = training,
                         idx = 1,
                         use_act_fn = True)
    else:
        l = he_s1block(name = name + 's1_block',
                       inputs = inputs,
                       channel_n = channel_in,
                       group_m = group_m,
                       group_n = group_n,
                       act_fn = act_fn,
                       norm_type = norm_type,
                       training = training,
                       repeat = repeat)

        l = he_s2block(name = name + 's2_block',
                       inputs = l,
                       channel_n = channel_out,
                       group_m = group_m,
                       group_n = group_n,
                       act_fn = act_fn,
                       norm_type = norm_type,
                       training = training,
                       resize = resize)

    return l

################################################################################################


def atrous_spatial_pyramid_pooling(name, inputs, channel_n, atrous_rate_list, act_fn, training):
    atrous_rates = atrous_rate_list

    ### a) 1x1 Conv * 1  +  3x3 Conv * 3
    conv_1x1 = conv2D(name + '_a_1x1', inputs, channel_n, [1,1], [1,1], padding='SAME')
    conv_1x1 = Normalization(conv_1x1, 'batch', training, name + '_a_1x1_norm')
    conv_1x1 = activation(name + '_a_1x1_act', conv_1x1, act_fn)

    conv_3x3_0 = conv2D(name + '_a_3x3_0', inputs, channel_n, [3,3], [1,1], dilation_rate=atrous_rates[0], padding='SAME')
    conv_3x3_0 = Normalization(conv_3x3_0, 'batch', training, name + '_a_3x3_0_norm')
    conv_3x3_0 = activation(name + 'a_3x3_0_act', conv_3x3_0, act_fn)

    conv_3x3_1 = conv2D(name + '_a_3x3_1', inputs, channel_n, [3, 3], [1, 1], dilation_rate=atrous_rates[1], padding='SAME')
    conv_3x3_1 = Normalization(conv_3x3_1, 'batch', training, name + '_a_3x3_1_norm')
    conv_3x3_1 = activation(name + 'a_3x3_1_act', conv_3x3_1, act_fn)

    conv_3x3_2 = conv2D(name + '_a_3x3_2', inputs, channel_n, [3, 3], [1, 1], dilation_rate=atrous_rates[2], padding='SAME')
    conv_3x3_2 = Normalization(conv_3x3_2, 'batch', training, name + '_a_3x3_2_norm')
    conv_3x3_2 = activation(name + 'a_3x3_2_act', conv_3x3_2, act_fn)

    ### (b) the image-level features
    # global average pooling
    img_lv_features = GlobalAveragePooling2D(inputs, channel_n, name + '_GAP', keep_dims=True)
    # 1x1 conv
    img_lv_features = conv2D(name + '_img_lv_features', img_lv_features, channel_n, [1,1], [1,1], padding='SAME')
    img_lv_features = Normalization(img_lv_features, 'batch', training, name + '_img_lv_features_norm')
    img_lv_features = activation(name + '_img_lv_features_act', img_lv_features, act_fn)
    # upsampling
    img_lv_features = re_conv2D(name + '_upsample', img_lv_features, [-1, tf.shape(inputs)[1], tf.shape(inputs)[2], channel_n])
    # concat
    aspp_layer = tf.concat([conv_1x1, conv_3x3_0, conv_3x3_1, conv_3x3_2, img_lv_features], axis=3, name=name+'_concat')
    # 1x1 conv
    aspp_layer = conv2D(name + '_aspp_layer', aspp_layer, channel_n, [1,1], [1,1], padding='SAME')
    aspp_layer = Normalization(aspp_layer, 'batch', training, name + '_aspp_layer_norm')
    aspp_layer = activation(name + '_aspp_layer_act', aspp_layer, act_fn)

    return aspp_layer

def xception_depthwise_separable_convlayer(name, inputs, channel_n, last_stride, act_fn, training, shortcut_conv=False, atrous=False, atrous_rate=2):
    rate = [atrous_rate, atrous_rate] if atrous else None
    # shortcut layer
    shortcut = tf.identity(inputs)

    if shortcut_conv:
        shortcut = conv2D(name + '_shortcut', shortcut, channel_n, [1,1], [last_stride, last_stride], padding='SAME')
        shortcut = Normalization(shortcut, 'batch', training, name + '_shortcut_norm')
        shortcut = activation(name + '_shortcut_act', shortcut, act_fn)

    in_channel = inputs.get_shape().as_list()[-1]
    width_mul = int(channel_n / in_channel)

    depthwise_filter1 = tf.get_variable(name = name + '_depthwise_filter1',
                                        shape = [3, 3, in_channel, width_mul],
                                        dtype = tf.float32,
                                        initializer = initializer)

    depthwise_filter2 = tf.get_variable(name = name + '_depthwise_filter2',
                                        shape = [3, 3, channel_n, 1],
                                        dtype = tf.float32,
                                        initializer = initializer)
    # conv layer 1
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter1, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep1')
    l = Normalization(l, 'batch', training, name + '_sep_norm1')
    l = activation(name + '_sep_act1', l, act_fn)

    # conv layer 2
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep2')
    l = Normalization(l, 'batch', training, name + '_sep_norm2')
    l = activation(name + '_sep_act2', l, act_fn)

    # conv layer 3
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, last_stride, last_stride, 1], 'SAME', rate = rate, name = name + '_sep3')
    l = Normalization(l, 'batch', training, name + '_sep_norm3')
    l = activation(name + '_sep_act3', l, act_fn)

    # add layer
    l = l + shortcut

    return l

def xception_depthwise_separable_convlayer_dr(name, inputs, channel_n, last_stride, drop_rate, act_fn, training, shortcut_conv=False, atrous=False, atrous_rate=2):
    rate = [atrous_rate, atrous_rate] if atrous else None
    # shortcut layer
    shortcut = tf.identity(inputs)

    if shortcut_conv:
        shortcut = conv2D(name + '_shortcut', shortcut, channel_n, [1,1], [last_stride, last_stride], padding='SAME')
        shortcut = Normalization(shortcut, 'batch', training, name + '_shortcut_norm')
        shortcut = activation(name + '_shortcut_act', shortcut, act_fn)

    in_channel = inputs.get_shape().as_list()[-1]
    width_mul = int(channel_n / in_channel)

    depthwise_filter1 = tf.get_variable(name = name + '_depthwise_filter1',
                                        shape = [3, 3, in_channel, width_mul],
                                        dtype = tf.float32,
                                        initializer = initializer)

    depthwise_filter2 = tf.get_variable(name = name + '_depthwise_filter2',
                                        shape = [3, 3, channel_n, 1],
                                        dtype = tf.float32,
                                        initializer = initializer)
    # conv layer 1
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter1, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep1')
    l = Normalization(l, 'batch', training, name + '_sep_norm1')
    l = activation(name + '_sep_act1', l, act_fn)
    l = dropout(name + '_dropout1', l, drop_rate, training)

    # conv layer 2
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep2')
    l = Normalization(l, 'batch', training, name + '_sep_norm2')
    l = activation(name + '_sep_act2', l, act_fn)
    l = dropout(name + '_dropout1', l, drop_rate, training)

    # conv layer 3
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, last_stride, last_stride, 1], 'SAME', rate = rate, name = name + '_sep3')
    l = Normalization(l, 'batch', training, name + '_sep_norm3')
    l = activation(name + '_sep_act3', l, act_fn)
    l = dropout(name + '_dropout1', l, drop_rate, training)

    # add layer
    l = l + shortcut

    return l
#############################################################################################################################
#                                                    Result Function                                                        #
#############################################################################################################################

def masking_rgb(img, color=None):
    if len(np.shape(img)) <= 3:
        _img = np.expand_dims(img, axis=3)
    else:
        _img = img
    rgb_list = [np.zeros(np.shape(_img)) for _ in range(3)]

    if color == 'yellow':
        rgb_list[1] = _img
        rgb_list[2] = _img
        B, G, R = rgb_list

    elif color != None:
        rgb_dic = {'blue': 0, 'green': 1, 'red': 2}
        rgb_list[rgb_dic[color]] = _img
        B, G, R = rgb_list
    else:
        B = G = R = _img

    concat_img = np.concatenate((B, G, R), axis=-1)
    out_img = concat_img * 255

    return out_img

#############################################################################################################################
#                                        Feature Extraction Function                                                        #
#############################################################################################################################

# def rgb2gray(rgb):
#     '''
#     Change rgb to grayscale image
#     '''
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
#
# def get_ND_bounding_box(input, margin):
#     '''
#     get bounding box excluding background
#     :return: min, max indexes, and width, length, height of bounding box
#     '''
#     input_shape = input.shape
#     if(type(margin) is int ):
#         margin = [margin]*len(input_shape)
#     assert(len(input_shape) == len(margin))
#     indices = np.nonzero(input)
#     idx_min = []
#     idx_max = []
#     for i in range(len(input_shape)):
#         idx_min.append(indices[i].min())
#         idx_max.append(indices[i].max())
#     for i in range(len(input_shape)):
#         idx_min[i] = max(idx_min[i] - margin[i], 0)
#         idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
#     width = idx_max[0] - idx_min[0]
#     length = idx_max[1] - idx_min[1]
#     height = idx_max[2] - idx_min[2]
#     return idx_min, idx_max, width, length, height
#
# def crop_volume_with_bounding_box(volume, min_idx, max_idx):
#     output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
#                                range(min_idx[1], max_idx[1] + 1),
#                                range(min_idx[2], max_idx[2] + 1))]
#     return output
#
# def get_area(label):
#     indices = np.nonzero(label)
#     area = len(indices[0])
#     return area
#
# def get_volume(label):
#     indices = np.nonzero(label)
#     vol = len(indices[0])
#     return vol
#
# def get_glcm(input):
#     '''
#     get gray level co-occurrence matrix and then it's features
#     '''
#     glcm = greycomatrix(input, [1], [0], 256, normed=True)
#     cont = greycoprops(glcm, 'contrast')
#     diss = greycoprops(glcm, 'dissimilarity')
#     homo = greycoprops(glcm, 'homogeneity')
#     eng = greycoprops(glcm, 'energy')
#     corr = greycoprops(glcm, 'correlation')
#     ASM = greycoprops(glcm, 'ASM')
#     return [cont, diss, homo, eng, corr, ASM]
#
# def save_array_as_nifty_volume(data, filename, reference_name = None):
#     '''
#     :param data: np array
#     :param filename: path + filename want to save
#     '''
#     if data.ndim == 3:
#         transposed_data = np.transpose(data, [2,1,0])
#     if data.ndim == 2:
#         transposed_data = np.transpose(data, [1,0])
#     img = sitk.GetImageFromArray(transposed_data)
#     if(reference_name is not None):
#         img_ref = sitk.ReadImage(reference_name)
#         img.CopyInformation(img_ref)
#     sitk.WriteImage(img, filename)
#
# def save_array_as_nifty_volume2(data, filename):
#     '''
#     :param data: np array
#     :param filename: path + filename want to save
#     '''
#     img = nibabel.Nifti1Image(data, affine=np.eye(4))
#     nibabel.save(img, filename)
#
# def get_array_as_nifty_volume(data):
#     nii = nibabel.Nifti1Image(data, affine=np.eye(4))
#     return nii
#
# def add_values_to_key(target_list):
#     '''
#     marge several dictionaries with same key in list to one dictionaries
#     :return:
#     '''
#     subdict = {}
#     for i in range(len(target_list)):
#         for k,v in target_list[i].items():
#             subdict.setdefault(k, [])
#             subdict[k].append(v)
#     for k,v in subdict.items():
#         subdict[k] = tuple(subdict[k])
#     return subdict

# def drop_col_contains_sth(dataframe,sth):
#     '''
#     drop columns contains certain strings(sth) in the dataframe
#     :param sth:  str 'sth'
#     :return:
#     '''
#     dropped_df = dataframe[dataframe.columns.drop(list(dataframe.filter(regex=sth)))]
#     return dropped_df
#
# def get_path_list(data_path):
#     '''
#     :return: all folder(or file list)
#     '''
#     id_list = []
#     for path in data_path:
#         path_list = tl.files.load_folder_list(path)
#         id_list += [os.path.join(path, os.path.basename(p), os.path.basename(p)) for p in path_list]
#     return id_list

def extract_patches_from_batch(imgs, patch_shape, stride):
    # simple version of sklearn.feature_extraction.image.extract_patches

    # if input imgs are not multiple imgs(just one img), then add axis=0 to make shape like [batch_size, w, h, ...]
    if imgs.ndim == 2 or (imgs.ndim == 3 and len(patch_shape) == 3):
        imgs = np.expand_dims(imgs, axis=0)

    patch_shape = (len(imgs),) + patch_shape
    patch_transpose = (3,0,1,2,4,5) if len(patch_shape) == 3 else (4,0,1,2,3,5,6,7)
    patch_reshape = (-1,) + patch_shape[1:]
    patch = image.extract_patches(imgs, patch_shape, extraction_step=stride)

    return patch.transpose(patch_transpose).reshape(patch_reshape)

def reconstruct_from_patches_nd(patches, image_shape, stride):
    # modified version of sklearn.feature_extraction.image.reconstruct_from_patches_2d
    # It can make only one image
    i_h, i_w = image_shape[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_shape)
    img_overlapped = np.zeros(image_shape)

    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1

    for p, (i, j) in zip(patches, product(range(0,n_h,stride), range(0,n_w,stride))):
        if patches.ndim == 3:
            img[i:i + p_h, j:j + p_w] += p
            img_overlapped[i:i + p_h, j:j + p_w] += 1
        elif patches.ndim == 4:
            print(np.shape(img))
            img[i:i + p_h, j:j + p_w,:] += p
            img_overlapped[i:i + p_h, j:j + p_w,:] += 1
    img /= img_overlapped

    return img

def discard_patch_idx(input, cut_line):
    n_non_zero = np.count_nonzero(input, axis=tuple(i for i in range(input.ndim) if not i == 0)) / np.prod(input.shape[1:])
    return np.where(n_non_zero >= cut_line)

#############################################################################################################################
#                                               Histogram Match                                                             #
#############################################################################################################################

# def cal_hm_landmark(arr, max_percent = 99.8, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
#     if arr.ndim > 1:
#         arr = arr.ravel()
#     arr_hist_sd, arr_edges_sd = np.histogram(arr, bins = range(np.max(arr)+2))
#     arr_mean = int(np.mean(arr))
#     black_peak = np.argmax(arr_hist_sd[:arr_mean])
#     white_peak = arr_mean + np.argmax(arr_hist_sd[arr_mean:])
#     # valley = arr_hist_sd[int(black_peak):int(white_peak)]
#     # consider only points over 10
#     # over_cutline = np.where(valley > 10)
#     # find local minmums among 500 points
#     # local_mins = argrelextrema(valley[over_cutline], np.less, order=250)
#     # take first local minimum
#     # local_min = over_cutline[0][local_mins[0][0]]
#     # local_min = argrelextrema(valley, np.less, order=250)[0][np.where(arr_hist_sd[argrelextrema(valley, np.less, order=250)] > 10)[0][0]]
#
#     threshold_dict = {}     # 'fuzzy', 'mean', 'median', 'valley'
#     threshold_dict['fuzzy'] = int((black_peak + white_peak) / 2)
#     threshold_dict['mean'] = arr_mean
#     threshold_dict['median'] = int(np.median(arr))
#     # threshold_dict['valley'] = black_peak + local_min
#
#     pc1 = threshold_dict[threshold]
#     pc2 = np.percentile(arr, max_percent)
#     ioi = arr[np.where((arr>=pc1) * (arr<=pc2))]
#     landmark_list = [np.percentile(ioi, i * (100/n_divide) ) for i in range(n_divide) if not i == 0]
#     landmark_list = [pc1] + landmark_list + [pc2]
#
#     if standard:
#         std_scale = (scale / pc2)
#         landmark_list = [landmark * std_scale for landmark in landmark_list]
#
#     return [int(landmark) for landmark in landmark_list]
#
# def get_mean_lm(input, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
#     landmarks_list = []
#     for i in input:
#         print('i.shape : ', i.shape)
#         landmark_list = cal_hm_landmark(i,threshold=threshold, n_divide = n_divide, standard=standard, scale=scale)
#         landmarks_list.append(landmark_list)
#     return landmarks_list, np.mean(landmarks_list, axis=0).astype(int)
#
# def hm_rescale(arr, input_landmark_list, standard_landmark_list):
#     arr_shape = arr.shape
#     if arr.ndim > 1:
#         arr = arr.ravel()
#     arr_copy = np.zeros_like(arr)
#
#     scale_idx = np.where((arr < input_landmark_list[0]))
#
#     # 0 ~ pc1 rescale
#     arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
#                                             in_range=(input_landmark_list[0] - 1, input_landmark_list[0]),
#                                             out_range=(standard_landmark_list[0] - 1, standard_landmark_list[0]))
#     # pc1 ~ m25 ~ m50 ~ m75 ~ pc2 rescale
#     for idx in range(len(input_landmark_list) - 1):
#
#         scale_idx = np.where((arr >= input_landmark_list[idx]) * (arr < input_landmark_list[idx+1]))
#         arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
#                                                 in_range=(input_landmark_list[idx], input_landmark_list[idx+1]),
#                                                 out_range=(standard_landmark_list[idx], standard_landmark_list[idx+1]))
#     # pc2 ~ max rescale
#     scale_idx = np.where((arr >= input_landmark_list[-1]))
#     arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
#                                                 in_range=(input_landmark_list[-1], input_landmark_list[-1] + 1),
#                                                 out_range=(standard_landmark_list[-1], standard_landmark_list[-1] + 1))
#
#
#
#     arr_copy = np.clip(arr_copy, a_min=standard_landmark_list[0], a_max=standard_landmark_list[-1])
#
#     return arr_copy.reshape(arr_shape)
#
# def batch_histogram_match(input, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
#     print('type(input) : ', type(input))
#     output = input
#     print('input.shape : ', input.shape)
#     print('>>> before g m l')
#     landmarks_list, stadard_list = get_mean_lm(input,threshold=threshold, n_divide = n_divide, standard=standard, scale=scale)
#     print('>>> passed get_mean_lm')
#     for idx in range(len(input)):
#         batch_match = hm_rescale(input[idx], landmarks_list[idx],stadard_list)
#         output[idx] = batch_match
#     output = tf.convert_to_tensor(output)
#     return output
#
# def std_list_loop(threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
#     def _cond(i,input):
#         return tf.less(i,input.shape[0])
#     def _body(i,input):
#         landmark_list = cal_hm_landmark(input[i],threshold=threshold, n_divide = n_divide, standard=standard, scale=scale)
#     pass

#############################################################################################################################
#                                        Tensor Histogram Match                                                             #
#############################################################################################################################

def cal_hm_landmark_tensor(arr, max_percent = 99.8, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
    if arr.get_shape().ndims > 1:
        arr = tf.reshape(arr, [-1])
    # if tf.errors.InvalidArgumentError:
    #     landmark_list = [[0] for _ in range(n_divide + 1)]
    def m1(arr):
        landmark_list = [tf.reduce_min(arr) for _ in range(n_divide + 1)]
        return landmark_list
    def m2(arr):
        hist = tf.histogram_fixed_width(arr, value_range = [0,tf.reduce_max(arr)], nbins = tf.cast(tf.reduce_max(arr),tf.int32))
        arr_mean = tf.cast(tf.reduce_mean(arr),tf.int32)
        if tf.errors.InvalidArgumentError:
            black_peak = tf.cast(0, tf.int32)
            white_peak = arr_mean + tf.cast(tf.argmax(hist[arr_mean:]), tf.int32)
        else :
            print('arr_mean : ', arr_mean)
            print('hist[:arr_mean] : ', hist[:arr_mean])
            black_peak = tf.cast(tf.argmax(hist[:arr_mean]),tf.int32)
            white_peak = arr_mean + tf.cast(tf.argmax(hist[arr_mean:]),tf.int32)
        valley = hist[black_peak:white_peak]
        # consider only points over 10
        # over_cutline = tf.where(valley > 10)
        # print('over_cutline : ', sess.run(over_cutline))
        # find local minmums among 500 points
        # local_mins = argrelextrema(valley[over_cutline], np.less, order=250)
        # take first local minimum
        # local_min = over_cutline[0][local_mins[0][0]]
        # local_min = argrelextrema(valley, np.less, order=250)[0][np.where(arr_hist_sd[argrelextrema(valley, np.less, order=250)] > 10)[0][0]]
        black_peak_val = tf.log(tf.cast(hist[black_peak],tf.float32))
        white_peak_val = tf.log(tf.cast(hist[white_peak],tf.float32))
        fuzzy_log = ((tf.cast(black_peak,tf.float32) * black_peak_val) + (tf.cast(white_peak,tf.float32) * white_peak_val)) / (black_peak_val + white_peak_val)


        threshold_dict = {}     # 'fuzzy', 'mean', 'median', 'valley', 'fuzzy_log'
        threshold_dict['fuzzy_log'] = fuzzy_log
        threshold_dict['fuzzy'] = tf.cast(((black_peak + white_peak) // 2),tf.int32)
        threshold_dict['mean'] = arr_mean
        threshold_dict['median'] = tf.contrib.distributions.percentile(arr, 50.0)
        # threshold_dict['valley'] = black_peak + local_min

        pc1 = tf.cast(threshold_dict[threshold], tf.float32)
        pc2 = tf.contrib.distributions.percentile(arr, max_percent)
        pc1 = tf.cond(tf.cast(pc1,tf.float32) > pc2, lambda: tf.contrib.distributions.percentile(arr, max_percent - 20),lambda: pc1)
        ioi = tf.gather(arr, tf.where(tf.logical_and(arr >= pc1,arr <= pc2)))
        landmark_list = [tf.contrib.distributions.percentile(ioi, i * (100 / n_divide)) for i in range(n_divide) if not i == 0]
        landmark_list = [pc1] + landmark_list + [pc2]

        if standard:
            std_scale = (scale / pc2)
            landmark_list = [landmark * std_scale for landmark in landmark_list]
        return landmark_list
    landmark_list = tf.cond(tf.equal(tf.cast(tf.reduce_max(arr), tf.int32),0), lambda: m1(arr), lambda: m2(arr))

    return [tf.cast(landmark, tf.int32) for landmark in landmark_list]

def rescale_tensor(arr, in_min,in_max, out_min,out_max):

    scaled_arr = (arr - tf.cast(in_min,tf.float32)) / tf.cast((in_max - in_min),tf.float32)
    # print('middle : ', sess.run(tf.reduce_min(scaled_arr)), sess.run(tf.reduce_max(scaled_arr)))
    scaled_arr = tf.cast(scaled_arr,tf.float32) * tf.cast((out_max - out_min),tf.float32) + tf.cast(out_min,tf.float32)

    return scaled_arr

def hm_rescale_tensor(arr, input_landmark_list, standard_landmark_list,first=True):
    # arr_shape = arr.shape
    arr_shape = tf.shape(arr)
    if arr.get_shape().ndims > 1:
        arr = tf.reshape(arr, [-1])
    arr_copy = tf.zeros_like(arr)

    scale_idx = tf.where(tf.cast(arr,tf.int32) < input_landmark_list[0])
    # 0 ~ pc1 rescale
    scaled_arr = rescale_tensor(tf.gather(arr, scale_idx), tf.cast(tf.reduce_min(tf.gather(arr, scale_idx)),tf.int32), input_landmark_list[0],
                                standard_landmark_list[0]-1, standard_landmark_list[0])
    # scaled_arr = tf.reshape(scaled_arr, [-1])
    with tf.variable_scope('rescale', reuse=True, dtype=tf.float32):
        if first :
            arr_copy_var = tf.Variable(arr_copy, validate_shape=False,name='arr_copy',trainable=False)
            tf.add_to_collection("arr_copy_var", arr_copy_var)
        if first == False:
            arr_copy_var = tf.get_collection('arr_copy_var')[0]

    update = tf.scatter_nd_update(arr_copy_var,
                                  scale_idx,
                                  tf.reshape(scaled_arr, [-1]))
    # pc1 ~ m25 ~ m50 ~ m75 ~ pc2 rescale
    for idx in range(len(input_landmark_list) - 1):
        scale_idx = tf.where(tf.logical_and(tf.cast(arr,tf.int32) >= input_landmark_list[idx], tf.cast(arr,tf.int32) < input_landmark_list[idx+1]))
        scaled_arr = rescale_tensor(tf.gather(arr, scale_idx), input_landmark_list[idx], input_landmark_list[idx+1],standard_landmark_list[idx], standard_landmark_list[idx+1])
        # scaled_arr = tf.reshape(scaled_arr, [-1])
        update = tf.scatter_nd_update(update,
                                      scale_idx,
                                      tf.reshape(scaled_arr, [-1]))
    # pc2 ~ max rescale
    scale_idx = tf.where((tf.cast(arr,tf.int32) >= input_landmark_list[-1]))
    scaled_arr = rescale_tensor(tf.gather(arr, scale_idx), input_landmark_list[-1], tf.cast(tf.reduce_max(tf.gather(arr, scale_idx)),tf.int32),
                                standard_landmark_list[-1], standard_landmark_list[-1] + 1)
    # scaled_arr = tf.reshape(scaled_arr, [-1])

    # print(zeros)
    update = tf.scatter_nd_update(update,
                                  scale_idx,
                                  tf.reshape(scaled_arr, [-1]))

    update = tf.clip_by_value(update, tf.cast(standard_landmark_list[0], tf.float32),
                              tf.cast(standard_landmark_list[-1], tf.float32))
    # tf.cond(tf.size(test_copy) > tf.size(test), lambda: f1(update,arr_copy),lambda : print('need to set first = True'))
    def f1(update,arr_copy):
        update = update[:tf.size(arr_copy)]
        return update
    def f2(update):
        return update
    update = tf.cond(tf.greater(tf.size(update), tf.size(arr_copy)), lambda: f1(update, arr_copy), lambda: f2(update))
    update = update[:tf.size(arr_copy)]
    update = tf.reshape(update,arr_shape)
    print('rescale update : ', update)
    return update

def batch_histogram_match_tensor(input, batch_size, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1,first=True):
    # output = tf.zeros_like(input)
    # print('output shape : ', output.get_shape())
    landmarks_list = []
    print('>>> get landmark')
    for i in range(batch_size):
        landmark_list = cal_hm_landmark_tensor(input[i],threshold=threshold, n_divide = n_divide, standard=standard, scale=scale)
        landmarks_list.append(landmark_list)
    stadard_list = tf.reduce_mean(landmarks_list, axis=0)
    # update = tf.Variable(output,validate_shape=False)
    # print('update shape : ', tf.shape(update))
    print('>>> start rescale')
    batch_match = hm_rescale_tensor(input[0], landmarks_list[0], stadard_list,first)
    zeros = tf.zeros_like([batch_match])
    zeros = tf.concat([zeros, [batch_match]], 0)
    for idx in range(1,batch_size):
        batch_match = hm_rescale_tensor(input[idx], landmarks_list[idx],stadard_list,first)
        zeros = tf.concat([zeros, [batch_match]], 0)
        # batch_match = tf.expand_dims(batch_match, 0)
        # update = tf.scatter_nd_update(update,
        #                               [idx],
        #                               batch_match)

    update = zeros[1:]
    return update

def batch_histogram_match_tensor_undefined(input, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1,first=True):

    landmarks_list = []
    print('>>> get landmark')
    batch_size = tf.shape(input)[0]
    i = tf.constant(0)
    def _cond(i, batches):
        return tf.less(i, tf.cast(batch_size, tf.int32))

    def _body(i, batches):
        landmark_list = cal_hm_landmark_tensor(batches[i], threshold=threshold, n_divide=n_divide, standard=standard,scale=scale)
        landmarks_list.append(landmark_list)
        i = tf.add(i, 1)
        return landmarks_list

    landmarks_list = tf.while_loop(_cond, _body, [i, input])

    stadard_list = tf.reduce_mean(landmarks_list, axis=0)
    # update = tf.Variable(output,validate_shape=False)
    # print('update shape : ', tf.shape(update))
    print('>>> start rescale')
    batch_match = hm_rescale_tensor(input[0], landmarks_list[0], stadard_list,first)
    zeros = [batch_match]
    # zeros = tf.concat([zeros, [batch_match]], 0)
    idx = tf.constant(0)
    def _cond1(idx, batches):
        return tf.less(idx, tf.cast(batch_size, tf.int32))

    def _body2(idx, batches):
        batch_match = hm_rescale_tensor(batches[idx], landmarks_list[idx], stadard_list, first)
        zeros = tf.concat([zeros, [batch_match]], 0)
        idx = tf.add(idx, 1)
        return idx, zeros

    _, update = tf.while_loop(_cond1, _body2, [idx, input],
                              shape_invariants=[idx.get_shape(), tf.TensorShape([None, None,None,None])])

        # batch_match = tf.expand_dims(batch_match, 0)
        # update = tf.scatter_nd_update(update,
        #                               [idx],
        #                               batch_match)

    # update = zeros[1:]
    return update

def filter_small_object(masks, cutline):
   '''

   :param mask: [B, H, W, 1] - one-hot mask image
   :param cutline: cutline for object size
   :return: [B, H, W, 1] - one-hot filtered mask image
   '''
   filtered_masks = []
   for mask in masks:
       _, markers, states,_ = cv2.connectedComponentsWithStats(mask)
       filter_idx = np.where(states[:, 4] < cutline)[0]

       for idx in filter_idx:
           markers[markers == idx] = 0

       markers[markers != 0] = 255
       markers = np.expand_dims(markers, axis=-1)
       filtered_masks.append(markers)

   filtered_masks = np.asarray(filtered_masks, dtype=np.uint8)
   return filtered_masks


def augmentation(flip, n_child, rot_angle, multiply, super_px_rate, n_super_px, aug_drop_rate, crop_rate,
                 sharpen_alpha, sharpen_lightness, emboss_alpha,emboss_strength,edge_alpha, add, add_gauss):
    if flip == True:
    # augmentation rate : 0.5 (Noop or Aug)
        aug = iaa.SomeOf(n_child, [
        iaa.Affine(rotate=rot_angle),
        iaa.CropAndPad(percent= (0,.2),
                      pad_mode = ['constant', 'edge'],
                      pad_cval = 0),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Multiply((0.5, 1.5)),
        iaa.Dropout(p=(0, 0.2), per_channel=0.5),
        iaa.Superpixels(p_replace=(0.1,0.9), n_segments=(0, 100)),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.AverageBlur(k=(2, 11)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
        iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
        iaa.EdgeDetect(alpha=(0.0, 1.0)),
        iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0)),
        iaa.Add((-40, 40)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
        ])
    else:
        aug = iaa.SomeOf(n_child, [
        iaa.Affine(rotate=rot_angle),
        iaa.CropAndPad(percent= crop_rate,
                      pad_mode = 'edge'),
        iaa.Multiply(multiply),
        iaa.Dropout(p=aug_drop_rate, per_channel=0.5),
        iaa.Superpixels(p_replace=super_px_rate, n_segments=n_super_px),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.Sharpen(alpha=sharpen_alpha, lightness=sharpen_lightness),
        iaa.Emboss(alpha=emboss_alpha, strength=emboss_strength),
        iaa.EdgeDetect(alpha=edge_alpha),
        iaa.Add(add),
        iaa.AdditiveGaussianNoise(scale=add_gauss)
        ])
    return aug

## Augmentation Option
# APPLY_OPTION_RANGE = (0, None)
# n_child = random.randrange(0, None)
# rot_angle = random.randrange(-45,45)
# FLIPUD_RATE = 0.5
# n_super_px = random.randrange(0, 100)
# flip = False
# FLIPLR_RATE = 0.5
# MULTIPLY_RATE = round(random.uniform(0.5, 1.5), 2)
# super_px_rate = round(random.uniform(0.1,0.9), 2)
# drop_rate = round(random.uniform(0,0.2), 2)
# crop_rate = round(random.uniform(0,0.2), 2)
# sharpen_alpha = round(random.uniform(0.0, 1.0), 2)
# sharpen_lightness = round(random.uniform(0.75, 2.0), 2)
# emboss_alpha = round(random.uniform(0.0, 1.0), 2)
# emboss_strength = round(random.uniform(0.5, 1.5), 2)
# edge_alpha = round(random.uniform(0.0, 1.0), 2)
# add = random.randrange(-40,40)
# add_gauss = round(random.uniform(0, 0.05 * 255), 2)
# CROP_PERCENTAGE = (0,.2)
# CROP_AND_PAD_RATE = (-0.15, 0.15)
# ROTATE_ANGLE = (-45, 45)
# AUG_DROPOUT_RATE = (0, 0.2)