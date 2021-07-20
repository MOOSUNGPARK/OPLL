import numpy as np
import pydicom as dicom
import sklearn
import sklearn.feature_extraction
from itertools import product
import cv2


def normalization(img):
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img


def standardization(img):
    '''
    for normally distributted img
    '''
    std_img =  (img - img.mean()) / img.std()
    return std_img


def val_input_loader(data_path):
    print(data_path)
    dic = dicom.read_file(data_path)
    img = dic.pixel_array
    shape = np.squeeze(img.shape)
    if img.max() > np.unique(img)[-2] * 1.3:
        idx = np.where(img == img.max())
        copy = img.copy()
        copy[idx] = np.unique(img)[-2] * 1.1
        img = copy
    if img.ndim == 3:
        # img = np.transpose(img, (1, 0, 2))
        img = np.reshape(img, (np.shape(img)[0], np.shape(img)[1]))
    # if img.ndim == 2:
    # img = np.transpose(img, (1, 0))
    img = cv2.resize(img, dsize=(480, 576))
    img = normalization(img) * 255
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)

    return img, shape


def extract_patches_from_batch(imgs, patch_shape, stride):
    # simple version of sklearn.feature_extraction.image.extract_patches
    # image shape 이 [n,192,160,4] 인 경우 patch_shape=[64,64,4]를 넣어주면 됨

    # if input imgs are not multiple imgs(just one img), then add axis=0 to make shape like [batch_size, w, h, ...]
    # print('imgs.ndim : ', imgs.ndim)
    # print('imgs.shape : ', imgs.shape)
    if imgs.ndim == 2 or (imgs.ndim == 3 and len(patch_shape) == 3):
        imgs = np.expand_dims(imgs, axis=0)
    # print('imgs.shape : ', imgs.shape)
    patch_shape = (len(imgs),) + patch_shape
    # print('patch_shape : ', patch_shape)
    patch_transpose = (3,0,1,2,4,5) if len(patch_shape) == 3 else (4,0,1,2,3,5,6,7)
    patch_reshape = (-1,) + patch_shape[1:]
    patch = sklearn.feature_extraction.image.extract_patches(imgs, patch_shape, extraction_step=stride)
    # print('patch.shape : ', patch.shape)

    return patch.transpose(patch_transpose).reshape(patch_reshape)


def reconstruct_from_patches_nd(patches, image_shape, stride):
    # modified version of sklearn.feature_extraction.image.reconstruct_from_patches_2d
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
            # print(np.shape(img))
            img[i:i + p_h, j:j + p_w,:] += p
            img_overlapped[i:i + p_h, j:j + p_w,:] += 1
    img /= img_overlapped

    return img

