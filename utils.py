"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import copy
import glob
import os

import h5py
import numpy as np
import scipy.misc
import scipy.ndimage
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def down_upscale(image, scale):
    res = copy.deepcopy(image)
    for channel in range(res.shape[2]):
        tmp = res[:, :, channel]
        tmp = scipy.ndimage.interpolation.zoom(tmp, 1.0 / scale, prefilter=False)
        tmp = scipy.ndimage.interpolation.zoom(tmp, scale, prefilter=False)
        res[:, :, channel] = tmp
    return res


def upscale(image, scale):
    res = np.zeros([image.shape[0] * scale, image.shape[1] * scale, image.shape[2]])
    for channel in range(res.shape[2]):
        res[:, :, channel] = scipy.ndimage.interpolation.zoom(image[:, :, channel], scale, prefilter=False)
    return res


def down_upscale_new(image, scale):
    res = scipy.misc.imresize(image, 1.0 / scale, interp='bicubic')
    res = scipy.misc.imresize(res, 1.0 * scale, interp='bicubic')
    return res


def preprocess(path, scale, is_grayscale, is_RGB):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation

    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    image = imread(path, is_grayscale=is_grayscale, is_RGB=is_RGB)
    label_ = modcrop(image, scale)

    # Must be normalized
    image = image / 255.
    label_ = label_ / 255.

    input_ = down_upscale(label_, scale)

    return input_, label_


def prepare_data(sess, dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        data_dir = os.path.join(os.sep, os.path.join(os.getcwd(), dataset), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))

    return data


def make_data(checkpoint_dir, data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), checkpoint_dir, 'train.h5')
    else:
        savepath = os.path.join(os.getcwd(), checkpoint_dir, 'test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


def imread(path, is_grayscale, is_RGB):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        res = scipy.misc.imread(path, flatten=True, mode='RGB' if is_RGB else 'YCbCr').astype(np.float)
        # Make one channel image
        res = res.reshape(res.shape[0], res.shape[1], 1)
    else:
        res = scipy.misc.imread(path, mode='RGB' if is_RGB else 'YCbCr').astype(np.float)

    return res


def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def read_image(input_image, config):
    """
    Read one image file
    """
    data = [input_image]

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6

    image = imread(input_image, is_grayscale=config.c_dim == 1, is_RGB=config.is_RGB)
    image = image / 255.
    input_ = upscale(image, config.scale)

    h, w, _ = input_.shape
    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0
    for x in range(0, h - config.image_size + 1, config.stride):
        nx += 1;
        ny = 0
        for y in range(0, w - config.image_size + 1, config.stride):
            ny += 1
            sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]

            sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])

            sub_input_sequence.append(sub_input)

    return np.asarray(sub_input_sequence), nx, ny


def input_setup(sess, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path
    if config.is_train:
        data = prepare_data(sess, dataset="Train")
    else:
        data = prepare_data(sess, dataset="Test")
        print data

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6

    if config.is_train:
        for i in xrange(len(data)):
            input_, label_ = preprocess(data[i], config.scale, config.c_dim == 1, config.is_RGB)

            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape

            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                    sub_label = label_[x + padding:x + padding + config.label_size,
                                y + padding:y + padding + config.label_size]  # [21 x 21]

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

    else:

        input_data = None
        for item in data:
            if 'butterfly' in item:
                input_data = item
                break

        input_, label_ = preprocess(input_data, config.scale, config.c_dim == 1, config.is_RGB)
        print input_data
        print input_.shape

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape

        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        nx = ny = 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1;
            ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                sub_label = label_[x + padding:x + padding + config.label_size,
                            y + padding:y + padding + config.label_size]  # [21 x 21]

                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                sub_label = sub_label.reshape([config.label_size, config.label_size, config.c_dim])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]

    make_data(config.checkpoint_dir, arrdata, arrlabel)

    if not config.is_train:
        return nx, ny


def imsave(image, path, is_RGB):
    if image.shape[2] == 3:
        image = scipy.misc.toimage(image, mode='RGB' if is_RGB else 'YCbCr').convert('RGB')
    elif image.shape[2] == 1:
        image = image[:, :, 0]
    else:
        image = None

    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w, dim = images.shape[1], images.shape[2], images.shape[3]
    img = np.zeros((h * size[0], w * size[1], dim))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img
