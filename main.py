import os
import pprint

import tensorflow as tf

from model import SRCNN

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 2, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_boolean("is_RGB", False, "True for load images in RGB, False for YCbCr [False]")
flags.DEFINE_integer("net_size_factor", 1, "Factor for scale network layers neurons from default size")
flags.DEFINE_string("input_image", None, "Path for input image for upscale")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        srcnn = SRCNN(sess,
                      image_size=FLAGS.image_size,
                      label_size=FLAGS.label_size,
                      batch_size=FLAGS.batch_size,
                      c_dim=FLAGS.c_dim,
                      net_size_factor=FLAGS.net_size_factor,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)

        if FLAGS.input_image:
            srcnn.scale_image(FLAGS)
        else:
            srcnn.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
