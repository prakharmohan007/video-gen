import os
import tensorflow as tf

from model import VideoGAN


flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_integer("zdim", 100, "The dimension of latent vector")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_string("checkpoint_file", None, "The checkpoint file name")
flags.DEFINE_float("lambd", 0.0, "The value of sparsity regularizer")
FLAGS = flags.FLAGS
video_dim = [32, 64, 64, 3]


def main(_):
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./genvideos"):
        os.makedirs("./genvideos")
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        videogan = VideoGAN(sess, video_dim = video_dim, zdim = FLAGS.zdim, batch_size = FLAGS.batch_size,
                            epochs=FLAGS.epoch, checkpoint_file = FLAGS.checkpoint_file, lambd = FLAGS.lambd)
        videogan.build_model()
        videogan.train()


if __name__ == '__main__':
    tf.app.run()
