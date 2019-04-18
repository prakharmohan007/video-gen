import os
import tensorflow as tf
from tkinter import *
import numpy as np
import cv2
from model import VideoGAN


class Test:
    def __init__(self):
        flags = tf.app.flags
        flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
        flags.DEFINE_integer("zdim", 100, "The dimension of latent vector")
        flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
        flags.DEFINE_string("checkpoint_file", "trafic/VideoGAN_8_97.ckpt", "The checkpoint file name")
        flags.DEFINE_float("lambd", 0.0, "The value of sparsity regularizer")
        FLAGS = flags.FLAGS
        video_dims = [32, 64, 64, 3]

        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        if not os.path.exists("./genvideos"):
            os.makedirs("./genvideos")

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        with tf.Session(config=run_config) as sess:
            self.videogan = VideoGAN(sess, video_dim=video_dims, zdim=FLAGS.zdim, batch_size=FLAGS.batch_size,
                                     epochs=FLAGS.epoch, checkpoint_file=FLAGS.checkpoint_file, lambd=FLAGS.lambd)
            self.videogan.build_model()

            master = Tk()
            master.title('GUI Video Generation')

            Label(master, text="Video Generation using GAN", fg="green",
                  font=("", 15, "bold")).grid(row=1, column=1)

            sbutton = Button(master, text="Get Video", command=self.genVideo)
            sbutton.grid(row=3, column=1)

            Label(master, text="press button to generate video", fg="green",
                  font=("", 15)).grid(row=2, column=1)

            ebutton = Button(master, text="Exit", command=self.exitWindow)
            ebutton.grid(row=9, column=1)

            master.title('GUI Video Generation')
            master.resizable(0, 0)
            master.mainloop()

    def genVideo(self):
        print("generating video")
        video = self.videogan.test()
        print(video.shape)
        print("video generated")
        # c = Canvas(self.master, width=64, height=64, bg='black', bd=1, relief=RAISED)
        # c.grid(row=7, column=1)

        vid = video[0, :, :, :, :]
        # vid = (vid + 1) * 127.5
        print(vid.shape)
        for j in range(vid.shape[0]):
            frame = vid[j, :, :, :]
            frame = (frame + 1) * 127.5
            print(frame.shape)
            # print(frame.astype(np.uint8))
            # im = Image.fromarray(frame.astype(np.uint8))
            im = frame.astype(np.uint8)
            new_image_red = im[:, :, 2]
            new_image_green = im[:, :, 1]
            new_image_blue = im[:, :, 0]
            new_rgb = np.dstack([new_image_red, new_image_green, new_image_blue])
            # c.pack(fill=BOTH, expand=TRUE)
            cv2.imshow("GAN generated short video", new_rgb)
            cv2.waitKey(30)

    def exitWindow(self):
        exit()


if __name__ == '__main__':
    Test()
