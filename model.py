import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import glob
from utils import *
from tensorflow.python.tools import inspect_checkpoint as chkp


class VideoGAN():
    def __init__(self, sess, video_dim, zdim, batch_size, epochs, checkpoint_file, lambd):
        self.bv1 = batch_norm(name="genb1")
        self.bv2 = batch_norm(name="genb2")
        self.bv3 = batch_norm(name="genb3")
        self.bv4 = batch_norm(name="genb4")
        self.bs1 = batch_norm(name="genb5")
        self.bs2 = batch_norm(name="genb6")
        self.bs3 = batch_norm(name="genb7")
        self.bs4 = batch_norm(name="genb8")
        self.bd1 = batch_norm(name="dis1")
        self.bd2 = batch_norm(name="dis2")
        self.bd3 = batch_norm(name="dis3")
        self.video_dim = video_dim
        self.zdim = zdim
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_file = checkpoint_file
        self.lambd = lambd
        self.sess = sess

    def summary(self):
        # model_vars = tf.trainable_variables()
        print("*********GENERATOR SUMMARY*********")
        slim.model_analyzer.analyze_vars(self.gen_var, print_info=True)
        print("**********DISCRIMINATOR SUMMARY*********")
        slim.model_analyzer.analyze_vars(self.dis_var, print_info=True)
        print("************************************")

    def build_model(self):
        # initialize noise and video placeholders
        self.z = tf.placeholder(tf.float32, [None, self.zdim])
        self.zsample = tf.placeholder(tf.float32, [None, self.zdim])
        self.real_video = tf.placeholder(tf.float32, [None] +self. video_dim)

        print("Build Generator")
        self.fake_video, self.foreground, self.background, self.mask = self.generator(self.z, False)

        print("Generator Build. Building visualize_videos with same variables")
        self.genvideo, self.bg = self.visualize_videos()
        print("visualize_videos done")

        print("Building discriminator for real videos")
        prob_real, logits_real = self.discriminator(self.real_video)
        print("Building discriminator for fake videos")
        prob_fake, logits_fake = self.discriminator(self.fake_video, reuse = True)

        # define cost functions
        d_real_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = tf.ones_like(prob_real)))
        d_fake_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.zeros_like(prob_fake)))
        self.g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(prob_fake))) + self.lambd*tf.norm(self.mask, 1)
        self.d_cost = d_real_cost + d_fake_cost

        # get trainable variables and apply optimizer
        self.gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        self.dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        self.g_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.05).minimize(self.g_cost, var_list=self.gen_var)
        self.d_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.05).minimize(self.d_cost, var_list=self.dis_var)

        # initialize weights
        self.saver = tf.train.Saver()
        if self.checkpoint_file == "None":
            self.ckpt_file = None

        # if checkpoint file provided, load it else initialize from scratch
        if self.checkpoint_file:
            saver_ = tf.train.import_meta_graph('./checkpoints/' + self.checkpoint_file + '.meta')
            # saver_.restore(self.sess, tf.train.latest_checkpoint('./checkpoints/'))
            saver_.restore(self.sess, './checkpoints/' + self.checkpoint_file)
            print("Restored model")
        else:
            tf.global_variables_initializer().run()

        # print model summary
        # self.summary()

    def train(self):
        visualize_count = 1

        # prepare data
        data_files = glob.glob("./trainvideos/*")
        video_data = load_data(data_files, 32)
        print("Video Data loaded")
        print("Dimentions of dataset (num videos, num frames, frame size (height x width), num channels): ",
              video_data.shape)
        num_samples = video_data.shape[0]
        num_batches = num_samples / self.batch_size
        print(num_batches)
        print_count = num_samples / self.batch_size

        dis_loss = []
        gen_loss = []

        for epoch in range(self.epochs):
            print("##################### epoch ", epoch, " #######################")

            for iteration in range(int(num_batches)):

                print(".........Iteration.........:", iteration)

                # define noise to start with
                noise_sample = np.random.normal(-1, 1, size=[visualize_count, self.zdim]).astype(np.float32)
                noise = np.random.normal(-1, 1, size=[self.batch_size, self.zdim]).astype(np.float32)

                # sample batches
                indices = np.random.choice(num_samples, self.batch_size, False)
                videos = video_data[indices]

                # process_and_write_video(videos,"true_video" + str(counter))
                _, dloss = self.sess.run([self.d_opt, self.d_cost], feed_dict = {self.z : noise, self.real_video: videos})
                _, gloss = self.sess.run([self.g_opt, self.g_cost], feed_dict = {self.z : noise, self.real_video: videos})
                # _, gloss = self.sess.run([self.g_opt, self.g_cost], feed_dict = {self.z : noise, self.real_video: videos})
                print("Discriminator Loss: ", dloss)
                dis_loss.append(dloss)
                print("Generator Loss", gloss)
                gen_loss.append(gloss)

        # if np.mod(iteration + 1, print_count) == 0:
            noise_sample_gen = np.random.normal(-1, 1, size=[visualize_count, self.zdim]).astype(np.float32)
            gen_videos, bg = self.sess.run([self.genvideo, self.bg], feed_dict={self.zsample: noise_sample_gen})
            process_and_write_video(gen_videos, "video" + str(epoch))
            process_and_write_image(bg, "bg" + str(epoch))
            print(".....Writing sample generated videos......")

            self.saver.save(self.sess, './checkpoints/VideoGAN_{}_{}.ckpt'.format(self.batch_size, epoch))
            print('Saved {}'.format(epoch))

        # plot losses
        plot_graph(dis_loss, gen_loss)

    def test(self):
        # print all tensors in checkpoint file
        # chkp.print_tensors_in_checkpoint_file('./checkpoints/'+self.checkpoint_file,
        #                                       tensor_name='generator/gen11/kernel', all_tensors=True)

        visualize_count = 1
        print("initialize noise")
        noise_sample_gen = np.random.normal(-1, 1, size=[visualize_count, self.zdim]).astype(np.float32)
        gen_videos, bg = self.sess.run([self.genvideo, self.bg], feed_dict={self.zsample: noise_sample_gen})
        # process_and_write_video(gen_videos, "video" + str(1234))
        # process_and_write_image(bg, "bg" + str(1234))
        # print(".....Writing sample generated videos......")
        return gen_videos

    def generator(self, z, reuse = False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            #Background
            z = tf.reshape(z, [-1, 1, 1, self.zdim])

            # layer1: conv
            deconvb1 = tf.layers.conv2d_transpose(z, 512, kernel_size=[4, 4], strides=[1, 1], name="gen1")
            deconvb1 = tf.nn.relu(self.bs1(deconvb1))

            # layer2: conv
            deconvb2 = tf.layers.conv2d_transpose(deconvb1, 256, kernel_size=[2, 2], strides=[2, 2],
                                                  padding="VALID", name="gen2")
            deconvb2 = tf.nn.relu(self.bs2(deconvb2))

            # layer3: conv
            deconvb3 = tf.layers.conv2d_transpose(deconvb2, 128, kernel_size=[2, 2], strides=[2, 2],
                                                  padding="VALID", name="gen3")
            deconvb3 = tf.nn.relu(self.bs3(deconvb3))

            # layer4: conv
            deconvb4 = tf.layers.conv2d_transpose(deconvb3, 64, kernel_size=[2, 2], strides=[2, 2],
                                                  padding="VALID", name="gen4")
            deconvb4 = tf.nn.relu(self.bs4(deconvb4))

            # layer5: Conv for three channels
            deconvb5 = tf.layers.conv2d_transpose(deconvb4, 3, kernel_size=[2, 2], strides=[2, 2],
                                                  padding="VALID", name="gen5")
            background = tf.nn.tanh(deconvb5)

            #Foreground
            #z  = tf.expand_dims(z,1)
            z = tf.reshape(z, [-1, 1, 1, 1, self.zdim])
            deconv1 = tf.layers.conv3d_transpose(z,filters = 512,kernel_size = [2,4,4],strides = [1,1,1], use_bias = False,name="gen6")
            deconv1 = tf.nn.relu(self.bv1(deconv1))
            deconv2 = tf.layers.conv3d_transpose(deconv1,filters= 256,kernel_size=[4,4,4],strides=[2,2,2],padding = "SAME",use_bias = False,name="gen7")
            deconv2 = tf.nn.relu(self.bv2(deconv2))
            deconv3 = tf.layers.conv3d_transpose(deconv2,filters= 128,kernel_size =[4,4,4],strides = [2,2,2], padding ="SAME",use_bias = False,name="gen8")
            deconv3 = tf.nn.relu(self.bv3(deconv3))
            deconv4 = tf.layers.conv3d_transpose(deconv3,filters=64,kernel_size=[4,4,4],strides=[2,2,2],padding ="SAME",use_bias = False,name="gen9")
            deconv4 = tf.nn.relu(self.bv4(deconv4))

            #Mask
            mask = tf.layers.conv3d_transpose(deconv4,filters= 1, kernel_size=[4,4,4], strides =[2,2,2],padding ="SAME",use_bias = False,name="gen10")
            mask = tf.nn.sigmoid(mask)
            #Video
            foreground = tf.layers.conv3d_transpose(deconv4,filters = 3, kernel_size = [4,4,4], strides = [2,2,2], padding ="SAME",use_bias = False,name="gen11")
            foreground = tf.nn.tanh(foreground)

            #Replicate background and mask
            background = tf.expand_dims(background,1)
            backreplicate = tf.tile(background,[-1,32,1,1,1])
            maskreplicate = tf.tile(mask,[-1,1,1,1,3])
            #Incorporate mask
            video = tf.add(tf.multiply(mask, foreground), tf.multiply(1-mask,background))
            print("Video Shape")
            print(video.get_shape())
            return video, foreground, background, mask

    def discriminator(self, vid, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = tf.layers.conv3d(vid, 64, kernel_size=[4, 4, 4], strides=[2, 2, 2], padding="SAME",
                                     reuse=reuse, name="dis1")
            conv1 = lrelu(conv1)
            conv2 = tf.layers.conv3d(conv1, 128, kernel_size=[4, 4, 4], strides=[2, 2, 2],
                                     padding="SAME", reuse=reuse, name="dis2")
            conv2 = lrelu(self.bd1(conv2))
            conv3 = tf.layers.conv3d(conv2,256,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",reuse=reuse,name="dis3")
            conv3 = lrelu(self.bd2(conv3))
            conv4 = tf.layers.conv3d(conv3,512,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",reuse=reuse,name="dis4")
            conv4 = lrelu(self.bd3(conv4))
            conv5 = tf.layers.conv3d(conv4,1,kernel_size=[2,4,4],strides=[1,1,1],padding="VALID",reuse=reuse,name="dis5")
            conv5 = tf.reshape(conv5, [-1,1])
            conv5sigmoid = tf.nn.sigmoid(conv5)
            return conv5sigmoid, conv5

    def visualize_videos(self):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            #Background
            z = tf.reshape(self.zsample,[-1,1,1,self.zdim])
            deconvb1 = tf.layers.conv2d_transpose(z,512,kernel_size=[4,4],strides =[1,1],name="gen1")
            deconvb1 = tf.nn.relu(self.bs1(deconvb1))
            deconvb2 = tf.layers.conv2d_transpose(deconvb1,256,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen2")
            deconvb2 = tf.nn.relu(self.bs2(deconvb2))
            deconvb3 = tf.layers.conv2d_transpose(deconvb2,128,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen3")
            deconvb3 = tf.nn.relu(self.bs3(deconvb3))
            deconvb4 = tf.layers.conv2d_transpose(deconvb3,64,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen4")
            deconvb4 = tf.nn.relu(self.bs4(deconvb4))
            deconvb5 = tf.layers.conv2d_transpose(deconvb4,3,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen5")
            background = tf.nn.tanh(deconvb5)
            #Foreground
            #z  = tf.expand_dims(z,1)
            z = tf.reshape(z,[-1,1,1,1,self.zdim])
            deconv1 = tf.layers.conv3d_transpose(z,filters = 512,kernel_size = [2,4,4],strides = [1,1,1], use_bias = False,name="gen6")
            deconv1 = tf.nn.relu(self.bv1(deconv1))
            deconv2 = tf.layers.conv3d_transpose(deconv1,filters= 256,kernel_size=[4,4,4],strides=[2,2,2],padding = "SAME",use_bias = False,name="gen7")
            deconv2 = tf.nn.relu(self.bv2(deconv2))
            deconv3 = tf.layers.conv3d_transpose(deconv2,filters= 128,kernel_size =[4,4,4],strides = [2,2,2], padding ="SAME",use_bias = False,name="gen8")
            deconv3 = tf.nn.relu(self.bv3(deconv3))
            deconv4 = tf.layers.conv3d_transpose(deconv3,filters=64,kernel_size=[4,4,4],strides=[2,2,2],padding ="SAME",use_bias = False,name="gen9")
            deconv4 = tf.nn.relu(self.bv4(deconv4))

            #Mask
            mask = tf.layers.conv3d_transpose(deconv4,filters= 1, kernel_size=[4,4,4], strides =[2,2,2],padding ="SAME",use_bias = False,name="gen10")
            mask = tf.nn.sigmoid(mask)
            #Video
            foreground = tf.layers.conv3d_transpose(deconv4,filters = 3, kernel_size = [4,4,4], strides = [2,2,2], padding ="SAME",use_bias = False,name="gen11")
            foreground = tf.nn.tanh(foreground)
            #Replicate background and mask
            background = tf.expand_dims(background,1)
            backreplicate = tf.tile(background,[-1,32,1,1,1])
            maskreplicate = tf.tile(mask,[-1,1,1,1,3])
            #Incorporate mask
            video = tf.add(tf.multiply(mask, foreground), tf.multiply(1-mask, background))
            print("Video Shape")
            print(video.get_shape())
            return video, background
