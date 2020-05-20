from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0

import time
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from monodepth_yolo_dataloader import Datasetloader
from core_yolo.utils import *
from monodepth_yolo import *
from core_yolo.config import cfg
from average_gradients import *



class monodepth_yoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END

        self.stage_epochs = cfg.TRAIN.STAGE_EPOCHS
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        self.batch_size     = cfg.TRAIN.BATCH_SIZE
        self.train_input_height  = cfg.TRAIN.INPUT_HEIGHT
        self.train_input_width   = cfg.TRAIN.INPUT_WIDTH
        self.encode         = cfg.MONODEPTH.ENCODER
        self.num_gpus       = cfg.MONODEPTH.GPU_NUMS
        self.monodepth_log_directory = cfg.MONODEPTH.LOG_DIRECTORY

        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150

        self.train_logdir = cfg.MONODEPTH.LOG_DIRECTORY
        self.trainset = Datasetloader('train')
        self.testset = Datasetloader('test')


        with tf.name_scope('define_input'):
            self.left_images = tf.placeholder(dtype=tf.float32, shape=[None,self.train_input_height, self.train_input_width,3], name='left_image')
            self.right_images = tf.placeholder(dtype=tf.float32,shape=[None,self.train_input_height, self.train_input_width,3], name='right_image')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable    = tf.placeholder(dtype=tf.bool,   name="training")

        self.global_step = tf.Variable(0,  trainable=False)

        self.steps_per_epoch = len(self.trainset)

        self.num_training_samples = self.steps_per_epoch * self.batch_size

        self.num_total_steps = self.stage_epochs * self.steps_per_epoch

        boundaries = [np.int32((3 / 5) * self.num_total_steps), np.int32((4 / 5) * self.num_total_steps)]
        values = [self.learn_rate_init, self.learn_rate_init / 2, self.learn_rate_init / 4]

        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)


        opt_step = tf.train.AdamOptimizer(learning_rate)
        print("total number of samples: {}".format(self.num_training_samples))
        print("total number of steps: {}".format(self.num_total_steps))

        tower_grads = []
        tower_losses = []
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.num_gpus):
                with tf.device('/gpu:%d' % i):
                    model = Monodepth_yoloModel("train", self.left_images, self.right_images, self.trainable, reuse_variables,0)
                    self.net_vars = tf.global_variables()

                    self.stage_trainable_var_list = []
                    for var in tf.trainable_variables():
                        var_name = var.op.name
                        var_name_mess = str(var_name).split('/')
                        if var_name_mess[1]  not in ["YOLOV3"]:
                            print(var)
                            self.stage_trainable_var_list.append(var)

                    model.build_yolo_loss(self.label_sbbox, self.label_mbbox, self.label_lbbox, self.true_sbboxes,
                                               self.true_mbboxes, self.true_lbboxes)
                    self.giou_loss, self.conf_loss, self.prob_loss, self.monodepth_loss = model.giou_loss, model.conf_loss, model.prob_loss, model.monodepth_loss

                    #计算损失时为为monodepth的损失
                    self.yolo_loss = self.giou_loss + self.conf_loss + self.prob_loss

                    loss = self.monodepth_loss #分开优化不同部分的损失
                    #loss = self.yolo_loss
                    #loss = self.monodepth_loss + self.yolo_loss
                    tower_losses.append(loss)

                    reuse_variables = True
                    grads = opt_step.compute_gradients(loss,var_list=self.stage_trainable_var_list)


                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        self.apply_gradient_op = opt_step.apply_gradients(grads, global_step=self.global_step)

        global_step_update = tf.assign_add(self.global_step, 1)
        self.total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar("giou_loss", self.giou_loss,["model_0"])
        tf.summary.scalar("conf_loss", self.conf_loss,["model_0"])
        tf.summary.scalar("prob_loss", self.prob_loss,["model_0"])
        tf.summary.scalar("yolo_loss", self.yolo_loss,["model_0"])


        tf.summary.scalar("learn_rate", learning_rate, ["model_0"])
        tf.summary.scalar("monodepth_loss", self.monodepth_loss, ['model_0'])

        self.summary_op = tf.summary.merge_all('model_0')

        if os.path.exists(self.train_logdir): shutil.rmtree(self.train_logdir)
        os.mkdir(self.train_logdir)

        # session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # SAVER
        self.summary_writer = tf.summary.FileWriter(self.train_logdir, self.sess.graph)
        self.loader = tf.train.Saver(self.net_vars)
        self.saver = tf.train.Saver(self.net_vars, max_to_keep=5)

        #COUNT PARAMS
        total_num_parameters = 0
        for variable in self.stage_trainable_var_list:
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

    def train(self):
        # INIT
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coordinator)

        try:
            print("Restore weights from: %s ..."% self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print("=> %s does not exist ! ! !"%self.initial_weight)
            print("=> Now it starts to train model  from scratch ... ")

        # GO!

        for epoch in range(1, self.stage_epochs + 1):
            for train_data in self.trainset:
                train_epoch_loss = []
                _, summary_str, loss_value,monodepth_loss, globalstep = self.sess.run([self.apply_gradient_op, self.summary_op,
                                                                        self.total_loss,self.monodepth_loss,self.global_step],
                                                      feed_dict={
                                                          self.left_images: train_data[0],
                                                          self.right_images: train_data[1],
                                                          self.label_sbbox: train_data[2],
                                                          self.label_mbbox: train_data[3],
                                                          self.label_lbbox: train_data[4],
                                                          self.true_sbboxes: train_data[5],
                                                          self.true_mbboxes: train_data[6],
                                                          self.true_lbboxes: train_data[7],
                                                          self.trainable: False,
                                                      })
                print("monodepth_yolo_loss is %s monodepth loss is %s"%(loss_value,monodepth_loss))
                train_epoch_loss.append(loss_value)
                self.summary_writer.add_summary(summary_str, global_step=globalstep)
                print(("train loss: %.2f" % loss_value))

            train_epoch_loss = np.mean(train_epoch_loss)
            ckpt_file =  self.monodepth_log_directory+"model=%.4f.ckpt" % train_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Saving %s ..."% (epoch, log_time,  train_epoch_loss, ckpt_file))

            self.saver.save(self.sess, ckpt_file, global_step=epoch)

if __name__ == '__main__': monodepth_yoloTrain().train()





