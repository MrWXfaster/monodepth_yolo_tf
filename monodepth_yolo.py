# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import cv2
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from bilinear_sampler import *
from monodepth_dataloader import *
from core_yolo.config import cfg
import core_yolo.utils as utils



class Monodepth_yoloModel(object):
    """monodepth model"""
    '''
    1 left/right: 是left_image_batch, right_image_batch.左右图以batch形式传进来
    2 mode: train或test
    3 params: 传进来很多参数
    4 build_model(): 创建模型
    5 build_output(): 创建输出
    6 build_losses(): 创建损数函数
    7 build_summaries: 可视化工具
    '''

    def __init__(self, mode, left, right,trainable, reuse_variables=None, model_index=0):
        self.mode                      = mode
        self.left                      = left
        self.right                     = right
        # self.label_sbbox               = label_sbbox
        # self.label_mbbox               = label_mbbox
        # self.label_lbbox               = label_lbbox
        # self.true_sboxes               = true_sbboxes
        # self.true_mboxes               = true_mbboxes
        # self.true_lboxes               = true_lbboxes

        self.trainable                 = trainable

        self.reuse_variables           = reuse_variables

        self.use_deconv                = cfg.MONODEPTH.USE_DECONV
        self.do_stereo                 = cfg.MONODEPTH.DO_STEREO
        self.encoder                   = cfg.MONODEPTH.ENCODER
        self.alpha_image_loss          = cfg.MONODEPTH.ALPHA_IMAGE_LOSS
        self.disp_gradient_loss_weight = cfg.MONODEPTH.DISP_GRADIENT_LOSS_WEIGHT
        self.lr_loss_weight            = cfg.MONODEPTH.LR_LOSS_WEIGHT
        self.full_summary              = cfg.MONODEPTH.FULL_SUMMARY

        self.strides                   = np.array(cfg.YOLO.STRIDES)

        self.upsample_method           = cfg.YOLO.UPSAMPLE_METHOD

        self.anchor_per_scale          = cfg.YOLO.ANCHOR_PER_SCALE
        self.anchors                   = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.classes                   = utils.read_class_names(cfg.YOLO.CLASSES)
        self.iou_loss_thresh           = cfg.YOLO.IOU_LOSS_THRESH
        self.num_class                 = len(self.classes)

        self.model_collection = ['model_' + str(model_index)]

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return


        self.build_nomodepth_loss()

        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        # 最近邻上采样调用
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        '''
        1. 先将原图(batch形式)放在队列里面
        2. 得到原图的长宽
        3. 根据尺度个数,生成1/2 1/4 1/8 ... 尺度的图,加入到队列中
        4. 输出队列,为图像金字塔结果
        :param img: left图像
        :param num_scales: 4
        :return:shape [[batch_size, h, w, 3]]
        '''
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1] #长
        w = s[2] #宽
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1) #2 4 8
            nh = h // ratio #h/2 h/4 h/8
            nw = w // ratio #w/2 w/4 w/8
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))

        return scaled_imgs

    def generate_image_left(self, img, disp):
        '''
        一维双线性采样
        :param img:
        :param disp:
        :return:
        '''
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)
    # '''
    # tf.nn.avg_pool(
    #         value,
    #         ksize,
    #         strides,
    #         padding,
    #         data_format='NHWC',
    #         name=None
    #     )
    # '''
    def avg_conv(self,input_data, name):
        input_channel = tf.split(input_data, 3, axis=3)
        output_datas = []
        for i, input_ in enumerate(input_channel):
            weight = tf.constant(1, dtype=tf.float32, shape=(3, 3, 1, 1), name=name)
            output_channel = tf.nn.conv2d(input_, weight, strides=(1, 1, 1, 1),
                                          padding="VALID") / 9  # (1, 720, 1280, 1)
            output_datas.append(output_channel)
        output = tf.concat([output_datas[0], output_datas[1], output_datas[2]], axis=-1)
        return output

    def SSIM(self,x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = self.avg_conv(x, name="mu_x")
        mu_y = self.avg_conv(y, name="mu_y")

        sigma_x = self.avg_conv(x ** 2, name="sigma_x") - mu_x ** 2
        sigma_y = self.avg_conv(y ** 2, name="sigma_y") - mu_y ** 2
        sigma_xy = self.avg_conv(x * y, name="sigma_xy") - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keepdims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keepdims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def conv_end(self,x, filter_shape,name):
        with tf.variable_scope(name):
            strides = (1, 1, 1, 1)
            padding = "SAME"
            weight = tf.get_variable(name="weight", dtype=tf.float32, trainable=True,
                                     shape=filter_shape, initializer=tf.random_normal_initializer(stddev=(0.01)))
            conv = tf.nn.conv2d(input=x, filter=weight, strides=strides, padding=padding)
            conv = tf.nn.sigmoid(conv)
        return conv

    def get_disp(self, x):
        '''
        在upconv之后使用CNN加sigmoid函数,乘以0.3之后作为视差图
        :param x:
        :return:
        '''
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv_op(self, x, filter_shape, trainable, name, downsample=False, activate=True, bn=False):
        with tf.variable_scope(name):
            if downsample:
                pad_h , pad_w = (filter_shape[0] - 2)  // 2 + 1, (filter_shape[1] -2) // 2 + 1
                paddings = tf.constant([[0,0], [pad_h, pad_h], [pad_w,pad_w], [0, 0]])
                x = tf.pad(x, paddings, "CONSTANT")
                strides = (1, 2, 2, 1)
                padding = "VALID"
            else:
                strides = (1, 1, 1, 1)
                padding = "SAME"
            weight = tf.get_variable(name="weight", dtype=tf.float32, trainable=True,
                                     shape=filter_shape, initializer=tf.random_normal_initializer(stddev=(0.01)))
            conv = tf.nn.conv2d(input=x, filter=weight, strides=strides, padding=padding)

            if bn:
                conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                    gamma_initializer=tf.ones_initializer(),
                                                     moving_mean_initializer=tf.zeros_initializer(),
                                                     moving_variance_initializer=tf.ones_initializer(),training=trainable)
            else:
                bias = tf.get_variable(name="bias", shape=filter_shape[-1], trainable=True,
                                    dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, bias)
            if activate == True: conv = tf.nn.elu(conv)
        return conv

    def upsample_yolo(self,x, name, method="deconv"):
        assert method in ["resize", "deconv"]
        if method == "resize":
            with tf.variable_scope(name):
                input_shape = tf.shape(x)
                output = tf.image.resize_nearest_neighbor(x, (input_shape[1] * 2, input_shape[2] * 2))

        if method == "deconv":
            # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
            numm_filter = x.shape.as_list()[-1]
            output = tf.layers.conv2d_transpose(x, numm_filter, kernel_size=2, padding='same',
                                                strides=(2, 2), kernel_initializer=tf.random_normal_initializer())
        return output



    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]


    def build_vgg(self):
        # set convenience functions
        # with tf.variable_scope("MONODEPTH"):
            conv = self.conv
            # 选择上采样还是反卷积
            if self.use_deconv:
                upconv = self.deconv  # 反卷积
            else:
                upconv = self.upconv  # 先上采样然后1*1的卷积核卷积

            with tf.variable_scope('encoder'):  # 编码
                conv1 = self.conv_block(self.model_input, 32, 7)  # H/2   1/2 卷积降低图像尺度和增加特征图1/2
                conv2 = self.conv_block(conv1, 64, 5)  # H/4   1/4
                conv3 = self.conv_block(conv2, 128, 3)  # H/8   1/8
                self.conv4 = self.conv_block(conv3, 256, 3)  # H/16  1/16
                self.conv5 = self.conv_block(self.conv4, 512, 3)  # H/32  1/32
                self.conv6 = self.conv_block(self.conv5, 512, 3)  # H/64  1/64
                conv7 = self.conv_block(self.conv6, 512, 3)  # H/128 1/128

            with tf.variable_scope('skips'):
                skip1 = conv1
                skip2 = conv2
                skip3 = conv3
                skip4 = self.conv4
                skip5 = self.conv5
                skip6 = self.conv6

            with tf.variable_scope('decoder'):
                upconv7 = upconv(conv7, 512, 3, 2)  # H/64
                concat7 = tf.concat([upconv7, skip6], 3)
                iconv7 = conv(concat7, 512, 3, 1)

                upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
                concat6 = tf.concat([upconv6, skip5], 3)
                iconv6 = conv(concat6, 512, 3, 1)

                upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
                concat5 = tf.concat([upconv5, skip4], 3)
                iconv5 = conv(concat5, 256, 3, 1)

                upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
                concat4 = tf.concat([upconv4, skip3], 3)
                iconv4 = conv(concat4, 128, 3, 1)
                self.disp4 = self.get_disp(iconv4)
                udisp4 = self.upsample_nn(self.disp4, 2)

                upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
                concat3 = tf.concat([upconv3, skip2, udisp4], 3)
                iconv3 = conv(concat3, 64, 3, 1)
                self.disp3 = self.get_disp(iconv3)
                udisp3 = self.upsample_nn(self.disp3, 2)

                upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
                concat2 = tf.concat([upconv2, skip1, udisp3], 3)
                iconv2 = conv(concat2, 32, 3, 1)
                self.disp2 = self.get_disp(iconv2)
                udisp2 = self.upsample_nn(self.disp2, 2)

                upconv1 = upconv(iconv2, 16, 3, 2)  # H
                concat1 = tf.concat([upconv1, udisp2], 3)
                iconv1 = conv(concat1, 16, 3, 1)
                self.disp1 = self.get_disp(iconv1)
            # for var in tf.global_variables("MONODEPTH"):
            #     tf.add_to_collection("MONODEPTH", var)

    def build_resnet50(self):
        pass

    def yolo_model(self):
        '''
        :return:
        '''
        with tf.variable_scope("YOLOV3"):
            input_data = self.conv_op(self.conv6, (1, 1, 512, 512),   self.trainable,  'conv52')
            input_data = self.conv_op(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
            input_data = self.conv_op(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
            input_data = self.conv_op(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
            input_data = self.conv_op(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

            conv_lobj_branch = self.conv_op(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
            conv_lbbox = self.conv_op(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

            input_data = self.conv_op(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
            input_data = self.upsample_yolo(input_data, name="upsample_yolo1", method = self.upsample_method)

            with tf.variable_scope('route_1'):
                input_data = tf.concat([input_data, self.conv5], axis=-1)

            input_data = self.conv_op(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
            input_data = self.conv_op(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
            input_data = self.conv_op(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
            input_data = self.conv_op(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
            input_data = self.conv_op(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

            conv_mobj_branch = self.conv_op(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
            conv_mbbox = self.conv_op(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

            input_data = self.conv_op(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
            input_data = self.upsample_yolo(input_data, name="upsample_yolo2", method = self.upsample_method)

            with tf.variable_scope('route_2'):
                input_data = tf.concat([input_data, self.conv4], axis=-1)

            input_data = self.conv_op(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
            input_data = self.conv_op(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
            input_data = self.conv_op(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
            input_data = self.conv_op(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
            input_data = self.conv_op(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

            conv_sobj_branch = self.conv_op(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
            conv_sbbox = self.conv_op(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)
        for var in tf.global_variables("YOLOV3"):
            tf.add_to_collection("YOLOV3", var)

        return conv_lbbox, conv_mbbox, conv_sbbox


    def decode(self, conv_output, anchors, stride):
        '''
        return tensor of shape[batch_size, height, width, anchor_per_scale, 5+num_classes]
                contains (x, y, w, h, score, probability)
        :param conv_output:
        :param anchors:
        :param stride:
        :return:
        '''
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_height = conv_shape[1]
        output_width = conv_shape[2]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_height, output_width, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        #(4, 1)  &  (1, 8)  = shape: (4, 8)
        y = tf.tile(tf.range(output_height, dtype=tf.int32)[:, tf.newaxis], [1, output_width])
        x = tf.tile(tf.range(output_width,  dtype=tf.int32)[tf.newaxis, :], [output_height,  1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :],[batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        '''
        :param target:
        :param actual:
        :param alpha:
        :param gamma:
        :return:
        '''
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_height = conv_shape[1]
        output_width = conv_shape[2]

        input_height = stride * output_height
        input_width = stride * output_width

        conv = tf.reshape(conv, (batch_size, output_height, output_width,
                                 self.anchor_per_scale, 5 + self.num_class))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)

        input_height = tf.cast(input_height, tf.float32)
        input_width = tf.cast(input_width, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (
                    input_height * input_width)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :],
                            bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def build_model(self):
        '''
        调用slim的基础CNN操作
        1 生成左图金字塔: 尺度为4
        2 如果训练则生成右图金字塔, 如果做stereo, 则把左右图在channel维上叠在一起作为模型输入.否则把左图作为模型输入
        3 根据parms里面的参数设定,选择vgg或者resnet50作为编码器

        :return:
        '''
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model',reuse=self.reuse_variables):
                '''
                tf.get_variable_scope().reuse == False， 调用该函数会创建新的变量
                tf.get_variable_scope().reuse == True，  调用该函数会重用已经创建的变量
                '''

                self.left_pyramid = self.scale_pyramid(self.left, 4)  # 图像金字塔四个尺度,shape [ [batch_size, h/2, w/2, 3]]
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)
                print("start do_stereo:", self.do_stereo)
                if self.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)  # 在个维度拼接[batch_size, h, w, 3]
                else:
                    self.model_input = self.left

                # build model
                if self.encoder == 'vgg':
                    print("vgg build start")
                    self.build_vgg()
                elif self.encoder == 'resnet50':
                    print("resnet50 build start")
                    self.build_resnet50()
                try:
                    self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.yolo_model()
                except:
                    raise NotImplementedError("Can not build up yolov3 network!")
                else:
                    return None


    def build_outputs(self):
        # STORE DISPARITIES #生成
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]  # 左视差图
            self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]  # 右视差图

        with tf.variable_scope("pred_sbbox"):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])
        with tf.variable_scope("pred_mbbox"):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])
        with tf.variable_scope("pred_sbbox"):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

        if self.mode == 'test':
            return

        # GENERATE IMAGES
        with tf.variable_scope('images'):  # 产生左图与右图
            self.left_est = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]
        # LR CONSISTENCY #左右一致性检测
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i]) for i in
                                       range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in
                                       range(4)]
        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)



    def build_yolo_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('giou_loss'):
            self.giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            self.conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            self.prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]



    def build_nomodepth_loss(self):
        with tf.variable_scope('monodepth_loss',reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]
            # SSIM
            self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [
                self.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.alpha_image_loss) *
                self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left = [
                self.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.alpha_image_loss) *
                self.l1_reconstruction_loss_left[i] for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY
            self.lr_left_loss = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in
                                 range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in
                                  range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)
            # TOTAL LOSS
            self.monodepth_loss =self.image_loss + self.disp_gradient_loss_weight * self.disp_gradient_loss + self.lr_loss_weight * self.lr_loss


    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i),
                                  self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4,
                                 collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4,
                                 collections=self.model_collection)

                if self.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_left_' + str(i), self.ssim_left[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('l1_left_' + str(i), self.l1_left[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4,
                                     collections=self.model_collection)

            # if self.full_summary:
            #     tf.summary.image('left', self.left, max_outputs=4, collections=self.model_collection)
            #     tf.summary.image('right', self.right, max_outputs=4, collections=self.model_collection)


if __name__ == "__main__":
    from monodepth_yolo_dataloader import Datasetloader

    left_images = tf.placeholder(dtype=tf.float32,shape=(None, 256, 512, 3), name='left_image')
    right_images = tf.placeholder(dtype=tf.float32,shape=(None, 256, 512, 3),name='right_image')

    label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
    label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
    label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
    true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
    true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
    true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
    trainable = tf.placeholder(dtype=tf.bool, name="training")

    model = Monodepth_yoloModel("train", left_images, right_images, trainable, None, 0)
    net_var = tf.global_variables()
    model.build_yolo_loss(label_sbbox, label_mbbox, label_lbbox, true_sbboxes,
                               true_mbboxes, true_lbboxes)
    giou_loss, conf_loss, prob_loss, monodepth_loss = model.giou_loss, model.conf_loss, model.prob_loss, model.monodepth_loss
    total_loss = monodepth_loss


    # Session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    testset = Datasetloader('test')

    test_data = next(testset)
    monodepthloss= sess.run(monodepth_loss,
    feed_dict={
        left_images: test_data[0],
        right_images: test_data[1],
        label_sbbox: test_data[2],
        label_mbbox: test_data[3],
        label_lbbox: test_data[4],
        true_sbboxes: test_data[5],
        true_mbboxes: test_data[6],
        true_lbboxes: test_data[7],
    })
    print(monodepthloss)






