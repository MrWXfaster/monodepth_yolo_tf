#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      :
#   File name   : monodepth_yolo_dataloader.py
#   Author      : WX1995
#   Created date: 2019-12-15 18:05:03
#   Description :
#
#================================================================
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from core_yolo.config import cfg
import core_yolo.utils as utils
from matplotlib import pyplot as plt





class Datasetloader(object):
    """implement Dataset here"""
    def __init__(self, mode):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH   if mode == 'train' else cfg.TEST.ANNOT_PATH
        self.height      = cfg.TRAIN.INPUT_HEIGHT if mode == 'train' else cfg.TEST.INPUT_HEIGHT
        self.width       = cfg.TRAIN.INPUT_WIDTH  if mode == 'train' else cfg.TEST.INPUT_WIDTH
        self.batch_size  = cfg.TRAIN.BATCH_SIZE   if mode == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG     if mode == 'train' else cfg.TEST.DATA_AUG

        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(mode)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self, mode):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations) #得到随机的标签列表
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            #train_input_height : 256 train_input_width : 512
            self.train_input_height, self.train_input_width = self.height, self.width

            self.train_output_height, self.train_output_width = self.train_input_height //self.strides, self.train_input_width // self.strides

            left_image_batch = np.zeros((self.batch_size, self.train_input_height, self.train_input_width,3))
            right_image_batch = np.zeros((self.batch_size, self.train_input_height, self.train_input_width, 3))



            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_height[0], self.train_output_width[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_height[1], self.train_output_width[1],
                                           self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_height[2], self.train_output_width[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num #得到一个batch_size的位置索引
                    if index >= self.num_samples: index -= self.num_samples #当索引超过最大值时， 减去总的样本数
                    annotation = self.annotations[index] # 得到一个训练样本的标签信息
                    imageL,  bboxes, imageR, = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    left_image_batch [num, :, :, :] = imageL
                    right_image_batch[num, :, :, :] = imageR
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return left_image_batch, right_image_batch, bboxes, annotation,batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration


    def monodepth_image_process(self, left_image_path,right_image_path, gt_boxes=None):

        #TODO: 替代image_process function
        left_image_path_str  = str(left_image_path)
        right_image_path_str = str(right_image_path)

        left_image  = np.array(cv2.imread(left_image_path_str,  cv2.IMREAD_UNCHANGED))
        right_image = np.array(cv2.imread(right_image_path_str, cv2.IMREAD_UNCHANGED))

        left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        ih, iw = self.train_input_height, self.train_input_width
        h, w, _ = np.shape(left_image)

        w_scale, h_scale = iw / w, ih / h


        imageL_paded = cv2.normalize(left_image_rgb, None, 0, 1, norm_type=cv2.NORM_MINMAX)
        imageR_paded = cv2.normalize(right_image_rgb, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        imageL_paded = cv2.resize(imageL_paded, (iw, ih), interpolation=cv2.INTER_AREA)
        imageR_paded = cv2.resize(imageR_paded, (iw, ih), interpolation=cv2.INTER_AREA)



        if gt_boxes is None:
            return imageL_paded,imageR_paded
        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * h_scale
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * w_scale
            return imageL_paded, imageR_paded, gt_boxes

    def augment_image_pair(self, cv_left_image, cv_right_image):
        '''图像增强操作'''

        # random shift gamma
        cv_random_gamma = random.uniform(0.8, 1.2)
        cv_left_image_aug = cv_left_image * cv_random_gamma
        cv_right_image_aug = cv_right_image * cv_random_gamma

        # radom shift brightness
        cv_random_brightness = random.uniform(0.5, 2.0)
        cv_left_image_aug *= cv_random_brightness
        cv_right_image_aug *= cv_random_brightness

        # random shfit color
        cv_random_color = [random.uniform(0.8, 1.2) for i in range(3)]
        cv_white = np.ones([np.shape(cv_left_image)[0], np.shape(cv_left_image)[1]])
        cv_color_image = np.stack([cv_white * cv_random_color[i] for i in range(3)], axis=2)

        cv_left_image_aug *= cv_color_image
        cv_right_image_aug *= cv_color_image


        return cv_left_image_aug, cv_right_image_aug



    def parse_annotation(self, annotation):
        line = annotation.split()
        image_pathL = line[0]
        image_pathR = line[1]
        if not os.path.exists(image_pathL) or not os.path.exists(image_pathR):
            raise KeyError("%s or %s does not exist ..." %(image_pathL,image_pathR))

        bboxes = np.array([list(map(int, box.split(','))) for box in line[2:]])

        #1.读取图像
        left_image,right_image, bboxes = self.monodepth_image_process(image_pathL,image_pathR, np.copy(bboxes))

        #2.randomly flip images
        cv_do_flip = random.uniform(0, 1)
        if cv_do_flip > 0.5:
            _, w, _ = left_image.shape
            cv_left_image_flip = right_image[:, ::-1,  :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        else:
            cv_left_image_flip = left_image

        #cv_left_image_flip  = right_image[:, ::-1, :] if cv_do_flip > 0.5 else left_image
        cv_right_image_flip = left_image[:, ::-1, :]  if cv_do_flip > 0.5 else right_image

        #3.图像增强
        cv_do_augment = random.uniform(0, 1)
        if cv_do_augment > 0.5:
            imageL, imageR = self.augment_image_pair(np.copy(cv_left_image_flip), np.copy(cv_right_image_flip))
        else:
            imageL = cv_left_image_flip
            imageR = cv_right_image_flip

        #staurate
        #tf.clip_by_value
        #为了解决下面的这个BUG
        #Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

        #<0 的变为0
        imageL_less = (1 - (imageL > 0)) * 1e-8
        imageL = imageL * (imageL > 0) + imageL_less

        #>1 的变为1
        left_greater_than =  (1 - (imageL < 1)).astype(np.float32) * 0.99999
        left_less_than  = imageL <  1
        imageL = imageL * left_less_than + left_greater_than

        imageR_less = (1 - (imageR > 0)) * 1e-8
        imageR = imageR * (imageR > 0) + imageR_less

        right_greater_than = (1 - (imageR < 1)).astype(np.float32) * 0.99999
        right_less_than = imageR < 1
        imageR = imageR * right_less_than + right_greater_than


        return imageL,  bboxes, imageR,

    def bbox_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.train_output_height[i], self.train_output_width[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs


if __name__=="__main__":
    import colorsys
    DATASET = Datasetloader("test")

    def draw_bbox(image, bboxes,classes, show_label=True):

        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    for images in DATASET:
        classes = ["flour"]
        batch_imageL = images[0]
        imageL       = images[0][7]


        batch_imageR = images[1]
        imageR       = images[1][7]

        annotation = images[3].split(" ")
        source_bboxes = np.array([list(map(int, box.split(','))) for box in annotation[2:]])

        imageL_path = annotation[0]
        print("image_pathL is %s"%imageL_path)
        bboxes_source = [np.insert(bbox,4,1) for bbox in source_bboxes]

        source_imageL = cv2.imread(str(imageL_path), cv2.IMREAD_UNCHANGED)
        source_imageL = cv2.cvtColor(source_imageL, cv2.COLOR_BGR2RGB).astype(np.float32)
        source_imageL = cv2.normalize(source_imageL,  None,0, 1, norm_type=cv2.NORM_MINMAX)

        flip_source_imageL = np.copy(source_imageL)
        flip_source_imageL = flip_source_imageL[:,::-1,:]

        draw_bbox(source_imageL,bboxes_source, classes)


        bboxes = [np.insert(bbox,4,1) for bbox in images[2]]
        draw_bbox(imageL, bboxes, classes)


        plt.figure(num=3, figsize=(8,5))
        plt.subplot(2,2,1)
        plt.imshow(imageL)
        plt.subplot(2,2,2)
        plt.imshow(imageR)

        plt.subplot(2,2,3)
        plt.imshow(source_imagel)

        plt.subplot(2,2,4)
        plt.imshow(flip_source_imageL)

        '''
        (2, 16, 32, 3, 6) (2, 8, 16, 3, 6) (2, 4, 8, 3, 6) (2, 150, 4) (2, 150, 4) (2, 150, 4)
        '''
        plt.show()
