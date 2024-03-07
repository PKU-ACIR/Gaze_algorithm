import torch
import torchvision
import torch.backends.cudnn as cudnn
import Detectron2
import pyrealsense2 as rs
from random import randint

import numpy as np
import os, json, cv2, random

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model  
from detectron2.checkpoint import DetectionCheckpointer

from object_tracking.utils.parser import get_config
from object_tracking.deep_sort import DeepSort
#from gpu_mem_track import MemTracker
import time
from yolact.data.config import cfg as data_name


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def draw_ids(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        # color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # cv2.rectangle(
        #     img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


class ObjectDetector:
    def __init__(self):
        setup_logger()

        self.cfg = get_cfg()
        self.cfg.merge_from_file('Detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
        self.cfg.MODEL.WEIGHTS = 'Detectron2/models/R101FPN.pkl'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        self.predictor = DefaultPredictor(self.cfg)


        cfg = get_config()
        cfg.merge_from_file("object_tracking/configs/deep_sort.yaml")

        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)


        cudnn.benchmark = True  # set True to speed up constant image size inference

    def get_prediction(self, color_image, depth_image, depth_intrin, depth_scale, total_points_num=10):

        time_s = time.time()
        detections = self.predictor(color_image)
        time_e = time.time()
        print("segmentation:")
        print(time_e-time_s)
        classes = detections['instances'].pred_classes.cpu().numpy()
        boxes = detections['instances'].pred_boxes.tensor.cpu().numpy()
        masks = detections['instances'].pred_masks.cpu().numpy()

        xywh_bboxs = []
        for box in boxes:
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*box)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
        if xywh_bboxs == []:
            return False,None, None, None
        xywhs = torch.Tensor(xywh_bboxs)
        confss = detections['instances'].scores
        time_s = time.time()
        outputs = self.deepsort.update(xywhs, confss, color_image)
        time_e = time.time()
        print("deep_sort:")
        print(time_e-time_s)
        time_s = time.time()
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            draw_ids(color_image, bbox_xyxy, identities)

            v = Visualizer(color_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(detections["instances"].to("cpu"))
            out_image = out.get_image()[:, :, ::-1]
            cv2.imshow('detection result', out_image)
            cv2.waitKey(1)

            point_masks = masks.sum(axis=0)
            point_idx = 0
            h, w, _ = color_image.shape
            class_lst = list()
            points_lst = list()
            id_lst = list()
            max_depth = 6  # max depth value of realsense D455 is 6 meters
            sample_count = 0

            while True:
                sample_count += 1
                ih, iw = randint(0, h - 1), randint(0, w - 1)
                if point_masks[ih, iw] != 0:
                    depth_value = depth_image[ih, iw] * depth_scale
                    if 0 < depth_value <= max_depth:
                        mask_pixel = masks[:, ih, iw]
                        temp_point_index = np.argwhere(mask_pixel == 1)[0, 0]
                        curr_class = classes[temp_point_index]
                        if curr_class == 0 or data_name.dataset.class_names[curr_class] == "dining table":  # not detecting person
                            continue
                        else:
                            curr_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [iw, ih], depth_value)
                            curr_point[0] =  curr_point[0]
                            curr_point[1] =  curr_point[1]
                            class_lst.append(curr_class)
                            points_lst.append(curr_point)
                            # TODO: check the mapping from detections to track output
                            #id_lst.append(identities[temp_point_index])
                            point_idx += 1
                if point_idx >= total_points_num or sample_count > 100000:
                    break
            time_e = time.time()
            print("point:")
            print(time_e-time_s)
            classes = np.asarray(class_lst)
            points = np.asarray(points_lst)
            #ids = np.asarray(id_lst)
            ids = np.zeros_like(classes)

            return True,classes, points, ids

        else:
            return False,None, None, None

        # outputs = self.predictor(color_image)
        # classes = outputs['instances'].pred_classes.cpu().detach().numpy()  # N dim vector
        # masks = outputs['instances'].pred_masks.cpu().detach().numpy()  # (N, H, W) array
        # boxes = outputs['instances'].pred_boxes  # (N, 4) array
        #
        # v = Visualizer(color_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # result_img = out.get_image()[:, :, ::-1]
        # cv2.imshow('result', result_img)
        # cv2.waitKey(1)
        #
        # points_num = 100
        # point_masks = masks.sum(axis=0)
        # point_idx = 0
        # h, w, _ = color_image.shape
        # class_lst = list()
        # points_lst = list()
        # id_lst = list()
        #
        # max_id = np.shape(classes)[0]
        #
        # while True:
        #     ih, iw = randint(0, h - 1), randint(0, w - 1)
        #     if point_masks[ih, iw] != 0:
        #         depth_value = depth_image[ih, iw] * depth_scale
        #         if 0 < depth_value <= 6:
        #             mask_pixel = masks[:, ih, iw]
        #             curr_id = np.argwhere(mask_pixel == 1)[0, 0]  # id ranging from 0~(N-1)
        #             curr_class = classes[curr_id]
        #             if curr_class == 0 or curr_class == 60:  # not detecting person
        #                 continue
        #             else:
        #                 curr_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [iw, ih], depth_value)
        #                 class_lst.append(curr_class)
        #                 points_lst.append(curr_point)
        #                 id_lst.append(curr_id)
        #                 point_idx += 1
        #     if point_idx >= points_num:
        #         break

        # torch_color_image = torch.from_numpy(color_image).cuda().float()
        # preds = evalimage(self.yolact, torch_color_image)
        # points, classes = generate_points(preds, torch_color_image, depth_image, depth_intrin, depth_scale)
        # points = np.asarray(points)
        # classes = np.asarray(classes)
        #
        # preds = evalimage(self.yolact, torch_color_image)
        # numpy_color_image = prep_display(preds, torch_color_image, None, None, undo_transform=False)
        # cv2.imshow('yolact result', numpy_color_image)
        # cv2.waitKey(1)
