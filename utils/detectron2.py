import os 
import time

import cv2
import numpy as np

# Detectron2
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.projects.deeplab import add_deeplab_config


# MLUtils
import utils.tools.config as config
from utils.tools.detectron2_objects import SemsegPredictor
import utils.tools as tools
import utils.tools.ann_data as ann_data










class detectron2_model():
    def __init__(self, cfg, target_model=""):
        if target_model != "":
            pass
        else:
            cfg = config.config(cfg_filename="", target_model=target_model)
        self.cfg = cfg
        self.conf_threshold = cfg.score_threshold
        self.nms_threshold = cfg.nms_threshold
        self.weights = cfg.inference_model_weights
        self.class_names = cfg.class_names

        self.operation = "object_detection"
        if "insseg" in cfg.target_model:
            self.operation = "instance_segmentation"
            self.imgsz_min = self.cfg.model_cfg["train"]["imgsz"]
            self.imgsz_max = self.cfg.model_cfg["train"]["imgsz"]
        if "semseg" in cfg.target_model:
            self.operation = "semantic_segmentation"
        print("Model: {} \nOperation: {} \nAnnotation format: {}".format(cfg.target_model, self.operation, cfg.ann_format))




        model_name = self.cfg.model_cfg["train"]["model"]
        if isinstance(model_name, list):
            model_name = model_name[0]



        if cfg.inference_runtime == "default":
            # Configure run
            d2_cfg = get_cfg()
            d2_cfg.MODEL.DEVICE = cfg.inference_device

            if self.operation == "instance_segmentation":
                # Set up detectron2
                if "keypoint" in self.weights:
                    d2_cfg.merge_from_file(model_zoo.get_config_file(model_name))
                    d2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
                    d2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_names)
                    d2_cfg.MODEL.KEYPOINT_ON = True
                    d2_cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(cfg.keypoint_names)
                    d2_cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((len(cfg.keypoint_names), 1), dtype=float).tolist()
                else:        
                    if "pointrend" in self.weights:
                        add_pointrend_config(d2_cfg)
                        d2_cfg.merge_from_file(model_name)
                        d2_cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(self.class_names)
                    if "mask_rcnn" in self.weights:
                        d2_cfg.merge_from_file(model_zoo.get_config_file(model_name))

                    d2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_names)
                    d2_cfg.MODEL.WEIGHTS = self.weights
                    d2_cfg.INPUT.MAX_SIZE_TRAIN = self.imgsz_max
                    d2_cfg.INPUT.MAX_SIZE_TEST = self.imgsz_max
                    d2_cfg.INPUT.MIN_SIZE_TRAIN = self.imgsz_min
                    d2_cfg.INPUT.MIN_SIZE_TEST = self.imgsz_min
                    d2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_threshold
                    d2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.nms_threshold
                    d2_cfg.TEST.DETECTIONS_PER_IMAGE = 5000
                    d2_cfg.MODEL.MASK_ON = True
                self.predictor = DefaultPredictor(d2_cfg)



            if self.operation == "semantic_segmentation":
                if "pointrend" in self.weights:
                    add_pointrend_config(d2_cfg)
                    d2_cfg.merge_from_file(model_name)
                    d2_cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(self.class_names) + 1
                    d2_cfg.MODEL.WEIGHTS = self.weights

                if "deeplab" in self.weights:
                    add_deeplab_config(d2_cfg)
                    d2_cfg.merge_from_file(model_name)
                    d2_cfg.MODEL.WEIGHTS = self.weights

                d2_cfg.SOLVER.IMS_PER_BATCH = 2 
                input_size = self.cfg.model_cfg["train"]["input_size"][0] * self.cfg.model_cfg["train"]["input_size"][1]
                d2_cfg.INPUT.FORMAT = 'BGR'
                d2_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(self.class_names) + 1
                d2_cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING= "choice"
                d2_cfg.INPUT.MIN_SIZE_TRAIN= input_size
                d2_cfg.INPUT.MAX_SIZE_TRAIN= input_size
                d2_cfg.INPUT.MIN_SIZE_TEST= input_size
                d2_cfg.INPUT.MAX_SIZE_TEST= input_size
                d2_cfg.INPUT.CROP.SIZE = (input_size,input_size)

                self.predictor = SemsegPredictor(d2_cfg)

                
        



    def inference(self, img):

        # Perform prediction
        start = time.perf_counter()
        if self.cfg.inference_runtime == "default":
            img_output = img.copy()
            if self.operation != "semantic_segmentation":
                outputs = self.predictor(img)
                if self.cfg.multiclass_nms:
                    final_boxes, final_scores, final_classes, final_masks  = tools.multiclass_nms(outputs, 
                                                                                                self.cfg.multiclass_nms_threshold)
                else:
                    instances = outputs["instances"].to("cpu")
                    final_classes = instances.pred_classes.numpy()
                    final_scores = instances.scores.numpy()
                    final_boxes = instances.pred_boxes.tensor.numpy()
                    final_masks = instances.pred_masks
                    final_masks = np.asarray(final_masks)
            else:
                outputs = self.predictor(img)["sem_seg"]
            elapsed_time = (time.perf_counter() - start)*1000



 
            
        final_indices = [i for i, s in enumerate(final_scores) if s >= self.conf_threshold]
        preds = []
        for i in final_indices:
            a = ann_data()
            a.score = final_scores[i]
            a.box = list(final_boxes[i])
            a.cl = final_classes[i]+1

            if self.operation != "object_detection":
                m = cv2.resize(np.int32(final_masks[i]), (img.shape[1], img.shape[0]))
                a.polygons, _ = tools.mask2polygon(m)
                a.mask = m
            preds.append(a)


        return preds, elapsed_time


    def process_image(self, file_name=None, img=None, show_output=True, score_threshold=0):

        if score_threshold != 0:
            self.conf_threshold = score_threshold

        # Load image
        if file_name != None:
            raw_img = cv2.imread(file_name)
        else:
            raw_img = img

        img = raw_img
        img_output = raw_img.copy()
        img_output_post = img_output.copy()

        preds, elapsed_time = self.inference(raw_img)



        # Get output classes
        output_classes = [list([a.cl for a in preds]).count(i+1) for i, c in enumerate(self.class_names)]


        # Visualize outputs
        if len(preds) > 0:
            img_output = tools.visualize_annotations(raw_img,
                                                    preds, 
                                                    self.cfg, 
                                                    ignore_classes=[], 
                                                    show_output=show_output)


        results = {"img": img, "output": {"img_output": img_output, "preds":preds, "class_total":output_classes, "elapsed_time": elapsed_time}}

        return results

