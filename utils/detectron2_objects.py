import torch 
import os
import numpy as np 

import stomatalia_tools.utils.detectron2_hooks as train_hook

# Insseg
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, DatasetCatalog, build_detection_test_loader

# Semseg
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.modeling import build_model



__all__ = ["CustomTrainer", "CustomTrainerWithTracking"]


def build_sem_seg_train_aug(cfg):
    print(cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
    augs = [
        # T.ResizeShortestEdge(
        #     cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        # )
        T.Resize(
            (cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    return augs




class SemsegPredictor:
  
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = T.Resize(
            (cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
        )
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format


    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]


            return predictions
        
        


        
class CustomTrainer(DefaultTrainer):


    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     if output_folder is None:
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    #     model_name = cfg.MODEL_NAME

    #     if "keypoint" in model_name[0]:
    #         return COCOEvaluator(dataset_name, ("bbox", "segm", "keypoints"), False, output_folder,
    #                          kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS)
    #     if "instance" in model_name[0]:
    #         return COCOEvaluator(dataset_name, output_dir=output_folder)
    #     if "Semantic" in model_name[0]:
    #         evaluator_list = []
    #         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    #         if evaluator_type == "sem_seg":
    #             return SemSegEvaluator(
    #                 dataset_name,
    #                 distributed=True,
    #                 output_dir=output_folder,
    #             )
    #         if len(evaluator_list) == 0:
    #             raise NotImplementedError(
    #                 "no Evaluator for the dataset {} with the type {}".format(
    #                     dataset_name, evaluator_type
    #                 )
    #             )
    #         if len(evaluator_list) == 1:
    #             return evaluator_list[0]
    #         return DatasetEvaluators(evaluator_list)


    @classmethod ## beging able to do evaluation using custom augmentations.
    def build_test_loader(cls, cfg, dataset_name):
        # if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            #this mapper is used for dataaugmentation see augmentations=

        mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize((cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN))])
        return build_detection_test_loader(DatasetCatalog.get(cfg.DATASETS.TEST[0]), mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)



    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)


    def build_hooks(self):
        hooks = super().build_hooks()
        # hooks.insert(-1, CheckpointSaver(self.cfg.TEST.EVAL_PERIOD, 0.0, 'keypoints/AP'))
        # hooks.insert(-1, train_hook.ExportSaver(self.cfg))

        try:
            getattr(self.cfg, "LOGGER")
            if self.cfg.LOGGER.TYPE == "mlflow":
                hooks.insert(-1, train_hook.MLFlowLogger(self.cfg))
            # if self.cfg.LOGGER.TYPE == "wandb":
            #     hooks.insert(-1, train_hook.WANDBLogger(self.cfg))
        except:
            print("No loggers enabled!")
        
        return hooks

