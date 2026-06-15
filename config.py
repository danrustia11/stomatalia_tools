import argparse
import yaml
import os
from pathlib import Path


import stomatalia_tools.shortcuts as s


class config():
    def __init__(self, cfg_filename="configs/user_default.yaml", target_model=""):

        if cfg_filename == "":
            parser = argparse.ArgumentParser(description="config")
            parser.add_argument(
                "--config",
                nargs="?",
                type=str,
                default="configs/user_default.yaml",
                help="Configuration file to use",
            )

            try:
                args, unknown_args = parser.parse_known_args()
                cfg_filename = args.config
            except:
                pass
        with open(cfg_filename) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
        self.cfg = cfg




        # Experiment
        self.project_name = cfg["experiment"]["name"]
        if target_model == "":
            self.target_model = cfg["experiment"]["target_model"]
        else:
            self.target_model = target_model

        self.model_cfg = cfg["models"][self.target_model]
        self.display_cfg = cfg["utils"]["display"]
        self.generate_annotation_images = cfg["utils"]["display"]["generate_annotation_images"]

        self.network_drive = cfg["experiment"]["network_drive"]
        self.project_code = cfg["experiment"]["project_code"]

        

        # self.dataset_folder = self.network_drive.format(os.getlogin(), "datasets", self.project_code)
        # self.experiment_folder = self.network_drive.format(os.getlogin(), "experiments", self.project_code)
        self.operation = "object_detection"
        if "classifier" in self.target_model:
            self.operation = "image_classification"
        if "Instance" in self.target_model or "insseg" in self.target_model or "seg" in self.target_model:
            self.operation = "instance_segmentation"
        if "Semantic" in self.target_model or "semseg" in self.target_model:
            self.operation = "semantic_segmentation"
        self.enable_tracking = cfg["utils"]["tracking"]["enabled"]
        if self.enable_tracking:
            self.tracking_software = cfg["utils"]["tracking"]["software"]
        self.test_source = cfg["experiment"]["test_source"]
        self.run_mode = cfg["experiment"]["run_mode"]
        self.debug_level = cfg["experiment"]["debug_level"]


        #
        # Datasets
        #
        self.ver_dir = self.model_cfg["data"]["ver_path"]
        # dataset/<project_name>/images/<type_of_image or location>/<ver>
        # self.img_dir = os.path.join(self.dataset_folder, self.model_cfg["data"]["img_path"]).format(self.ver_dir)
        # s.create_dir(self.img_dir)
        # dataset/<project_name>/anns/<type_of_image or location>/<ver>
        # if self.operation != "image_classification":
        self.ann_format = self.model_cfg["data"]["ann_format"]
        #     self.ann_dir = self.model_cfg["data"]["ann_path"].format(self.ver_dir, self.ann_format)
        # else:
        #     self.ann_dir = self.model_cfg["data"]["ann_path"].format(self.ver_dir)
        # self.ann_dir = os.path.join(self.dataset_folder, self.ann_dir)
        # # s.create_dir(self.ann_dir)
        # self.test_img_dir = os.path.join(self.dataset_folder, self.model_cfg["data"]["test_img_path"])
        # self.test_ann_dir = os.path.join(self.dataset_folder, self.model_cfg["data"]["test_ann_path"])


  



        # Train
        self.train_val_test_split = self.model_cfg["train"]["split"]
        self.retain_split = self.model_cfg["train"]["retain_split"]
        self.model = self.model_cfg["train"]["model"]
        self.optimizer = self.model_cfg["train"]["optimizer"]
        self.lr = self.model_cfg["train"]["lr"]
        self.imgsz = self.model_cfg["train"]["imgsz"]
        self.batch_size = self.model_cfg["train"]["batch_size"]
        self.iterations = self.model_cfg["train"]["iterations"]
        
        # Crop
        self.crop_size = cfg["utils"]["crop"]["size"]
        self.crop_transparent = cfg["utils"]["crop"]["transparent"]
        self.crop_style = cfg["utils"]["crop"]["style"]
        self.crop_square_crop = cfg["utils"]["crop"]["square_crop"]
        self.crop_pad = cfg["utils"]["crop"]["pad"]


        # Inference
        self.inference_device = self.model_cfg["inference"]["device"]
        self.inference_runtime = self.model_cfg["inference"]["model_runtime"]
        self.score_threshold = self.model_cfg["inference"]["score_threshold"]
        if self.operation != "image_classification":
            self.nms_threshold = self.model_cfg["inference"]["nms_threshold"]
            self.semanticize = self.model_cfg["inference"]["semanticize"]
            self.multiclass_nms = self.model_cfg["inference"]["multiclass_nms"]
            self.multiclass_nms_threshold = self.model_cfg["inference"]["multiclass_nms_threshold"]
            # self.custom_processor = self.model_cfg["inference"]["custom_processor"]
            # self.post_process = self.model_cfg["inference"]["post_process"]
            # self.mask_smoothing = self.model_cfg["inference"]["mask_smoothing"]


        # Preannotator
        self.pre_annotate_enabled = cfg["utils"]["pre_annotate"]["enabled"]
        self.pre_annotate_skip_annotated = cfg["utils"]["pre_annotate"]["skip_annotated"]
        self.pre_annotate_upload_image = cfg["utils"]["pre_annotate"]["upload_image"]
        self.pre_annotate_upload_ann = cfg["utils"]["pre_annotate"]["upload_ann"]
        self.pre_annotate_prompt = cfg["utils"]["pre_annotate"]["prompt"]
        self.pre_annotate_temp_folder = cfg["utils"]["pre_annotate"]["temp_folder"]
            
        # Darwin
        self.darwin_dataset_name = cfg["utils"]["darwin"]["dataset_name"]
        self.darwin_release_name = self.model_cfg["data"]["ver_path"]
        self.darwin_folder = cfg["utils"]["darwin"]["folder"]
        self.darwin_api_key = cfg["utils"]["darwin"]["api_key"]
        self.darwin_base_dir = cfg["utils"]["darwin"]["base_dir"]
        self.darwin_team_name = cfg["utils"]["darwin"]["team_name"]
        # self.darwin_data_dir = os.path.join(self.darwin_base_dir, self.darwin_team_name, self.darwin_dataset_name).format(os.getlogin())
        self.darwin_version = cfg["utils"]["darwin"]["version"]

        # Data 
        self.class_names = self.model_cfg["data"]["classes"]
        if self.operation != "image_classification":
            self.include_empty = self.model_cfg["train"]["include_empty"]

            self.tiling_enabled = self.model_cfg["data"]["tiler"]["enabled"]
            if self.tiling_enabled:
                self.tile_size = self.model_cfg["data"]["tiler"]["tile_size"]
                self.tile_padding = self.model_cfg["data"]["tiler"]["padding"]
                self.tile_style = self.model_cfg["data"]["tiler"]["tile_style"]
                self.tile_iou_threshold = self.model_cfg["data"]["tiler"]["iou_threshold"]
        else:
            self.normalize = self.model_cfg["train"]["normalize"]






        # if self.operation == "semantic_segmentation":
        #     self.gt_dir = ""
        #     if "Semantic" in self.target_model:
        #         self.gt_dir = os.path.join(self.data_dir, cfg[self.target_model]["data"]["gt_path"])
        

        self.keypoint_enabled = self.model_cfg["data"]["keypoint"]["enabled"]
        if self.keypoint_enabled:
            self.keypoint_names = self.model_cfg["data"]["keypoint"]["classes"]
        else:
            self.keypoint_names = ['']









        #
        # Fix directories
        #          

        # self.experiment_folder_models = os.path.join(self.experiment_folder, "models")
        # models/<target_model>
        # self.experiment_folder_models_target_model = os.path.join(self.experiment_folder_models, self.target_model)
        # models/<target_model>/<ver>
        # self.experiment_folder_models_target_model_ver = os.path.join(self.experiment_folder_models_target_model, self.ver_dir)
        
        # s.create_dir(self.experiment_folder_models_target_model_ver)

        '''
        if self.run_mode != "train":
            self.weights_name = self.model_cfg["inference"]["model_weights"].split("/")[0]

            # models/<target_model>/<ver>/<weights>
            self.experiment_folder_models_target_model_ver_weights = os.path.join(self.experiment_folder_models_target_model_ver, self.weights_name)
            # models/<target_model>/<ver>/<weights>/weights
            self.experiment_folder_models_target_model_ver_weights_weights = os.path.join(self.experiment_folder_models_target_model_ver_weights, "weights")
            # models/<target_model>/<ver>/<weights>/cropped
            self.experiment_folder_models_target_model_ver_weights_cropped= os.path.join(self.experiment_folder_models_target_model_ver_weights, "cropped")
            # models/<target_model>/<ver>/<weights>/inference
            self.experiment_folder_models_target_model_ver_weights_inference = os.path.join(self.experiment_folder_models_target_model_ver_weights, "inference")
            # models/<target_model>/<ver>/<weights>/output_data
            self.experiment_folder_models_target_model_ver_weights_output_data = os.path.join(self.experiment_folder_models_target_model_ver_weights, "output_data")


            if "mm" in self.target_model:
                self.inference_model_name = self.model_cfg["inference"]["model_name"]
                self.mm_weights_dir = os.path.join(self.experiment_folder_models_target_model, self.inference_model_name)
                self.experiment_folder_models_target_model_weights = self.mm_weights_dir
                self.experiment_folder_models_target_model_weights_model = os.path.join(self.mm_weights_dir, "weights")
                self.inference_model_py = os.path.join(self.experiment_folder_models_target_model_weights_model, self.model_cfg["inference"]["model_py"])
                self.inference_model_weights = os.path.join(self.experiment_folder_models_target_model_weights_model, self.model_cfg["inference"]["model_weights"])
                self.experiment_folder_models_target_model_inference = os.path.join(self.mm_weights_dir, "inference")
                self.experiment_folder_models_target_model_cropped = os.path.join(self.mm_weights_dir, "cropped")
                self.experiment_folder_models_target_model_output_data = os.path.join(self.mm_weights_dir, "output_data")
            else:
                self.inference_model_weights = os.path.join(self.experiment_folder_models_target_model_ver_weights_weights, os.path.basename(self.model_cfg["inference"]["model_weights"]))

            
            # s.create_dir(self.experiment_folder_models_target_model_ver_weights)
            # s.create_dir(self.experiment_folder_models_target_model_ver_weights_cropped)
            # s.create_dir(self.experiment_folder_models_target_model_ver_weights_inference)
            # s.create_dir(self.experiment_folder_models_target_model_ver_weights_output_data)

        '''