from detectron2.utils.logger import setup_logger
from detectron2.engine import HookBase
from detectron2.utils import comm
# import mlflow
import wandb
import torch

import os
from collections.abc import MutableMapping
import pandas as pd
def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

__all__ = ["CheckpointSaver", "MLFlowLogger"]


class CheckpointSaver(HookBase):
    def __init__(self, eval_period, val_value, metric):
        self._period = eval_period
        self.val_value = val_value
        self.metric = metric
        self.logger = setup_logger(name="d2.checkpointer.best")

    def store_best_model(self):
        metric = self.trainer.storage._latest_scalars
        try:
            current_value = metric[self.metric][0]
            try:
                highest_value = metric['highest_value'][0]
            except:
                highest_value = self.val_value

            self.logger.info(
                "current-value ({:s}): {:.2f}, highest-value ({:s}): {:.2f}".format(self.metric, current_value,
                                                                                    self.metric, highest_value))
            if current_value > highest_value:
                self.logger.info("saving best model...")
                self.trainer.checkpointer.save("best_model")
                self.trainer.storage.put_scalar('highest_value', current_value)
                comm.synchronize()
        except:
            pass

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self.store_best_model()
        self.trainer.storage.put_scalars(timetest=12)

#
# Make a custom .yaml file for exporting
# - DJAR
class ExportSaver(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
    def after_train(self):
        with open(os.path.join(self.cfg.OUTPUT_DIR, "model-export.yaml"), "w") as f:
            del self.cfg["MODEL_NAME"]
            del self.cfg["MLFLOW"]
            del self.cfg["NUM_GPUS"]
            self.cfg.INPUT.MIN_SIZE_TRAIN = (self.cfg.INPUT.MIN_SIZE_TRAIN, )
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
            f.write(self.cfg.dump())








# class MLFlowLogger(HookBase):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg.clone()

#     def before_train(self):
#         with torch.no_grad():
#             # mlflow.set_tracking_uri(self.cfg.LOGGER.TRACKING_URI)
#             mlflow.set_experiment(self.cfg.LOGGER.EXPERIMENT_NAME)
#             mlflow.start_run(run_name=self.cfg.LOGGER.RUN_NAME)
#             mlflow.set_tag("mlflow.note.content", self.cfg.LOGGER.RUN_DESCRIPTION)
#             flattened_dict = flatten_dict(self.cfg)
#             for k, v in flattened_dict.items():
#                 mlflow.log_param(k, v)

#     def after_step(self):
#         with torch.no_grad():
#             latest_metrics = self.trainer.storage.latest()
#             for k, v in latest_metrics.items(): 
#                 mlflow.log_metric(key=k, value=v[0], step=v[1])

#     def after_train(self):
#         with torch.no_grad():
#             with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
#                 f.write(self.cfg.dump())
#             # mlflow.log_artifacts(self.cfg.OUTPUT_DIR)
#             mlflow.end_run()




class WANDBLogger(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

    def before_train(self):
        wandb.init(project=self.cfg.LOGGER.EXPERIMENT_NAME, 
                  name=self.cfg.LOGGER.RUN_NAME) if not wandb.run else wandb.run


    def after_step(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            for key, value in latest_metrics.items():
                wandb.log({key: value[0]}, step=value[1])

    def after_train(self):
        with torch.no_grad():
            with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
                f.write(self.cfg.dump())
            # mlflow.log_artifacts(self.cfg.OUTPUT_DIR)
            wandb.finish()

