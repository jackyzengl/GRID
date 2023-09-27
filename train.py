from typing import Any, Optional
import math
import torch
import random
from torch import optim, nn
from torchmetrics import Accuracy
import torch.nn.functional as F
import time
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.distributed as dist
from dataloader.dataset import InstructSG_Dataset
from dataloader.dataset_wo_pre import InstructSG
import warnings
import numpy as np
from utils import visualize as vis
from utils import timer
from multiprocessing import set_start_method
import matplotlib.pyplot as plt
import os
from dataloader.data_preprocessor import data_preprocessor as data_pre
from lightning.pytorch.profilers import PyTorchProfiler as PTProfiler
from utils.logging import get_logger
from utils import get_accuracy
from arguments import create_parser
import logging

# Create logger 
printer = get_logger(logger_name='main',logging_level=logging.DEBUG, save_dir="debug.log")
# Ignore warnings
warnings.filterwarnings('ignore')
# Create argument and configuration parser
args_, config_ = create_parser('trainer')
# Path to save accuracy
save_accuracy_path_ = "accuracy.xlsx" # "accuracy.txt"


class GRIDLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_loss_scale = config.action_loss_scale
        self.object_loss_scale = config.object_loss_scale
        self.action_loss_func = nn.CrossEntropyLoss()
        self.object_loss_func = nn.CrossEntropyLoss()

        # Set if enable L1 L2 norm
        self.enable_l1 = config.enable_l1
        self.enable_l2 = config.enable_l2
        self.weight_decay = config.weight_decay
        self.L1_param = config.l1_param
        self.L2_param = config.l2_param

    def forward(self, action_hat, object_hat, obj_mask, action, object):
        object_hat = object_hat * ~obj_mask
        
        action_loss = self.action_loss_func(action_hat, action)
        object_loss = self.object_loss_func(object_hat, object)

        loss = self.action_loss_scale * action_loss + self.object_loss_scale * object_loss

        # L2
        if self.enable_l2:
            L2_reg_loss = 0.
            for w in self.parameters():
                L2_reg_loss += torch.sum(w ** 2)
            L2_reg_loss *= self.weight_decay
            
            loss += self.L2_param*L2_reg_loss

        # L1
        if self.enable_l1:
            L1_reg_loss=0.
            for w in self.parameters():
                L1_reg_loss+=torch.sum(torch.abs(w))
            L1_reg_loss*=self.weight_decay
            
            loss += self.L1_param*L1_reg_loss

        return loss, action_loss, object_loss

class LitGRID(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.language_model == "bert":
            from GRID.models.GRID_bert import GRID_bert
            self.grid = GRID_bert()
        else: # default lm is instructor
            from GRID.models.GRID_instructor import GRID_instructor
            self.grid = GRID_instructor(config)

        self.loss = GRIDLoss(config)
        self.dataset_size = config.dataset_size
        self.batch_size = config.batch_size
        self.max_lr = config.max_lr
        self.action_acc_func = Accuracy(task="multiclass", num_classes=config.num_action)
        self.object_acc_func = Accuracy(task="multiclass", num_classes=100) # max node_num
        self.action_cms = []
        self.object_cms = []
        self.predict_step_outputs = []

    def forward(self, batch):
        return self.grid(batch)

    def training_step(self, batch, batch_idx):
        action = batch['output']['encoded_action']
        object = batch['output']['encoded_object_id']
        obj_mask = batch['input']['scene_graph']['node_index_mask']
        
        action_hat, object_hat = self.grid(batch)
        loss, action_loss, object_loss = self.loss(action_hat, object_hat, obj_mask, action, object) 

        # Calculate metrics
        action_acc, object_acc= self.cal_metrices(action_hat, object_hat, action, object)

        # Only support epoch-level log
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_action_loss", action_loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_object_loss", object_loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_action_acc", action_acc, on_step=False, on_epoch=True, logger=True)
        if object_acc != None:
            self.log("train_object_acc", object_acc, on_step=False, on_epoch=True, logger=True)
            self.log("train_total_acc", action_acc * object_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Model forward
        action = batch['output']['encoded_action']
        object = batch['output']['encoded_object_id']
        obj_mask = batch['input']['scene_graph']['node_index_mask']

        action_hat, object_hat = self.grid(batch)
        loss, action_loss, object_loss = self.loss(action_hat, object_hat, obj_mask, action, object)

        # Calculate metrics
        action_acc, object_acc= self.cal_metrices(action_hat, object_hat, action, object)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_action_loss", action_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_object_loss", object_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_action_acc", action_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if object_acc != None:
            self.log("val_object_acc", object_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_total_acc", action_acc * object_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx):
        
        obj_mask = batch['input']['scene_graph']['node_index_mask']

        # Model forward
        action_hat, object_hat = self.grid(batch)

        if "output" in batch:
            action = batch['output']['encoded_action']
            object = batch['output']['encoded_object_id']

            loss, action_loss, object_loss = self.loss(action_hat, object_hat, obj_mask, action, object)

            # Calculate metrics
            action_acc, object_acc, action_hat_id, action_id, object_hat_id_excluded, object_id_excluded = self.cal_metrices(action_hat, object_hat, action, object, if_predict=True)

            printer.debug(f"val_loss={loss}")
            printer.debug(f"action_loss={action_loss}")
            printer.debug(f"object_loss={object_loss}")
            printer.debug(f"action_acc={action_acc}")
            
            if object_id_excluded.numel()!=0:
                printer.debug(f"object_acc={object_acc}")
            
            # Append necessary results to predict_step_outputs
            self.predict_step_outputs.append({
                "action_hat_id": action_hat_id,
                "action_id": action_id,
                "object_hat_id_excluded": object_hat_id_excluded,
                "object_id_excluded": object_id_excluded,
                "action_hat": action_hat,
                "object_hat": object_hat,
                "action": action,
                "object": object,
                # "raw_data_path": batch['raw_data_path']
                "scene_id": batch['raw_data_path']['scene_id'],
                "instr_id": batch['raw_data_path']['instr_id'],
                "graph_id": batch['raw_data_path']['graph_id']
            })

        return action_hat, object_hat
    
    def on_predict_epoch_end(self):
        # Gather all predictions
        all_preds = get_accuracy.collate_dict(self.predict_step_outputs)
        
        # Plot confusion matrix of all predictions 
        vis.plot_confusion_matrix(
                parent_directory=save_accuracy_path_, #os.path.dirname(save_accuracy_path_),
                batch_idx=0,
                action_id=all_preds["action_id"],
                action_hat_id=all_preds["action_hat_id"],
                object_id_excluded=all_preds["object_id_excluded"],
                object_hat_id_excluded=all_preds["object_hat_id_excluded"],
            )
        
        # Calculate overall accuracies
        metric = self.cal_metrices(all_preds['action_hat'], all_preds['object_hat'], all_preds['action'], all_preds['object'], get_all_metrics=True)
        
        # Save selected metrics to disk
        save_metric = {"action_acc": metric['action_acc'].item(),
                       "object_acc": metric['object_acc'].item(),
                       "tot_err": metric['tot_error_percentage'].item(),
                       "tot_acc": metric['tot_accuracy_percentage'].item(),
                       }
        save_label = {
            "scene_id":         all_preds['scene_id'].cpu().numpy(),
            "instr_id":         all_preds['instr_id'].cpu().numpy(),
            "graph_id":         all_preds['graph_id'].cpu().numpy(),
            "action_id_pre":    metric['action_id_pre'].cpu().numpy(),
            "action_id_gt":     metric['action_id_gt'].cpu().numpy(), 
            # Include all instances of object while
            # forcing the predicted value of finish action to 0
            "object_id_origin_pre":  metric['object_id_origin_pre'].cpu().numpy(),
            "object_id_origin_gt":   metric['object_id_origin_gt'].cpu().numpy(),
            # Get the labels
            "object_label":         metric["object_label"].cpu().numpy(),
            "action_label":         metric["action_label"].cpu().numpy(),
            "action_label":         metric["action_label"].cpu().numpy(),
            "label":                metric['label'].cpu().numpy()
        }
        get_accuracy.save_excel(os.path.dirname(save_accuracy_path_)+'/sub_task_label.xlsx', save_label)
        get_accuracy.save_excel(save_accuracy_path_, save_metric)

        # Clear predict_step_outputs to free memory
        self.predict_step_outputs.clear()


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=5e-3)
        steps_per_epoch = math.ceil(self.dataset_size*self.config.train_split_ratio/self.batch_size/self.trainer.accumulate_grad_batches/self.trainer.world_size)

        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                     max_lr=self.max_lr, 
                                                     epochs=self.trainer.max_epochs, 
                                                     steps_per_epoch=steps_per_epoch, 
                                                     div_factor=10, 
                                                     final_div_factor=1e4) 
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return {
                "optimizer": optimizer,
                "lr_scheduler":{
                    "scheduler": lr_scheduler,
                    "monitor": "val_loss",
                    "interval": "step",
                },
            }
    
    def cal_metrices(self, action_hat, object_hat, action, object, if_predict: bool = False, get_all_metrics: bool = False):
        """Calculate metrices

        Args:
            action_hat (tensor): act predict
            object_hat (tensor): obj predict
            action (tensor): act ground truth
            object (tensor): obj ground truth

        Returns:
            action_acc, object_acc: Accuracy
        """
        # Pad the onehot object predictions
        object_hat = F.pad(object_hat, (0, self.config.max_node_number_sg - object_hat.shape[1]), "constant", 0) 
        object_id_pre = object_hat.argmax(1)    # predicted object id
        object_id_gt = object.argmax(1)         # groundtruth object id
        action_id_pre = action_hat.argmax(1)    # predicted action id
        action_id_gt = action.argmax(1)         # groundtruth action id

        # The encoded finish action value
        specific_value = data_pre.action_encoder.transform('finish').to(action.device)
        
        # Find the indices to exclude based on the presence of the specific value
        indices_to_exclude = torch.all(action == specific_value, dim=1)

        # Get the failed predictions in boolean tensors
        failed_predictions_act = action_id_pre != action_id_gt
        failed_predictions_obj = object_id_pre != object_id_gt
        # Boolean tensor where true stands for failed cases, excluding the object prediction when action is finsih
        all_failed_predictions = failed_predictions_act + (failed_predictions_obj * ~indices_to_exclude)
        tot_error_percentage = all_failed_predictions.sum()/all_failed_predictions.numel()
        tot_accuracy_percentage = 1-tot_error_percentage

        # Exclude the object instance from predicted labels and ground truth labels tensors 
        # when the action is 'finish'. The resultant matrix has less number of elements 
        # than the original matrix
        object_hat_id_excluded = object_hat[~indices_to_exclude].argmax(dim=1)
        object_id_excluded = object[~indices_to_exclude].argmax(dim=1)

        # force the excluded value to be zeros
        object_hat[indices_to_exclude] = 0
        object_id_origin = object_hat.argmax(dim=1)

        # Accuracy calculation
        action_acc = self.action_acc_func(action_id_pre, action_id_gt)
        object_acc = None
        if object_id_excluded.numel()!=0:
            object_acc = self.object_acc_func(object_hat_id_excluded, object_id_excluded)

        if get_all_metrics:
            return {"action_acc":       action_acc, 
                    "object_acc":       object_acc, 
                    "action_id_pre":    action_id_pre,
                    "action_id_gt":     action_id_gt, 
                    # Predicted and groundtruth object id excluding the finish action instances
                    "object_hat_id_excluded":   object_hat_id_excluded, 
                    "object_id_excluded":       object_id_excluded,
                    # Include all instances of object while
                    # forcing the predicted value of finish action to 0
                    "object_id_origin_pre":   object_id_origin, 
                    "object_id_origin_gt":   object_id_gt, 
                    # Get the labels
                    "object_label":         (object_id_origin==object_id_gt).int(),
                    "action_label":         1 - failed_predictions_act.int(),
                    "label":        1 - all_failed_predictions.int(),
                    "tot_error_percentage":     tot_error_percentage,
                    "tot_accuracy_percentage":  tot_accuracy_percentage}
        if if_predict:
            return action_acc, object_acc, action_id_pre, \
            action_id_gt, object_hat_id_excluded, object_id_excluded
        else:
            return action_acc, object_acc

def main():
    time_f=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    experiment_name = "test"
    
    # Create Logger to save process
    logger = TensorBoardLogger(save_dir="logs", name=experiment_name)

    # Write hyperparameters to disk: This line copys the network parameter file to log dir
    logger.log_hyperparams({key: value for key, value in vars(config_).items() if key != 'configuration_path'})
    global save_accuracy_path_
    save_accuracy_path_ = os.path.join(logger.log_dir, save_accuracy_path_) 

    # Print variables
    printer.debug(f'experiment_name: {experiment_name}')
    for arg_name in vars(args_):
        arg_value = getattr(args_, arg_name)
        printer.debug(f'{arg_name}: {arg_value}')
    for key, value in vars(config_).items():
        printer.debug(f'{key}: {value}')
    
    # Dataset
    if config_.use_preprocessed_data:
        dataset = InstructSG_Dataset(config=config_, data_path=args_.preprocessed_data_path)
    else:
        dataset = InstructSG(config=config_, data_path=args_.data_path)

    # Define should we split training and testing sets   
    if config_.train_test_split_enable:
        # Define the splitted training size
        if args_.fit_flag:
            train_size = int(config_.train_split_ratio * len(dataset))
        else:
            train_size = 0

        indices = torch.arange(len(dataset))

        # Split train valid datasets
        train_indices, val_indices = indices[:train_size], indices[train_size:]
        
        # Load the training and testing dataloaders
        if args_.fit_flag:
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            train_loader = DataLoader(train_dataset, batch_size=config_.batch_size, shuffle=True, drop_last = False, num_workers=config_.num_workers)
        
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

    
        # No shuffle ensures that different models will be tested on the same data
        val_loader = DataLoader(val_dataset, batch_size=config_.batch_size, shuffle=False, drop_last = False, num_workers=config_.num_workers)
    
    else:
        # Define training and testing dataloaders to be the same
        train_loader = DataLoader(dataset, batch_size=config_.batch_size)

        # Validation set is the same as the training set
        val_loader = DataLoader(dataset, batch_size=config_.batch_size)
    
    # Model
    model = LitGRID(config_)

    # Define Callback functions for trainer
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_callback = ModelCheckpoint(save_top_k=1, 
                                    monitor="val_total_acc",
                                    mode="max", 
                                    every_n_epochs = 25, 
                                    filename='{epoch:02d}')

    # Create Profiler
    # profiler = PTProfiler(filename="test")

    # Create Trainer
    trainer = pl.Trainer(max_epochs=config_.max_epoch, 
                        accelerator=args_.accelerator, 
                        devices=args_.gpu_devices, 
                        logger=logger, 
                        callbacks=[lr_monitor, ckpt_callback],
                        strategy='ddp_find_unused_parameters_true')

    if args_.fit_flag: # train mode
        if args_.from_ckpt_flag:
            # Fit and predict from checkpoint
            trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader,
                ckpt_path=args_.ckpt_path
                )
        else:
            # Start new fitting if we are not predicting from checkpoint
            trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader
                )
    else:  # predict mode
        if args_.from_ckpt_flag:
            # predict from check point，no training
            out = trainer.predict(model=model, dataloaders=val_loader, 
                                ckpt_path=args_.ckpt_path)
        else:
            printer.error("Can not predict without ckpt!!")
            return

    # REQUIRES GRAPHVIZ INSTALLATION 
    if config_.visualize_network_flag:
        import torchviz
        # create plot object
        dot_action = torchviz.make_dot(torch.tensor(out[min(len(out), config_.vis_sample_id)][0]))
        dot_object = torchviz.make_dot(torch.tensor(out[min(len(out), config_.vis_sample_id)][1]))
        # visualize plot
        dot_action.view()
        dot_object.view()
        # Save the computation graph as a PNG image
        dot_action.render(logger.log_dir + "/action", format='png')
        dot_object.render(logger.log_dir + "/object", format='png')
    

if __name__ == "__main__":
    main()



