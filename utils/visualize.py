import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import torch
import pandas as pd
from tabulate import tabulate
import textwrap
from dataloader.data_preprocessor import data_preprocessor as data_pre

from .logging import get_logger
logging = get_logger(__name__)

def get_confusion_matrix(y_true, y_pred):
    ''' 
    y_true: contains the ground truth labels
    y_pred: contains the predicted labels
    '''
    return confusion_matrix(y_true.numpy(), y_pred.numpy())

def configure_confusion_matrix_plot(cm, classes, show_plot=False, normalize=False, save_path=None):
    '''
    cm: confusion matrix
    show_plot: set to True to show plot on screen
    classes: List of class labels e.g. [0,1,2]
    normalize: Set to True if you want to normalize the confusion matrix
    save_path: Save the plot to the path specified
    '''
    plt.figure(figsize=(15, 15))  # Set the figure size.

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # write x,y axis text
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):        
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.title('Confusion Matrix')
    if save_path:
        pltname = os.path.basename(save_path).split('.')[0]
        plt.title(pltname)
        plt.savefig(save_path)
    if show_plot:
        plt.show()



def plot_confusion_matrix(parent_directory, batch_idx, action_id, action_hat_id, object_id_excluded, object_hat_id_excluded):
    # plot confusion matrix
    action_cm = get_confusion_matrix(action_id.cpu(), action_hat_id.cpu())
    object_cm = get_confusion_matrix(object_id_excluded.cpu(), object_hat_id_excluded.cpu())
    
    if os.path.isdir(parent_directory) is False:
        logging.warning(f'Cannot write confusion matrices to disk as parent directory {parent_directory} does not exist')
        return

    #save confusion matrices into dictionaries with the batch index as keys
    action_cms = {}
    object_cms = {}
    action_cms[batch_idx] = action_cm
    object_cms[batch_idx] = object_cm

    # print(object_cms)

    action_cm_0 = action_cms[batch_idx]
    object_cm_0 = object_cms[batch_idx]

    plt.clf()  # clear the current figure
    plt.cla()  # clear the current axes

    save_path_action = os.path.join(parent_directory, f'action_confusion_matrix_{batch_idx}.png')
    configure_confusion_matrix_plot(action_cm_0, np.sort(np.unique(action_id.cpu().numpy())), save_path=save_path_action)
    logging.info(f'Write {save_path_action} to disk.')

    plt.clf()  # clear the current figure
    plt.cla()  # clear the current axes

    save_path_object = os.path.join(parent_directory, f'object_confusion_matrix_{batch_idx}.png')
    configure_confusion_matrix_plot(object_cm_0, np.sort(np.unique(object_id_excluded.cpu().numpy())), save_path=save_path_object)
    logging.info(f'Write {save_path_object} to disk.')

def print_table(object_hat, object, action_hat, action, raw_data):
    # Get predicted results
    object_id_pre = object_hat.argmax(1)
    object_id_gt = object.argmax(1)
    action_id_pre = action_hat.argmax(1)
    action_id_gt = action.argmax(1)
    
    # Get the indices when the action is finish
    specific_value = data_pre.action_encoder.transform('finish').to(action.device)
    finsih_indices = torch.all(action == specific_value, dim=1)

    # Get the failed predictions
    failed_predictions_acc = action_id_pre != action_id_gt
    failed_predictions_obj = object_id_pre != object_id_gt
    all_failed_predictions = failed_predictions_acc + (failed_predictions_obj * ~finsih_indices)
    # Exclude finish actions
    failed_indices = torch.nonzero(all_failed_predictions).squeeze()
    if failed_indices.numel() == 0:
        logging.info('All predictions are true')
        return
    # gathering data
    data = {}
    data['object_predict'] = object_id_pre[failed_indices].cpu().numpy()
    data['object_groudtruth'] = object_id_gt[failed_indices].cpu().numpy()
    data['action_predict'] = action_id_pre[failed_indices].cpu().numpy()
    data['action_groudtruth'] = action_id_gt[failed_indices].cpu().numpy()
    
    data["scene_id"] = raw_data['scene_id'][failed_indices].cpu().numpy()
    data["instr_id"] = raw_data['instr_id'][failed_indices].cpu().numpy()
    data["graph_id"] = raw_data['graph_id'][failed_indices].cpu().numpy()
    
    path = {}
    path['instruct_file_path'] = [raw_data['instruct_file_path'][i] for i in range(len(raw_data['instruct_file_path'])) if i in failed_indices]
    path['sg_path'] = [raw_data['sg_file_path'][i] for i in range(len(raw_data['sg_file_path'])) if i in failed_indices]
    path['rg_path'] = [raw_data['rg_file_path'][i] for i in range(len(raw_data['rg_file_path'])) if i in failed_indices]
    data['path'] = ['\n'.join([path[key][i] for key in path.keys()]) for i in range(data["scene_id"].size)]
    # convert data to pandas DataFrame
    df = pd.DataFrame.from_dict(data)
    # print the DataFrame
    headers = [textwrap.fill(text=col, width=6) for col in df.columns]
    print(tabulate(df, headers=headers))
    # print(df)
    return data