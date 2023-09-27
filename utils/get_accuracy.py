import pandas as pd
import numpy as np
import openpyxl
import math
from typing import *
import torch
from collections import defaultdict
import os

from .logging import get_logger
logging = get_logger(__name__)

# def collate_dict(data):
#     stacked = {}
#     if isinstance(data, list):
#         for d in data:
#             for k, v in d.items():
#                 if k in stacked:
#                     stacked[k] = torch.stack((stacked[k], v))
#                 else:
#                     stacked[k] = v
#     return stacked

def collate_dict(data, stacked = {}):
    if isinstance(data, list):
        for d in data:
            stacked = collate_dict(d)
    elif isinstance(data, dict):
        # If value is a dictionary, recursively stack
        for k, v in data.items():
            if k in stacked:
                stacked[k] = collate_dict(v, stacked[k])
            else:
                stacked[k] = v
    elif isinstance(data, torch.Tensor):
        # If value is a tensor, stack along the first dimension
        stacked = torch.cat((stacked, data), dim=0)
        # stacked = torch.cat(padding_stacked(stacked, data), dim=0)
    return stacked

def padding_stacked(tensor1, tensor2, pad_value=0):
    '''Pad the shorter tensor with pad_value which has defaults 0'''
    ## Example tensors
    # tensor1 = torch.tensor([1, 2, 3])
    # tensor2 = torch.tensor([4, 5, 6, 7, 8])
    ## Stack the output padded tensors
    # stacked_tensor = torch.stack([padded_tensor1, padded_tensor2])

    # Determine the sizes of the tensors
    size1 = tensor1.size(-1)
    size2 = tensor2.size(-1)
    
    # Return if sizes are the same
    if size1 == size2:
        return tensor1, tensor2
    
    while tensor1.ndim > tensor2.ndim:
        tensor2 = tensor2.unsqueeze(0)
    
    # Determine the size of the larger tensor
    max_size = max(size1, size2)

    # Pad the smaller tensor with zeros
    padded_tensor1 = torch.nn.functional.pad(tensor1, (0, max_size - size1), value=pad_value)  # Pad along dimension 0
    padded_tensor2 = torch.nn.functional.pad(tensor2, (0, max_size - size2), value=pad_value)  # Pad along dimension 0

    return padded_tensor1, padded_tensor2

def append_excel(file_path:str, data:dict):
    # Read the existing Excel file with header inferred
    df = pd.read_excel(file_path, engine='openpyxl')
    # Append a new column to the dataframe
    for k, v in data.items():
        df[k] = v
    # Overwrite the updated dataframe back to the Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as file:
        df.to_excel(file, index=False)
        logging.info(f'Overwrite {file_path}')
    

def save_excel(save_path:str, data:Union[list, dict]):
    """Save data to excel file

    Args:
        save_path (str): The directory to save the excel sheet
        data (Union[list, dict]): The data to save
    """    
    import pandas as pd
    parent_dir = os.path.dirname(save_path)
    if os.path.isdir(parent_dir) is False:
        logging.warning(f'Cannot write file {save_path} as directory {parent_dir} does not exist.') 
        return
    # create a pandas dataframe from the dictionary/list
    if isinstance(data, dict): 
        first_key = next(iter(data))
        if isinstance(data[first_key], list) or \
           isinstance(data[first_key], dict) or \
           isinstance(data[first_key], np.ndarray) or \
           isinstance(data[first_key], torch.Tensor):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data, index=[0])
    else:
        df = pd.DataFrame(data)
    # save the dataframe to an excel file
    df.to_excel(save_path)
    logging.info(f'Write data to disk: {save_path}')


def get_sheet_names(file_path:str)->list:
    """Get a list of sheet names from excel file

    Args:
        file_path (str): The directory to load the excel file

    Returns:
        sheet_names (list): A list of sheet names
    """    
    workbook = openpyxl.load_workbook(file_path)
    sheet_names = workbook.sheetnames
    return sheet_names

def load_excel(file_path:str, sheet_name:str="")->dict:
    """Load excel file to dictionary

    Args:
        file_path (str): the path to the excel file on disk
        sheet_name (str, optional): The name of sheet where we read data from. Defaults to "" to read from the first sheet.

    Returns:
        dict: the data loaded from the excel file
    """    
    sheet_name = get_sheet_names(file_path)[0] if not sheet_name else sheet_name

    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Convert the DataFrame to a dictionary
    data_dict = df.to_dict(orient='list')
    
    return data_dict

def cal_accuracy(label:list)->float:
    """Calculate the accuracy of a list labels

    Args:
        label (list): a list that labels whether or not the prediction is correct. 
        the data should be labelled by integer or float zeros and ones.

    Returns:
        float: accuracy of the data
    """    
    # The labeled data
    labeled_labels = [x for x in label if not (isinstance(x, float) and math.isnan(x))]
    if len(label) > len(labeled_labels):
        logging.warning('The data is not fully labeled.')
    if len(labeled_labels) == 0:
        logging.warning('Data is not labeled.')
        return 0.
    
    # Check if the labeling is correct
    unique_elements = list(set(labeled_labels))
    assert len(unique_elements)<=2, \
        f'The elements in the label is {unique_elements}, please make sure the data is labeled with integer 0 and 1'
    assert all(element == 0 or element == 1 for element in unique_elements), \
        f'The element {unique_elements} in the label is invalid, please make sure the data is labeled with integer 0 and 1'
    
    # Calculate accuracy
    acc = sum(labeled_labels)/len(labeled_labels)
    logging.info('acc='+str(acc))
    return acc

def custom_key(item):
    return (item['scene_id'], item['instr_id'], item['graph_id'])

def load_data_hierarachy(data_path)->dict:
        """Browse through the file hieriachy and get all ids
        Returns:
            dict: dict of dictionaries in structure {sceneid:{instr_id:{graph_id: False}}}
        """ 
        # Initialize sample id place holder
        # hierarchy = {}
        hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # Obtain scene_id, instr_id, graph_id 
        for graph_dir in os.listdir(data_path):
            path = os.path.join(data_path, graph_dir)
            if os.path.isdir(path):
                scene_id = int(graph_dir.split('.')[1])
                # graph_dir_sorted = sorted(graph_dir, key=sort_rg)
                for graph_name in os.listdir(path):
                    if (graph_name.split('.')[4] == 'rg'):
                        graph_id = int(graph_name.split('.')[5])
                        instr_id = int(graph_name.split('.')[3])
                        hierarchy[scene_id][instr_id][graph_id]=False
                        # hierarchy.append({'scene_id':scene_id, 'instr_id':instr_id, 'graph_id':graph_id})
        
        # Sort the sample according to id number
        hierarchy = sort_keys_nested(hierarchy)
        # hierarchy = sorted(hierarchy, key=custom_key)
        return hierarchy

def sort_keys_nested(dictionary):
    sorted_dict = {}
    for key in sorted(dictionary.keys()):
        value = dictionary[key]
        if isinstance(value, dict):
            sorted_dict[key] = sort_keys_nested(value)
        else:
            sorted_dict[key] = value
    return sorted_dict

def get_task_acc(data:dict, sample_hierarchy:dict) -> Tuple[dict, dict]:
    """Get overall task accuracy and task accuracy of different length of sub-tasks

    Args:
        data (dict): 
            the labelled prediction result. 
            This should be a dictionary read from an 
            excel sheet named 'sub_task_label'
            produced by grid network prediction  

        sample_hierarchy (dict): 
            The complete hierarchy of scene id, instr id and 
            graph id of the scene tested.  
            This hierarchy is used to look up as a dictionary 
            to check whether each current task in the predicted 
            data is accomplish.

    Returns:
        overall_task_accuracy (dict): 
            {
                'task_acc': task_acc (float), \n
                'num_finished_tasks': num_finished_tasks (int), \n
                'len_data': len_data (int)\n
            }
        subtask accuracy (dict): 
            {
                'task_acc_of_len':task_acc_of_len (float),\n
                'num_task_of_len':num_task_of_len (int),\n
                'sum_task_label':sum_task_label (float),\n
            }
    """

    # Get length of data
    len_data = len(data[next(iter(data))])
    # Sort data
    data_ids = [{'scene_id': data['scene_id'][i], 
                'instr_id': data['instr_id'][i], 
                'graph_id': data['graph_id'][i],
                'index': i} for i in range(len_data)]
    data_ids = sorted(data_ids, key=custom_key)
    
    # Initialize
    i = -1
    accomplised_task_labels = []
    not_finished_task = []
    finished_task = []

    # Loop through the dataset
    while True:
        i = i+1
        scene_id = data['scene_id'][data_ids[i]['index']]
        instr_id = data['instr_id'][data_ids[i]['index']]
        # graph_id = data[i]['graph_id']
        len_sub_task = len(sample_hierarchy[scene_id][instr_id])
        current_task_label = data['label'][data_ids[i]['index']]
        current_task_ids = [data_ids[i]]

        # Check if there is than one sub-tasks in the task 
        if len_sub_task >1:
            # Attempt to loop through the sub-task of the same task
            # same_task_flag = False
            for j in range(1, len_sub_task):
                i+=1
                # If the index is out of range, it is an exception circumstance, we skip it
                if i >= len_data:
                    logging.warning(f'Index {i} is out of range {len_data}, it is an exception circumstance')
                    break
                current_task_ids.append(data_ids[i])
                # If we are still on the same task, accumulate the task label
                if data['instr_id'][data_ids[i]['index']] == instr_id and \
                    data['scene_id'][data_ids[i]['index']] == scene_id:
                    current_task_label *= data['label'][data_ids[i]['index']]
                else:
                    # The this sub-task is not within same task, mark as unfinished 
                    current_task_label = -1
                    break

        if current_task_label != -1: # -1 stands for an incomplete task
            accomplised_task_labels.append(current_task_label)
            finished_task.append({'sub_task_ids': current_task_ids, 'task_label':current_task_label})
        else:
            not_finished_task.append(current_task_ids)
        # Break while loop
        if i >= len_data-1:
            break

    # calculate statistics
    num_finished_tasks = len(accomplised_task_labels)
    num_not_finished_tasks = len(not_finished_task)
    total_task_num = num_finished_tasks+num_not_finished_tasks

    # Summerize subtasks of different tasks in a dictionary
    diff_len_tasks = defaultdict(list)    
    for t in finished_task:
        diff_len_tasks[len(t['sub_task_ids'])].append(t)
    diff_len_tasks = dict(diff_len_tasks)

    # calculate task accuracy for tasks in different length
    sub_task_acc = {}
    for task_len, val in diff_len_tasks.items():
        sum_task_label = sum(l['task_label'] for l in val)
        num_task_of_len = len(val)
        task_acc_of_len = sum_task_label/num_task_of_len
        sub_task_acc[task_len] = {'task_acc':task_acc_of_len,
                                  'num_subtask':num_task_of_len,
                                  'num_successful_subtask':sum_task_label,
                                  # 'data_ids':val
                                }
                                
    # Calculate overall task accuracy
    if num_finished_tasks > 0:
        task_acc = sum(accomplised_task_labels)/num_finished_tasks
        logging.debug(f'Accomplished tasks number: {num_finished_tasks}/{total_task_num}')
    else:
        task_acc=0
        logging.warning('No tasks are completed.')
    if num_not_finished_tasks > 0:
        logging.debug(f'Unfinished tasks number: {num_not_finished_tasks}/{total_task_num}')
        logging.debug(pd.DataFrame(not_finished_task))
    
    logging.info(f'Task accuracy: {task_acc:.4f} over {len_data} samples')
        
    return {
            'task_acc': task_acc, 
            'num_finished_tasks': num_finished_tasks, 
            'len_data': len_data
            },  sub_task_acc


def save_txt(file_path:str, acc:dict, input_file_path:str=""):
    # Save accuracy to text file
    with open(file_path, 'w') as f:
        if input_file_path:
            f.write('Input file path: ' + input_file_path + '\n')
        for row in acc:
            f.write(row + ': ' + str(acc[row]) + "\n")


def cal_save_acc_for_gpt():
    file_path = 'chat_gpt_output_20230803-124355.xlsx'
    out_acc_file_path = f'{file_path.split(".")[0]}.acc.txt'
    data = load_excel(file_path)
    acc = cal_accuracy(data['label'])
    acc_dict = {'total_accuracy': acc}
    save_txt(out_acc_file_path, acc_dict, file_path)
