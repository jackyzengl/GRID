import os
import numpy as np
from typing import *
import json
# from sklearn.preprocessing import OneHotEncoder
from .onehot_encoder import custom_onehot_encoder
# from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import Data
import torch
import tqdm
from collections import defaultdict
import copy
import sys
sys.path.append("..")
from .categories import categories as categories_
from .categories import actions as actions_
# from data_generation.categories_ten import categories
from .dataloader_tokenize import tokenizer
from .embedding_generator import embedding_generator
import h5pickle as h5p
import logging
from utils.logging import get_logger
logger=get_logger(__name__, logging.INFO)


def get_dict(input_list:list)->dict:
    assert isinstance(input_list, list), "input_list must be a list."
    # Assign empty string as the first element
    output_dict = {"": 0}
    # Assign keys from input_list, this avoids the repeated assignment
    output_dict.update(dict.fromkeys(input_list))
    # Assign values according to the index of the key
    output_dict = {key:idx for idx,key in enumerate(output_dict.keys())}
    return output_dict

class data_preprocessor():
    # set types of action, color, objects
    action_list = list(actions_.keys())
    color_dict = get_dict(categories_['color'])
    object_dict = get_dict(categories_['label'])
    save_type_keys = ['torch', 'hdf5']
    
    # Create action encoder
    action_encoder = custom_onehot_encoder(action_list)
    
    # the built-in pre-processing methods for node features and instructions
    process_node_feature_keys = ['sentence', 'onehot', 'tokenize']
    process_instruction_keys = ['sentence', 'tokenize']

    def __init__(self, arg, config) -> None:
        # load class variables from function arguments
        self.config = config
        self.arg = arg
        
        self.len_data = self.config.get('dataset_size')
        self.show_progress_bar = self.config.get('preprocessor_show_progress_bar')
        self.text_encoder_type = self.config.get('text_encoder_type')
        self.preprocessed_language_flag = self.config.get('preprocessed_language') 
        self.process_node_feature_method = self.set(self.config.get('process_node_feature_method', self.process_node_feature_keys[-1]), self.process_node_feature_keys, -1)
        self.process_instruction_method = self.set(self.config.get('process_instruction_method', self.process_node_feature_keys[-1]), self.process_node_feature_keys, -1)
        self.save_type = self.set(self.config.get('save_type', self.save_type_keys[0]), self.save_type_keys, 0) #  保存模式设置
        self.save_compress_flag = self.config.get('save_compress_flag', True)
        self.save_dir = self.arg.preprocessed_data_path

        # set node feature processing properties for padding
        self.max_node_number_sg = self.config.get('max_node_number_sg')
        self.max_edge_number_sg = self.max_node_number_sg
        self.max_node_feature_number = self.config.get('max_node_feature_number')
        self.max_sequence_length = self.config.get('max_sequence_length')
        

        # Generate file name for hdf5 file
        if self.save_type == 'hdf5':
            self.save_filename = 'preprocessed_data'
            # Creates a unique file name in output directory and open the file
            self.makefile(self.save_filename)
       
        
        # Set pre-processing methods and language model
        if self.preprocessed_language_flag:
            # Tokenize texts if we preprocess texts with language model
            self.process_node_feature_method = 'tokenize'
            self.process_instruction_keys = 'tokenize'
            
            # Initialize language model to generate embeddings
            self.embedding_generator = embedding_generator(arg=arg, config=self.config)


        # Initialize tokenizer if it is required
        if self.process_node_feature_method=='tokenize' or self.process_instruction_keys=='tokenize':
            # The tokenizer will pre-process node features as well as instructions 
            self.tokenizer = tokenizer(text_encoder_type=self.text_encoder_type,
                                       max_node_number_sg=self.max_node_number_sg,
                                       max_node_feature_number=self.max_node_feature_number,
                                       max_seq_length=self.max_sequence_length)

        # create object encoder
        self.object_id_encoder = custom_onehot_encoder([i for i in range(self.max_node_number_sg)])
        # self.object_id_encoder = OneHotEncoder(sparse_output=False, categories=[[i for i in range(self.max_node_number_sg)]])
        

    def preprocess_from_file(self, data_path: str) -> None:
        """prerpocess from a series of raw dataset file.

        Args:
            data_pah (str): the path for data loader to load datat
        """
        self.data_path = data_path

        # Create folder to load preprocessed data
        self.makedir(self.save_dir)

        # Reset count for data saving
        self.__reset_count()

        # Load data
        self.len_data, X = self.__load_data()
        
        # Unpack the loaded data into class variables 
        self.robot_graph = X['robot_graph']
        self.scene_graph = X['scene_graph']
        self.instruct = X['instruct']
        self.encoded_object_id = X['encoded_object_id']
        self.encoded_action = X['encoded_action']
        # self.object_id_mask = X['object_id_mask']
        # For debug purpose
        self.raw_data_path = X['raw_data_path']

        # Change the config if the given dataset size is zero
        self.config.dataset_size = self.len_data 

        # Save the data
        if self.preprocessed_language_flag:
            # Preprocess graph feature texts and instruction sentences, save data to disk
            self.__preprocess_text_with_lm(self.robot_graph, self.scene_graph, self.instruct)
        else:
            # Save raw data to disk without preprocessing
            self.__save_wo_pre(compress=self.save_compress_flag)
    
    def preprocess_input_once(self, raw_data: list) -> dict:
        """_summary_

        Args:
            raw_data (dict): [rg,sg,inst]

        Returns:
            dict: preprocessed_data
        """
        in_rg = []
        in_sg = []
        in_rg.append(self.__preprocess_graph_data(raw_data[0]))
        in_sg.append(self.__preprocess_graph_data(raw_data[1]))
        self.robot_graph = in_rg
        self.scene_graph = in_sg
        in_instruct=raw_data[2]
        in_instruct = self.tokenizer.tokenize(in_instruct, pad_to_length=self.max_sequence_length)
        for key, value in in_instruct.items():
            in_instruct[key] = value.unsqueeze(1)
        # Change the config if the given dataset size is zero
        self.config.dataset_size = 1

        # Save the data
        if self.preprocessed_language_flag:
            # Preprocess graph feature texts and instruction sentences, save data to disk
            preprocessed_data = self.__preprocess_text_with_lm(in_rg, in_sg, in_instruct, predict_mode=True)
        else:
            # Save raw data to disk without preprocessing
            self.__save_wo_pre(compress=self.save_compress_flag)
        
        preprocessed_data = {key1: 
                                {key2: 
                                    {key3: 
                                    torch.unsqueeze(value3, 0).to(preprocessed_data['input']['robot_graph']['sentence_embedding'].device) if type(value3) == torch.Tensor else value3  
                                    for key3,value3 in value2.items()
                                    }  
                                for key2,value2 in value1.items()
                                }
                            for key1,value1 in preprocessed_data.items()
                            }

        return preprocessed_data

    def __del__(self):
        """
        Called when the class is going to be destroy
        """        
        if self.save_type == 'hdf5':
            # Close the file
            self.h5p_file.close()

    def set(self, value:Any, preset_values:list, default_position:int=0)->Any:
        """checks if the value in is the preset values. If it is not, set to the default value

        Args:
            value (Any): The input value that should be one of the preset_values,
            if it is not in the preset_values, the default value which is one of the pre-set values will be returned
            preset_values (list): All possible values the input value can be
            default_position (int, optional): The position of present_value to be returned as default value. Defaults to 0.

        Returns:
            Any: the value itself
        """        
        if value in preset_values:
            return value
        else:
            logger.warning(f'{value} is not one member of the built-in keys {preset_values}. Set to default {preset_values[default_position]}')
            return preset_values[default_position]



    def __load_data(self) -> dict:
        """Load all data from the data path specified. Preprocess data if required.

        Returns:
            dict: A dictionary containing all raw data and pre-processed data that is required in the following steps
        """        
        # Raw inputs
        in_rg = []
        in_sg = []
        # in_sg = {}
        in_instruct = []
        # Raw outputs
        raw_object_label = []
        raw_object_id = []
        # processed output
        encoded_object_id = []
        encoded_action = []
        num_of_nodes = []
        scene_id_of_idx = []
        object_id_mask = []
        raw_data_path = []
        
        count_len_data = 0
        scene_ids = [int(f.path.split('.')[1]) for f in os.scandir(self.data_path) if f.is_dir()]
        scene_ids.sort()
        break_outer_loop = False
        # Iterate through scenes
        # Create a progress bar for the outer loop
        for scene_id in tqdm.tqdm(scene_ids, desc='Loading Scene', position=0, disable=not self.show_progress_bar):
            instruct_file = os.path.join(self.data_path, f'scene.{scene_id}.instr.json')
            instr_json = json.load(open(instruct_file, 'r'))
            
            # Iterate through high-level instructions
            for i in tqdm.trange(len(instr_json['commands']), desc='Loading Instructions', position=1, leave=False, disable=not self.show_progress_bar):
                instr_id = instr_json['commands'][i]['id']
                cmd_l = instr_json['commands'][i]['low']
                cmd_h = instr_json['commands'][i]['high']
                
                # Iterate through low-level instructions
                for j in tqdm.trange(len(cmd_l), desc='low-level-cmd ID', position=2, leave=False, disable=not self.show_progress_bar):
                    sg_json = json.load(open(os.path.join(self.data_path, f'scene.{scene_id}.graphs', f'scene.{scene_id}.instr.{instr_id}.sg.{j}.json'), 'r'))
                    rg_json = json.load(open(os.path.join(self.data_path, f'scene.{scene_id}.graphs', f'scene.{scene_id}.instr.{instr_id}.rg.{j}.json'), 'r'))
                    raw_data_path.append({'scene_id': scene_id,
                                          'instr_id': instr_id,
                                          'graph_id': j,
                                          'instruct_file_path': instruct_file,
                                          'sg_file_path': os.path.join(self.data_path, f'scene.{scene_id}.graphs', f'scene.{scene_id}.instr.{instr_id}.sg.{j}.json'),
                                          'rg_file_path': os.path.join(self.data_path, f'scene.{scene_id}.graphs', f'scene.{scene_id}.instr.{instr_id}.rg.{j}.json')
                                          })
                    
                    # Encode sg rg based on requirement
                    # node_feature, node_index_mask, edge_index, edge_idx_mask = self.__preprocess_graph_data(sg_json)
                    in_sg.append(self.__preprocess_graph_data(sg_json))
                    in_rg.append(self.__preprocess_graph_data(rg_json))
                    
                    in_instruct.append(cmd_h)

                    raw_action = cmd_l[j].split()[0]
                    operate_node_id = int(cmd_l[j].split()[-1])
                    opetate_object_label = cmd_l[j].split()[1:-1]
                    
                    num_of_nodes.append(len(sg_json['nodes']))
                    raw_object_label.append(opetate_object_label)
                    raw_object_id.append(operate_node_id)

                    # Encode action
                    encoded_action.append(self.action_encoder.transform(raw_action))
                    # encoded_action.append(self.action_encoder.fit_transform([[raw_action]]).astype(np.float32))

                    # Add scene ID for each index of objects
                    scene_id_of_idx.append(scene_id)
                    
                    # Add object ID mask
                    obj_mask = np.full((self.max_node_number_sg), False)
                    obj_id_encoded = torch.zeros((self.max_node_number_sg,), dtype=torch.float32)

                    if raw_action != 'finish':
                        # Mask object to the number of nodes of current scene
                        obj_mask[:num_of_nodes[-1]] = True
                        # Encode object id for each operating nodes at current scene
                        # Note that it encodes the node id instead of object label name
                        obj_id_encoded = self.object_id_encoder.transform(operate_node_id)
                        # obj_id_encoded = self.object_id_encoder.fit_transform([[operate_node_id]]).astype(np.float32)
                    else:
                        # obj_mask is all False, saying no prediction for object
                        pass
                    encoded_object_id.append(obj_id_encoded)
                    object_id_mask.append(obj_mask)

                    # The maximum length of the input dataset
                    count_len_data += 1
                    if self.len_data > 0 and self.len_data == count_len_data:
                        logger.critical('\nBreak loading due to the restricted datasize.')
                        break_outer_loop = True
                        break
                if break_outer_loop:
                    break
            if break_outer_loop:
                break

        if self.process_instruction_method=='tokenize':
            in_instruct = self.tokenizer.tokenize(in_instruct, pad_to_length=self.max_sequence_length)
            for key, value in in_instruct.items():
                in_instruct[key] = value.unsqueeze(1)

        # Record the length of data processed. if len_data is zero, use load all data
        len_data = count_len_data if self.len_data <= 0 else min(count_len_data, self.len_data)
        
        return len_data, {
            'robot_graph' : in_rg,
            'scene_graph' : in_sg,
            'instruct' : in_instruct,
            'raw_object_label' : raw_object_label,
            'raw_object_id' : raw_object_id,
            'encoded_object_id' : encoded_object_id,
            'encoded_action' : encoded_action,
            'object_id_mask' : np.array(object_id_mask),
            'raw_data_path' : raw_data_path
        }

    def __preprocess_text_with_lm(self, in_rg:list, in_sg:list, instruct_tokens:dict, predict_mode:bool = False):
        """Preprocess the texts from robot graph, scene graph and instruction by obtaining the text embeddings with language model.
        Save the processed messages to disk.

        Args:
            in_rg (list): contains all robot graphs that are tokenized. Each row is a dictionary of token of the sample.
            in_sg (list): contains all scene graphs that are tokenized. Each row is a dictionary of token of the sample.
            instruct_tokens (dict): a dictionary of tokens of all samples. Fields are stacked already.
        """        
        if predict_mode:
            # Stack the node features field
            rg_tokens = self.collate_dict(in_rg, 'node_feature')
            sg_tokens = self.collate_dict(in_sg, 'node_feature')
            
            # Save embedding and all other variables required by the network into files, one sample per file. 
            robot_graph_emb = self.embedding_generator.generate_embeddings(data_preprocessor.batch_token(rg_tokens, 0, self.config.batch_size), 
                                                                            show_progress_bar=self.show_progress_bar,
                                                                            position=1, 
                                                                            progress_bar_desc='robot graph batches')
            scene_graph_emb = self.embedding_generator.generate_embeddings(data_preprocessor.batch_token(sg_tokens, 0, self.config.batch_size), 
                                                                            show_progress_bar=self.show_progress_bar, 
                                                                            position=1,
                                                                            progress_bar_desc='scene graph batches')
            instruct_emb = self.embedding_generator.generate_embeddings(data_preprocessor.batch_token(instruct_tokens, 0, self.config.batch_size), 
                                                                        show_progress_bar=self.show_progress_bar, 
                                                                        position=1,
                                                                        progress_bar_desc='Instruction batches')
            # Invert the boolean mask for pytorch transformer
            robot_graph_emb['attention_mask'] = ~robot_graph_emb['attention_mask'].bool()
            scene_graph_emb['attention_mask'] = ~scene_graph_emb['attention_mask'].bool()
            
            # Squeeze instruction embeddings
            instruct_emb['attention_mask'] = ~instruct_emb['attention_mask'].bool().squeeze(1)
            instruct_emb['token_embeddings'] = instruct_emb['token_embeddings'].squeeze(1)
            instruct_emb['sentence_embedding'] = instruct_emb['sentence_embedding'].squeeze(1)

            # iterate through the class variables and embedded variables used in current batch  
            sample = self.__get_item__(class_idx=0, 
                                    current_idx=0, 
                                    robot_graph_emb=robot_graph_emb, 
                                    scene_graph_emb=scene_graph_emb, 
                                    instruct_emb=instruct_emb,
                                    predict_mode=True)
            return sample
                
        else:
            # Stack the node features field
            rg_tokens = self.collate_dict(in_rg, 'node_feature')
            sg_tokens = self.collate_dict(in_sg, 'node_feature')
            
            # Save embedding and all other variables required by the network into files, one sample per file. 
            for start_index in tqdm.trange(0, len(in_rg), self.config.batch_size, desc= "Embedding Batches", disable=False, leave=False, position=0):
                robot_graph_emb = self.embedding_generator.generate_embeddings(data_preprocessor.batch_token(rg_tokens, start_index, self.config.batch_size), 
                                                                                show_progress_bar=self.show_progress_bar,
                                                                                position=1, 
                                                                                progress_bar_desc='robot graph batches')
                scene_graph_emb = self.embedding_generator.generate_embeddings(data_preprocessor.batch_token(sg_tokens, start_index, self.config.batch_size), 
                                                                                show_progress_bar=self.show_progress_bar, 
                                                                                position=1,
                                                                                progress_bar_desc='scene graph batches')
                instruct_emb = self.embedding_generator.generate_embeddings(data_preprocessor.batch_token(instruct_tokens, start_index, self.config.batch_size), 
                                                                            show_progress_bar=self.show_progress_bar, 
                                                                            position=1,
                                                                            progress_bar_desc='Instruction batches')
                # Invert the boolean mask for pytorch transformer
                robot_graph_emb['attention_mask'] = ~robot_graph_emb['attention_mask'].bool()
                scene_graph_emb['attention_mask'] = ~scene_graph_emb['attention_mask'].bool()
                
                # Squeeze instruction embeddings
                instruct_emb['attention_mask'] = ~instruct_emb['attention_mask'].bool().squeeze(1)
                instruct_emb['token_embeddings'] = instruct_emb['token_embeddings'].squeeze(1)
                instruct_emb['sentence_embedding'] = instruct_emb['sentence_embedding'].squeeze(1)
                # Get the indices of class variables of the current batch operation
                end_index = min(self.len_data, start_index+self.config.batch_size)
                class_idx = list(range(start_index, end_index))

                # iterate through the class variables and embedded variables used in current batch  
                for index in tqdm.trange(0, len(class_idx), desc='Save current batch', disable=False, leave=False, position=1):
                    sample = self.__get_item__(class_idx=class_idx[index], 
                                            current_idx=index, 
                                            robot_graph_emb=robot_graph_emb, 
                                            scene_graph_emb=scene_graph_emb, 
                                            instruct_emb=instruct_emb)
                    logger.debug(sample['raw_data_path'])
                    logger.debug(f"encoded_action_id: {self.action_encoder.inverse_transform(sample['output']['encoded_action'])}")
                    logger.debug(f"encoded_object_id: {self.object_id_encoder.inverse_transform(sample['output']['encoded_object_id'])}")
                    self.save(sample, compress=self.save_compress_flag)
            

    def __get_item__(self, class_idx, current_idx=None, robot_graph_emb=None, scene_graph_emb=None, instruct_emb=None, predict_mode:bool=False) -> dict:
        """ Get a sample
        Args:
            current_idx (int): index of the network output
            class_idx (int): index of the class variables

        Returns:
            (dict): A sample that contains all tensors required by the network
        """    
        if self.preprocessed_language_flag:   # Return the language model processed output
            if predict_mode:
                    return {
                        'input': {  'robot_graph': {   
                                        'token_embeddings' :    robot_graph_emb['token_embeddings'][current_idx].clone(), 
                                        'attention_mask' :      robot_graph_emb['attention_mask'][current_idx].clone(), 
                                        'sentence_embedding':   robot_graph_emb['sentence_embedding'][current_idx].clone(),
                                        'edge_index':           self.robot_graph[class_idx]['edge_index'].clone(),
                                        'edge_index_mask':      self.robot_graph[class_idx]['edge_index_mask'].clone(),
                                        'node_index_mask':      self.robot_graph[class_idx]['node_index_mask'].clone()
                                    }, 
                                    'scene_graph': {
                                        'token_embeddings':     scene_graph_emb['token_embeddings'][current_idx].clone(), 
                                        'attention_mask' :      scene_graph_emb['attention_mask'][current_idx].clone(),
                                        'sentence_embedding':   scene_graph_emb['sentence_embedding'][current_idx].clone(),
                                        'edge_index':           self.scene_graph[class_idx]['edge_index'].clone(),
                                        'edge_index_mask':      self.scene_graph[class_idx]['edge_index_mask'].clone(),
                                        'node_index_mask':      self.scene_graph[class_idx]['node_index_mask'].clone()
                                    }, 
                                    'instruct': {
                                        'token_embeddings':     instruct_emb['token_embeddings'][current_idx].clone(), 
                                        'attention_mask' :      instruct_emb['attention_mask'][current_idx].clone(),
                                        'sentence_embedding':   instruct_emb['sentence_embedding'][current_idx].clone()
                                    }
                                }
                    }
            else:
                return {
                        # For debug purpose
                        'raw_data_path': self.raw_data_path[class_idx],
                        'input': {  'robot_graph': {   
                                        'token_embeddings' :    robot_graph_emb['token_embeddings'][current_idx].clone(), 
                                        'attention_mask' :      robot_graph_emb['attention_mask'][current_idx].clone(), 
                                        'sentence_embedding':   robot_graph_emb['sentence_embedding'][current_idx].clone(),
                                        'edge_index':           self.robot_graph[class_idx]['edge_index'].clone(),
                                        'edge_index_mask':      self.robot_graph[class_idx]['edge_index_mask'].clone(),
                                        'node_index_mask':      self.robot_graph[class_idx]['node_index_mask'].clone()
                                    }, 
                                    'scene_graph': {
                                        'token_embeddings':     scene_graph_emb['token_embeddings'][current_idx].clone(), 
                                        'attention_mask' :      scene_graph_emb['attention_mask'][current_idx].clone(),
                                        'sentence_embedding':   scene_graph_emb['sentence_embedding'][current_idx].clone(),
                                        'edge_index':           self.scene_graph[class_idx]['edge_index'].clone(),
                                        'edge_index_mask':      self.scene_graph[class_idx]['edge_index_mask'].clone(),
                                        'node_index_mask':      self.scene_graph[class_idx]['node_index_mask'].clone()
                                    }, 
                                    'instruct': {
                                        'token_embeddings':     instruct_emb['token_embeddings'][current_idx].clone(), 
                                        'attention_mask' :      instruct_emb['attention_mask'][current_idx].clone(),
                                        'sentence_embedding':   instruct_emb['sentence_embedding'][current_idx].clone()
                                    }
                                },
                        # 'output':{  'encoded_action':           torch.from_numpy(self.encoded_action[class_idx].squeeze()).clone(),
                        #             'encoded_object_id':        torch.from_numpy(self.encoded_object_id[class_idx].squeeze()).clone()
                        #         }
                        'output':{  'encoded_action':           self.encoded_action[class_idx].clone(),
                                    'encoded_object_id':        self.encoded_object_id[class_idx].clone()
                                }
                }
        else: 
            if self.process_instruction_method == 'tokenize': # Return the tokens
                instruct = {     # tokenized instructions  
                                'input_ids':            self.instruct['input_ids'][class_idx].clone(),
                                'attention_mask':       self.instruct['attention_mask'][class_idx].clone()
                            }  
            elif self.process_instruction_method == 'sentence': 
                # raw text 
                instruct = self.instruct[class_idx] if self.save_type != 'torch' else torch.tensor([self.instruct[class_idx]])
            
            robot_graph = {   
                            'edge_index':           self.robot_graph[class_idx]['edge_index'].clone(),
                            'edge_index_mask':      self.robot_graph[class_idx]['edge_index_mask'].clone(),
                            'node_index_mask':      self.robot_graph[class_idx]['node_index_mask'].clone()
                        }
            scene_graph = {   
                            'edge_index':           self.scene_graph[class_idx]['edge_index'].clone(),
                            'edge_index_mask':      self.scene_graph[class_idx]['edge_index_mask'].clone(),
                            'node_index_mask':      self.scene_graph[class_idx]['node_index_mask'].clone()
                        }
            if self.process_node_feature_method == 'tokenize':
                robot_graph['node_feature'] = {   # tokenized node features
                                                'input_ids':            self.robot_graph[class_idx]['node_feature']['input_ids'].clone(),
                                                'attention_mask':       self.robot_graph[class_idx]['node_feature']['attention_mask'].clone()
                                            }
                scene_graph['node_feature'] = {    # tokenized node features
                                                'input_ids':            self.scene_graph[class_idx]['node_feature']['input_ids'].clone(),
                                                'attention_mask':       self.scene_graph[class_idx]['node_feature']['attention_mask'].clone()
                                            }
            elif self.process_node_feature_method == 'sentence':
                robot_graph['node_feature'] = torch.tensor([self.robot_graph[class_idx]['node_feature']]).clone()
                scene_graph['node_feature'] = torch.tensor([self.scene_graph[class_idx]['node_feature']]).clone()
            elif self.process_node_feature_method == 'onehot':
                robot_graph['node_feature'] = self.robot_graph[class_idx]['node_feature'].clone()
                scene_graph['node_feature'] = self.scene_graph[class_idx]['node_feature'].clone()
               
            # Group everything together
            return {
                    # For debug purpose
                    'raw_data_path': self.raw_data_path[class_idx],

                    'input': {  'robot_graph':              robot_graph, 
                                'scene_graph':              scene_graph,
                                'instruct':                 instruct         
                            },
                    
                    # 'output':{  'encoded_action':           torch.from_numpy(self.encoded_action[class_idx].squeeze()).clone(),
                    #             'encoded_object_id':        torch.from_numpy(self.encoded_object_id[class_idx].squeeze()).clone()
                    #         }
                    'output':{  'encoded_action':           self.encoded_action[class_idx].clone(),
                                'encoded_object_id':        self.encoded_object_id[class_idx].clone()
                            }
                    }
    

    def __preprocess_node_feature_onehot(self, graph_type:str, node_type:str, attributes:dict) -> List[int]:
        """ 
        create appropriate node features.
                
        Return 
            Node_features:
                List of One-Hot encoded features, features being encoded include node types and node colors
                
                The list returned is one-dimensional with the length of the sum of number of types and number
                of colors
        """
        if graph_type == 'robot':
            node_type_dict = {"":0, "robot": 1, "room": 2, "floor":3, "large_object": 4, "small_object": 5}
        else:
            node_type_dict = {"":0, "room": 1, "floor":2, "large_object": 3, "small_object": 4}
        
        # one-hot encoding for the node type and color.
        node_features = [0] * (len(node_type_dict) + len(self.color_dict) + len(self.object_dict))
        node_features[node_type_dict[node_type]] = 1
        node_features[len(node_type_dict) + self.color_dict[attributes["color"]]] = 1
        node_features[len(node_type_dict) + len(self.color_dict)+self.object_dict[attributes["label"]]]=1

        return node_features
    
    def __preprocess_graph_data(self, json_data:dict) -> dict:
        """Process graph data to node features and edge indices based on self.process_node_feature_method

        Args:
            json_data (dict): A sample of robot graph or scene graph

        Returns:
            dict: A dictionary that contains fields: 'node_feature', 'node_index_mask', 'edge_index', 'edge_index_mask'
        """        

        # Convert all words to lower case to process 
        lowercase = True

        # Extract nodes and edges from the JSON data.
        nodes = json_data["nodes"]
        edges = json_data["edges"]

        # Check if the graph is rg or sg: 'floor' means sg, 'robot' means rg
        graph_type = nodes[0]['type']
        pad_edge_flag = False # There is no need to padd edges, since not all nodes have to be connected
        
        node_feature = []
        # one hot encoded node fetures represented float type tensor
        if self.process_node_feature_method=='onehot':
            node_feature = [self.__preprocess_node_feature_onehot(graph_type, n["type"], n["attributes"]) for n in nodes]
            node_feature = torch.tensor(node_feature, dtype=torch.int32)
            
            # Get number of node of the graph before padding
            num_node = node_feature.size(0)
            
            # Create mask of the number of node of the graph
            node_index_mask = torch.ones(num_node, dtype=torch.int32)
            
            # pad scene graph vertically to maximum number of node
            if graph_type == 'floor':
                pad_edge_flag = True
                node_feature = torch.nn.functional.pad(node_feature, (0, 0, 0, self.max_node_number_sg-num_node))
                node_index_mask = torch.nn.functional.pad(node_index_mask, (0, self.max_node_number_sg-num_node))                

        # get sentence of node features 
        elif self.process_node_feature_method=='sentence':
            # node feature is a list of strings, each column is a string of feature 
            for node in json_data['nodes']:
                # group the features into a sentence, replace _ with space, remove excess space, use lower case
                if lowercase:
                    fea=(' ').join([node['type'].replace('_', ' ').strip().lower(),
                                    node['attributes']['color'].replace('_', ' ').strip().lower(),
                                    node['attributes']['label'].replace('_', ' ').strip().lower()]).strip()
                else:
                    fea=(' ').join([node['type'].replace('_', ' ').strip(),
                                    node['attributes']['color'].replace('_', ' ').strip(),
                                    node['attributes']['label'].replace('_', ' ').strip()]).strip()
                node_feature.append(fea)

            # Get number of node of the graph before padding
            num_node = len(node_feature)

            # Create mask of the number of node of the graph
            node_index_mask = torch.ones(num_node, dtype=torch.int32)
            
            #  pad node features and node index mask
            if graph_type == 'floor':
                feas = ['']
                node_feature += feas*(self.max_node_number_sg - num_node)
                pad_edge_flag = True
                node_index_mask = torch.nn.functional.pad(node_index_mask, (0, self.max_node_number_sg-num_node))

                    
        # tokenize node feature
        elif self.process_node_feature_method=='tokenize':
            feas = []
            for node in json_data['nodes']:                
                # obtain sentence of features
                if lowercase: # convert to lower case
                    fea=(' ').join([node['type'].replace('_', ' ').strip().lower(),
                                    node['attributes']['color'].replace('_', ' ').strip().lower(),
                                    node['attributes']['label'].replace('_', ' ').strip().lower()]).strip()
                else: # leave it as it is
                    fea=(' ').join([node['type'].replace('_', ' ').strip(),
                                    node['attributes']['color'].replace('_', ' ').strip(),
                                    node['attributes']['label'].replace('_', ' ').strip()]).strip()
                feas.append(fea)
            
            # Get number of node of the graph before padding
            num_node = len(feas)

            # Create mask of the number of node of the graph
            node_index_mask = torch.ones(num_node, dtype=torch.int32)
            
            # tokenize features
            node_feature = self.tokenizer.tokenize(feas,pad_to_length=self.max_node_feature_number,return_dict=True)
            
            # Pad node tensors
            if graph_type == 'floor':
                node_feature['input_ids'] = torch.cat([node_feature['input_ids'], 
                                                       torch.zeros((self.max_node_number_sg - num_node, self.max_node_feature_number), 
                                                                   dtype=node_feature['input_ids'].dtype)], 
                                                       dim=0)
                node_feature['attention_mask'] = torch.cat([node_feature['attention_mask'], 
                                                            torch.zeros((self.max_node_number_sg - num_node, self.max_node_feature_number), 
                                                                        dtype=node_feature['attention_mask'].dtype)], 
                                                            dim=0)
                pad_edge_flag=True
                node_index_mask = torch.nn.functional.pad(node_index_mask, (0, self.max_node_number_sg-num_node))

            # Convert node feature types
            node_feature['input_ids'] = node_feature['input_ids'].to(torch.int32)
            node_feature['attention_mask'] = node_feature['attention_mask'].to(torch.int32)
               
        else:
            raise AssertionError('Unrecognized node feature processing method')
            
        # obtain edge index
        edge_index = [[int(e["source"]), int(e["target"])] for e in edges]
        
        # convert edge index to torch tensor
        edge_index = torch.tensor(edge_index).t().contiguous()
        
        # pad edge index to the padded number of node 
        if pad_edge_flag:
            edge_idx_mask = torch.zeros(2,self.max_edge_number_sg)
            edge_idx_mask[:, :edge_index.size(1)] = 1
            edge_index=torch.nn.functional.pad(edge_index, (0,self.max_edge_number_sg-edge_index.size(1),0,0))
        else:
            edge_idx_mask = torch.ones_like(edge_index)
            
        return {
            'node_feature':             node_feature,
            'node_index_mask':          ~node_index_mask.bool(),
            'edge_index':               edge_index.to(torch.int32),
            'edge_index_mask':          ~edge_idx_mask.bool()
        }

    def __count(self)->int:
        """Step the count index once called

        Returns:
            int: the number of count
        """        
        self.__save_count += 1
        return self.__save_count
    
    def __reset_count(self):
        """Reset the number of count to zero
        """        
        self.__save_count = 0

    def save(self, sample:dict, compress:bool=True):
        """Save the sample to disk according to the save type (self.save_type). 
            If self.save_type is 'torch', the function saves the sample to a '{sample_number}.pt' file with the number of count as file name,
            one saved file per sample.  
            
            If self.save_type is 'hdf5', the function appends the sample to a new group in the opened hdf5 file with a name of 'sample_{sample_number}', 
            the hieriachy of the sample group is the same as the input sample dictionary 
            all samples will be saved to a single file.

        Args:
            sample (dict): A single sample to be saved to disk 
            compress (bool, optional): Define if the saved file is compressed. Defaults to True.
        """        
        if self.save_type=='torch':
            output_filename = os.path.join(self.save_dir, f'{self.__count()}.pt')
            torch.save(sample, output_filename, _use_new_zipfile_serialization=compress)
        else:
            # save the sample dictionary to an HDF5 file
            sample_group = self.h5p_file.create_group(f'sample_{self.__count()}')
            self.recursive_save(sample_group, sample)
            # input_group = sample_group.create_group('input')
            # for type, value in sample['input'].items():
            #     sub_group = input_group.create_group(type)
            #     for field, data in value.items():
            #         sub_group.create_dataset(field, data=data.cpu(), compression="gzip")

            # # create datasets for the output data
            # output_group = sample_group.create_group('output')
            # for key, value in sample['output'].items():
            #     output_group.create_dataset(key, data=value, compression="gzip")
        
    def recursive_save(self, group:Union[h5p.File, h5p.Group], input:dict):
        """Recursively save the dictionary to hdf5 object

        Args:
            group (Union[h5p.File, h5p.Group]): The parent group for the current data to be saved to
            input (dict): The nested input
        """        
        if isinstance(value, dict):
            for key, value in input.items():
                if isinstance(value, dict):
                    sub_group = group.create_group(key)
                    self.recursive_save(sub_group, value)
                else:
                    self.recursive_save(group, value)
        elif isinstance(value, torch.Tensor):
            group.create_dataset(key, data=value.cpu(), compression="gzip")
        else:
            group.create_dataset(key, data=value, compression="gzip")
            

    def __save_wo_pre(self, compress:bool=True):
        """Write raw class variables to disk, the class variables to be saved are specified in __get_item__() function

        Args:
            compress (bool, optional): Set True to compress the files to be saved. Defaults to True.
        """        
        for i in range(self.len_data):
            sample = self.__get_item__(current_idx=None, class_idx=i)
            self.save(sample, compress=compress)

    @staticmethod
    def batch_token(token:dict, start_id:int, batch_size:int) -> dict:
        """Get a batch of tokens from all tokens

        Args:
            token (dict): A dictionary of tokens of all samples
            start_id (int): The starting index of the batch
            batch_size (int): Size of the batch

        Returns:
            dict: tokens of the current batch of samples
        """            
        return { key: token[key][start_id:start_id+batch_size] for key in token.keys()
            # 'input_ids': token['input_ids'][start_id:start_id+batch_size],
            # 'attention_mask': token['attention_mask'][start_id:start_id+batch_size]
        }

    @staticmethod
    def makedir(output_dir:str):
        """Generate directory to save files if it does not exist

        Args:
            output_dir (str): The directory to hold output files
        """        
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        except Exception as e:
            logger.error(e)

    def makefile(self, output_filename:str) -> None:
        """Creates a unique filename in the output directory without extension

        Args:
            output_filename (str): the file name to be saved
        """        
        target_dir = os.path.join(self.save_dir, output_filename+'.h5')
        if os.path.exists(target_dir):
            fn_split = output_filename.split('_')
            if fn_split[-1].isnumeric():
                new_idx = int(fn_split[-1])+1
                new_output_filename = '_'.join(fn_split[:-1]) + f'_{new_idx}'
            else:
                new_idx = 1
                new_output_filename = output_filename + f'_{new_idx}'
            self.file_exists_flag = True
            self.makefile(new_output_filename)
        else:
            self.save_filename = output_filename+'.h5'
            self.output_file_dir = target_dir
            self.h5p_file = h5p.File(self.output_file_dir, 'w')
            if hasattr(self, 'file_exists_flag'):
                logger.warning(f'File exists, rename it to {output_filename}')
                self.file_exists_flag = False

    @staticmethod
    def collate_dict(input:list, position=None) -> dict:
        """ Stack the tensors of the same fields in lists of dictionarie together

        Args:
            input (list): Either a list of dictionary or a list of tuple of dictionary to stacked
            position (int, optional): Specify the nested component to stack

        Returns:
            stacked_dict (dict): the dictionary after stacking
        
        Example: 
            input list of dictionaries with tensor fields
            
            input = [{'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])},  
                    {'a': torch.tensor([7, 8, 9]), 'b': torch.tensor([10, 11, 12])},  
                    {'a': torch.tensor([13, 14, 15]), 'b': torch.tensor([16, 17, 18])}]  
            stacked_dict = {'a': torch.tensor([[1, 2, 3], [7, 8, 9], [13, 14, 15]]),  
                            'b': torch.tensor([[4, 5, 6], [10, 11, 12], [16, 17, 18])}
        """        
        # Create a defaultdict to store the stacked dictionary
        stacked_dict = defaultdict(list)

        # Stack the tensors in lists
        if isinstance(input[0], dict):
            if not position:
                # stack all fields of the dictionary in the input
                for d in input:
                    for key, value in d.items():
                        stacked_dict[key].append(value.unsqueeze(0))
            else:
                # stack all fields of the nested dictionary at the specified position of the input dictionary
                if not isinstance(input[0][position], dict):
                    raise AssertionError('The nested component given is not a dictionary')
                for d in input:
                    for key, value in d[position].items():
                        stacked_dict[key].append(value.unsqueeze(0))
        elif isinstance(input[0], tuple):
            # Stack all fields of the dictionary at the given position of input tuple
            position = position if position else 0
            if not isinstance(input[0][position], dict):
                raise AssertionError('The nested component given is not a dictionary')
            for t in input:
                for key, value in t[position].items():
                    stacked_dict[key].append(value.unsqueeze(0))

        # Concatenate the tensors along the new dimension
        for key, value in stacked_dict.items():
            stacked_dict[key] = torch.cat(value, dim=0)

        # Convert the defaultdict to a regular dictionary
        stacked_dict = dict(stacked_dict)

        return stacked_dict
    
