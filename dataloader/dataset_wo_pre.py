import os
import numpy as np
from typing import *
import json
from sklearn.preprocessing import OneHotEncoder
import json
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch
import tqdm
from collections import defaultdict

import sys
sys.path.append("..")
from .categories import categories
from .dataloader_tokenize import tokenizer
from .embedding_generator import embedding_generator

class InstructSG():
    def __init__(self, config, data_path: str, process_node_feature_method:str=None) -> None:
        '''
        data_path: the path for data loader to load datat
        process_node_feature_method: either 'sentence', 'onehot', 'tokenize'. Default 'sentence', will group the node features into one sentence. Set to 'onehot' to obtain pytorch tensor of onehot encoded node features. Set to 'tokenize' to tokenize the features, return node features as tensors
        '''
        # load class variables from function arguments
        self.config = config
        self.data_path = data_path
        self.len_data = self.config.dataset_size
        self.show_progress_bar = True
        
        # set node feature processing properties for padding
        self.max_node_number_sg = config.max_node_number_sg
        self.max_edge_number_sg = self.max_node_number_sg
        self.max_node_feature_number = config.max_node_feature_number
        self.max_sequence_length = config.max_sequence_length
        
        # set types of action, color, objects
        self.action_list = ['move', 'pick', 'place_to', 'finish']
        self.color_dict = {"": 0, "red": 1, "green": 2, "blue": 3}
        self.object_dict = self._get_category()
        
        # set the method node features is processed by 
        process_node_feature_keys = ['sentence', 'onehot', 'tokenize']
        if process_node_feature_method in process_node_feature_keys:
            self.process_node_feature_method = process_node_feature_method
        else:
            self.process_node_feature_method = process_node_feature_keys[0]
        
        # set tokenizer if the method is to tokenize
        if self.process_node_feature_method=='tokenize':
            # The tokenizer will pre-process node features as well as instructions 
            self.tokenizer = tokenizer(text_encoder_type=config.text_encoder_type,
                                       max_node_number_sg=self.max_node_number_sg,
                                       max_node_feature_number=self.max_node_feature_number,
                                       max_seq_length=self.max_sequence_length)

        # self.preprocessed_language_flag = self.config.preprocessed_language
        # if self.preprocessed_language_flag:
        #     self.embedding_generator = embedding_generator(config=config)
        #     self.instruct_output_value = 'sentence_embedding'
        
        # Load data
        self._load_data()
        
        return

    def _get_category(self):
        object_dict = {"": 0}
        count = 1
        for var_type in categories:
            for var in categories[var_type]:
                object_dict.update({var: count})
                count += 1
        return object_dict

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], \
                                               Tuple[torch.Tensor, torch.Tensor], \
                                               str, \
                                               np.ndarray, \
                                               np.ndarray, \
                                               np.ndarray]:
        '''
            Return 
                Tuple: robot_graph[index] = (Torch.Tensor: node_features, Torch.Tensor: edge_idx, Torch.Tensor: edge_mask)
                Tuple: scene_graph[index] = (Torch.Tensor: node_features, Torch.Tensor: edge_idx, Torch.Tensor: edge_mask)
                str: instruct[index]
                np.ndarray: encoded_action[index] 
                np.ndarray: encoded_object_id: 
                    zeros: if the action name of current index is 'place' or 'finish'.
                    self.encoded_object_id[index]: otherwise
                np.ndarray: object_id_mask
                    boolean mask with the length of maximum number of nodes 
        '''
        # if self.preprocessed_language_flag:
        #     return  { 'robot_graph': {   
        #                 'token_embeddings' :    self.robot_graph_emb['token_embeddings'][index], 
        #                 'attention_mask' :      self.robot_graph_emb['attention_mask'][index], 
        #                 'sentence_embedding':   self.robot_graph_emb['sentence_embedding'][index],
        #                 'edge_index':           self.robot_graph[index][1],
        #                 'edge_index_mask':      self.robot_graph[index][2]
        #             }, 'scene_graph': {
        #                 'token_embeddings':     self.scene_graph_emb['token_embeddings'][index], 
        #                 'attention_mask' :      self.scene_graph_emb['attention_mask'][index],
        #                 'sentence_embedding':   self.scene_graph_emb['sentence_embedding'][index],
        #                 'edge_index':           self.scene_graph[index][1],
        #                 'edge_index_mask':      self.scene_graph[index][2]
        #             }, 'instruct': {
        #                 'token_embeddings':     self.instruct_emb['token_embeddings'][index], 
        #                 'attention_mask' :      self.instruct_emb['attention_mask'][index],
        #                 'sentence_embedding':   self.instruct_emb['sentence_embedding'][index]
        #             }}, \
        #             self.encoded_action[index].squeeze(), \
        #             self.encoded_object_id[index].squeeze(), \
        #             self.object_id_mask[index].squeeze()
        # else:
        return  self.robot_graph[index], \
                self.scene_graph[index], \
                self.instruct[index], \
                self.encoded_action[index].squeeze(), \
                self.encoded_object_id[index].squeeze(), \
                self.object_id_mask[index].squeeze()
         
    def __len__(self) -> int:
        return self.len_data

    # def _get_action_id(self, action_name:str) ->int:
    #     '''
    #         Return an action id
    #     '''
    #     action_dict = {act : idx for idx, act in enumerate(self.action_list)}
    #     return action_dict[action_name]

    # def _get_action(self, encoded_action_id:np.ndarray) -> str:
    #     '''
    #         Return the decoded action name in string type 
    #     '''
    #     return self.action_encoder.inverse_transform(encoded_action_id)

    def _load_data(self) -> None:
        '''
            Load data from self.data_path.
            Write to the following class variables
                robot_graph: List[dict] 
                scene_graph: List[dict] 
                instruct: List[str] 
                raw_object_label: List[str] 
                raw_object_id: List[int] 
                raw_action_id: List[int] 
                encoded_object_id: List[np.ndarray]
                encoded_action: List[np.ndarray]
        '''
        # create action encoder
        self.action_encoder = OneHotEncoder(categories=[self.action_list], sparse_output=False)
        self.object_id_encoder = OneHotEncoder(sparse_output=False, categories=[[i for i in range(self.max_node_number_sg)]])
        # Raw inputs
        in_rg = []
        in_sg = []
        in_instruct = []
        # Raw outputs
        raw_object_label = []
        raw_object_id = []
        # raw_action_id = []
        # processed output
        encoded_object_id = []
        encoded_action = []
        num_of_nodes = []
        scene_id_of_idx = []
        object_id_mask = []
        count_len_data = 0

        scene_ids = [f.path.split('.')[1] for f in os.scandir(self.data_path) if f.is_dir()]
        break_outer_loop = False
        # Iterate through scenes
        # for scene_id in scene_ids:
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
                    
                    # Encode sg rg based on requirement
                    in_sg.append(self._get_graph_data(sg_json))
                    in_rg.append(self._get_graph_data(rg_json))
                    
                    if self.process_node_feature_method=='tokenize':
                        in_instruct.append(self.tokenizer.tokenize(cmd_h, pad_to_length=self.max_sequence_length))
                    else:
                        in_instruct.append(cmd_h)

                    raw_action = cmd_l[j].split()[0]
                    operate_node_id = int(cmd_l[j].split()[-1])
                    opetate_object_label = cmd_l[j].split()[1:-1]
                    
                    num_of_nodes.append(len(sg_json['nodes']))
                    raw_object_label.append(opetate_object_label)
                    raw_object_id.append(operate_node_id)

                    # Encode action
                    encoded_action.append(self.action_encoder.fit_transform([[raw_action]]).astype(np.float32))

                    # Add scene ID for each index of objects
                    scene_id_of_idx.append(scene_id)
                    
                    # Add object ID mask
                    obj_mask = np.full((self.max_node_number_sg), False)
                    obj_id_encoded = np.zeros((self.max_node_number_sg,), dtype=np.float32)

                    if raw_action != 'finish':
                        # Mask object to the number of nodes of current scene
                        obj_mask[:num_of_nodes[-1]] = True
                        # Encode object id for each operating nodes at current scene
                        # Note that it encodes the node id instead of object label name
                        obj_id_encoded = self.object_id_encoder.fit_transform([[operate_node_id]]).astype(np.float32)
                    else:
                        # obj_mask is all False, saying no prediction for object
                        # TODO obj_id_encoded is zeros, but should it be zeros? Or encode as usual?
                        pass
                    encoded_object_id.append(obj_id_encoded)
                    object_id_mask.append(obj_mask)

                    # The maximum length of the input dataset
                    count_len_data += 1
                    if self.len_data != 0 and self.len_data == count_len_data:
                        break_outer_loop = True
                        break
                if break_outer_loop:
                    break
            if break_outer_loop:
                break


        # Record the length of data processed. if len_data is zero, use load all data
        self.len_data = count_len_data if self.len_data == 0 else self.len_data
        
        # change the config if the given dataset size is zero
        self.config.dataset_size = self.len_data 

        # number of nodes in scene graph
        self.num_of_nodes = num_of_nodes
        

        self.robot_graph: List[Tuple[dict, torch.Tensor, torch.Tensor]] = in_rg 
        self.scene_graph: List[Tuple[dict, torch.Tensor, torch.Tensor]] = in_sg 
        self.instruct: List[Union[str,dict]] = in_instruct
        
        self.raw_object_label: List[str] = raw_object_label # label of the object being manipulate 
        self.raw_object_id: List[int] = raw_object_id # unique node_ID of the object being manipulate in the scene graph
        
        self.encoded_object_id: List[np.ndarray] = encoded_object_id # One hot encoded raw object ID
        self.encoded_action: List[np.ndarray]   = encoded_action # One hot encoded raw action name
    
        self.object_id_mask: np.ndarray = np.array(object_id_mask)
        return 

    # def preprocess_text(self, in_rg, in_sg, in_instruct):
    #     instruct_tokens = self.stack_dictionary(dict_list=in_instruct)
    #     rg_tokens = self.stack_dictionary(tuple_list=in_rg)
    #     sg_tokens = self.stack_dictionary(tuple_list=in_sg)

    #     for start_index in tqdm.trange(0, len(rg_tokens), self.config.batch_size, desc= "Batches", disable=True, leave=False):
    #         self.robot_graph_emb = self.embedding_generator.generate_embeddings(rg_tokens[start_index:start_index+self.config.batch_size], 
    #                                                                             show_progress_bar=self.show_progress_bar, 
    #                                                                             progress_bar_desc='robot graph batches')
    #         self.scene_graph_emb = self.embedding_generator.generate_embeddings(sg_tokens[start_index:start_index+self.config.batch_size], 
    #                                                                             show_progress_bar=self.show_progress_bar, 
    #                                                                             progress_bar_desc='scene graph batches')
    #         self.instruct_emb = self.embedding_generator.generate_embeddings(instruct_tokens[start_index:start_index+self.config.batch_size], 
    #                                                                         show_progress_bar=self.show_progress_bar, 
    #                                                                         progress_bar_desc='Instruction batches')
    #         self.instruct_emb['token_embeddings'] = self.instruct_emb['token_embeddings'].squeeze(1)
    #         self.instruct_emb['attention_mask'] = ~self.instruct_emb['attention_mask'].bool().squeeze(1)
    #         self.instruct_emb['sentence_embedding'] = self.instruct_emb['sentence_embedding'].squeeze(1)
        
    #         class_idx = list(range(start_index, start_index+self.config.batch_size)) 
    #         for index in range(0, self.config.batch_size):
    #             sample = {'input': {   
    #                                 'robot_graph': 
    #                                 {   
    #                                 'token_embeddings' :    self.robot_graph_emb['token_embeddings'][index], 
    #                                 'attention_mask' :      self.robot_graph_emb['attention_mask'][index], 
    #                                 'sentence_embedding':   self.robot_graph_emb['sentence_embedding'][index],
    #                                 'edge_index':           self.robot_graph[class_idx[index]][1],
    #                                 'edge_index_mask':      self.robot_graph[class_idx[index]][2]
    #                                 }, 
    #                                 'scene_graph': 
    #                                 {
    #                                     'token_embeddings':     self.scene_graph_emb['token_embeddings'][index], 
    #                                     'attention_mask' :      self.scene_graph_emb['attention_mask'][index],
    #                                     'sentence_embedding':   self.scene_graph_emb['sentence_embedding'][index],
    #                                     'edge_index':           self.scene_graph[class_idx[index]][1],
    #                                     'edge_index_mask':      self.scene_graph[class_idx[index]][2]
    #                                 }, 
    #                                 'instruct': 
    #                                 {
    #                                     'token_embeddings':     self.instruct_emb['token_embeddings'][index], 
    #                                     'attention_mask' :      self.instruct_emb['attention_mask'][index],
    #                                     'sentence_embedding':   self.instruct_emb['sentence_embedding'][index]
    #                                 },
    #                     'output':{
    #                             'encoded_action': self.encoded_action[class_idx[index]].squeeze(),
    #                             'encoded_object_id': self.encoded_object_id[class_idx[index]].squeeze(),
    #                             'object_id_mask': self.object_id_mask[class_idx[index]].squeeze()
    #                         }}
    #                     }
    #             self.embedding_generator.save(sample)


    def stack_dictionary(self, dict_list=None, tuple_list=None):
        # Example list of dictionaries with tensor fields
        # dict_list = [{'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])},
        #             {'a': torch.tensor([7, 8, 9]), 'b': torch.tensor([10, 11, 12])},
        #             {'a': torch.tensor([13, 14, 15]), 'b': torch.tensor([16, 17, 18])}]

        # Create a defaultdict to store the stacked dictionary
        stacked_dict = defaultdict(list)

        # Stack the tensors in lists
        if dict_list:
            for d in dict_list:
                for key, value in d.items():
                    stacked_dict[key].append(value.unsqueeze(0))
        if tuple_list:
            for t in tuple_list:
                for key, value in t[0].items():
                    stacked_dict[key].append(value.unsqueeze(0))

        # Concatenate the tensors along the new dimension
        for key, value in stacked_dict.items():
            stacked_dict[key] = torch.cat(value, dim=0)

        # Convert the defaultdict to a regular dictionary
        stacked_dict = dict(stacked_dict)

        return stacked_dict

    def _get_node_features(self, graph_type:str, node_type:str, attributes:dict) -> List[int]:
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

    def _get_graph_data(self, json_data:dict) -> Tuple[Union[torch.Tensor, list], torch.Tensor, torch.Tensor]:
        """Process graph data to node features and edge indices based on self.process_node_feature_method

        Args:
            json_data (dict): A sample of robot graph or scene graph loaded from disk

        Returns:
            node_feature (Union[torch.Tensor, list]): 
                if self.process_node_feature_method is 'onehot', return a tensor of onehot encoded node feature; 
                
                elif self.process_node_feature_method is 'sentence' return a list of strings, 
                
                elif self.process_node_feature_method is 'tokenize', return a list of tokens tokenized by self.tokenizer 
            
            edge_index (torch.Tensor):

            edge_idx_mask (torch.Tensor):
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
            node_feature = [self._get_node_features(graph_type, n["type"], n["attributes"]) for n in nodes]
            node_feature = torch.tensor(node_feature, dtype=torch.float)
            # pad scene graph vertically to maximum number of node
            if graph_type == 'floor':
                # pad_edge_flag = True
                node_feature = torch.nn.functional.pad(node_feature, 
                                                    (0, 0, 0, 
                                                        self.max_node_number_sg-node_feature.size(0)))

        # get sentence of node features 
        elif self.process_node_feature_method=='sentence':
            # node feature is a list of strings, each column is a string of feature 
            num_node = 0
            for node in json_data['nodes']:
                num_node += 1
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
            # pad node features
            if graph_type == 'floor':
                feas = ['']
                node_feature += feas*(self.max_node_number_sg - num_node)
                pad_edge_flag = True
                    
        # tokenize node feature
        elif self.process_node_feature_method=='tokenize':
            feas = []
            for node in json_data['nodes']:                
                # obtain sentence of features
                if lowercase:
                    fea=(' ').join([node['type'].replace('_', ' ').strip().lower(),
                                    node['attributes']['color'].replace('_', ' ').strip().lower(),
                                    node['attributes']['label'].replace('_', ' ').strip().lower()]).strip()
                else:
                    fea=(' ').join([node['type'].replace('_', ' ').strip(),
                                    node['attributes']['color'].replace('_', ' ').strip(),
                                    node['attributes']['label'].replace('_', ' ').strip()]).strip()
                feas.append(fea)
            if graph_type == 'floor':
                feas += ['']*(self.max_node_number_sg - len(feas))
                pad_edge_flag=True
            # tokenize features
            node_feature = self.tokenizer.tokenize(feas,pad_to_length=self.max_node_feature_number,return_dict=True)
            
        else:
            raise('Unrecognized node feature processing method')
            
        # obtain edge index
        edge_index = [[int(e["source"]), int(e["target"])] for e in edges]
        
        # convert edge index to torch tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # pad edge index to the padded number of node 
        if pad_edge_flag:
            edge_idx_mask = torch.zeros(2,self.max_edge_number_sg)
            edge_idx_mask[:,edge_index.size(1)] = 1
            edge_index=torch.nn.functional.pad(edge_index, (0,self.max_edge_number_sg-edge_index.size(1),0,0))
        else:
            edge_idx_mask = torch.ones_like(edge_index)
            
        return node_feature, edge_index, edge_idx_mask
