from transformers import AutoTokenizer
import numpy as np
from typing import *
import torch

class tokenizer():
    def __init__(self, 
                 text_encoder_type:str, 
                 max_node_feature_number:int = 10,
                 max_node_number_sg:int = 100,
                 max_seq_length:int = 100) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=text_encoder_type)
        self.max_node_feature_number = max_node_feature_number # each number represents a field, eg. a node has a non-empty colour attribute,the number of node feature is 1
        self.max_node_number_sg = max_node_number_sg
        self.max_seq_length = max_seq_length


    def tokenize(self, texts:Union[list,str,np.ndarray], pad_to_length:int=None, return_dict:bool=True) -> Union[dict, Tuple[torch.Tensor, torch.Tensor]] :
        """Tokenize the input texts into a list of integers, masks are provided. The end of the input texts is append with a token=1

        Args:
            texts (Union[list,str,np.ndarray]): The input text to be tokenized into input_ids (also called tokens) and attention masks
            pad_to_length (int, optional): The length of the tokens to be padded to. Defaults to None.
            return_dict (bool, optional): Set true to return everything in a dictionary, set false to return each variables. Defaults to True.
        Returns:
            tokenized Union[dict, Tuple[torch.Tensor, torch.Tensor]]: The input text tokens and attention masks
        """
        if isinstance(texts, np.ndarray):
            to_tokenize = texts.squeeze()
            assert to_tokenize.ndim==1, f'The dimension of text {to_tokenize.ndim} is greater than 1'
            to_tokenize = to_tokenize.tolist()

        elif isinstance(texts, list):
            to_tokenize = np.array(texts).squeeze()
            assert to_tokenize.ndim==1, f'The dimension of text {to_tokenize.ndim} is greater than 1'
            to_tokenize = to_tokenize.tolist()
    
        elif isinstance(texts, tuple):
            to_tokenize = np.asanyarray(texts).squeeze()
            assert to_tokenize.ndim==1, f'The dimension of text {to_tokenize.ndim} is greater than 1'
            to_tokenize = to_tokenize.tolist()

        elif isinstance(texts, str):
            to_tokenize = [texts]
                    
        else:
            raise(f'text type {type(texts)} is not supported')
        
        if pad_to_length:
            # Pad token to a fixed length
            tokenized = self.tokenizer(to_tokenize, padding='max_length', truncation='longest_first', return_tensors="pt", max_length=pad_to_length)
        else: 
            # concatenate the string to the maximum sequence length
            tokenized = self.tokenizer(to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length)
        
        if return_dict:
            return tokenized.data
        else:
            return tokenized.data['input_ids'], tokenized.data['attention_mask']