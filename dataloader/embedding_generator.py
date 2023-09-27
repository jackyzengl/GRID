from InstructorEmbedding import INSTRUCTOR
from .dataloader_tokenize import tokenizer
import torch
import json
import os

class embedding_generator():
    def __init__(self, arg, config, outputfile=None) -> None:
        self.config = config
        self.arg = arg
        self.output_filename = outputfile

        # Text Encoder
        self.lm = INSTRUCTOR(self.config.text_encoder_type)
        self.batch_size = self.config.batch_size
        self.device = self.arg.gpu_devices[0]
        return
    
    def generate_embeddings(self, input_tokens:dict, output_value:str=None, show_progress_bar:bool=True, progress_bar_desc:str=None, position=0):
        """_summary_

        Args:
            input_tokens (dict): a dictionary contains the following fields: input_ids, attention_mask 
            output_value (str, optional): _description_. Defaults to None.
            show_progress_bar (bool, optional): _description_. Defaults to True.
            process_bar_desc (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """        
        for key in input_tokens.keys():
            input_tokens[key] = input_tokens[key].to(self.device)
        return self.lm.encode_tokens(tokens=input_tokens,
                              batch_size=self.batch_size,
                              show_progress_bar=show_progress_bar,
                              position=position,
                              progress_bar_desc=progress_bar_desc,
                              output_value=output_value,
                              output_list=False,
                              device=self.device,
                              padding_size=None)
        
