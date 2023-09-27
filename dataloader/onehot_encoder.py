import numpy as np
import torch
from typing import *

class custom_onehot_encoder():
    def __init__(self, input_list) -> None:
        # Convert to numpy array
        if isinstance(input_list, list):
            input_list_np = np.array(input_list).squeeze()
        elif isinstance(input_list, np.ndarray):
            input_list_np = input_list.squeeze()
        elif isinstance(input_list, torch.Tensor):
            input_list_np = np.array(input_list.cpu().numpy()).squeeze()
        else:
            raise AssertionError(f'Input type {type(input_list)} is not accepted')
        
        # Check dimension validity
        assert input_list_np.ndim == 1, 'The number of input dimension must not be greater or less than one'

        # Gather class variables
        self.idx_to_category = {idx:val for idx, val in enumerate(input_list_np)}
        self.category_to_idx = {val:idx for idx, val in enumerate(input_list_np)}
        self.len = input_list_np.shape[0]

    def __len__(self):
        """Get the number of categories recorded

        Returns:
            int: the number of categories recorded
        """        
        return self.len
    
    def __get_item__(self, index)->Any:
        """Get category name by index

        Args:
            index (int): The index pointing to the category name

        Returns:
            Any: The recorded category name
        """        
        return self.idx_to_category[index]

    def transform(self, category_name:Union[str, int, float], output_type:str='tensor')->Union[list, torch.Tensor, np.ndarray]:
        """Transform a category name to onehot encoded array

        Args:
            category_name (Any): The category name to be encoded
            output_type (str, optional): Specify which output type is used. Defaults to 'tensor'.

        Returns:
            Union[list, torch.Tensor, np.ndarray]: Output one hot encoded array of the given category name
        """        
        assert category_name in self.category_to_idx.keys(), f'The input category {category_name} is not recorded.'
        assert isinstance(category_name, (str, int, float)), f'The type of category name {type(category_name)} is invalid.'

        data = [0]*self.len
        data[self.category_to_idx[category_name]]=1

        if output_type == 'list':
            return data
        elif output_type == 'tensor':
            return torch.tensor(data, dtype=torch.float32)
        elif output_type == 'numpy':
            return np.array(data)
        else:
            raise AssertionError(f'Output data type {output_type} not recognized')

    def inverse_transform(self, one_hot_array:Union[list, np.ndarray, torch.Tensor]):
        """Revert the onehot encoded array to category name

        Args:
            one_hot_array (Union[list, np.ndarray, torch.Tensor]): 
                The array of numbers to be recovered category names from. 
                The element with the highest number will be used as the index of category.

        Raises:
            AssertionError: _description_

        Returns:
            _type_: _description_
        """           
        # Convert to numpy array
        if isinstance(one_hot_array, list):
            one_hot_array = np.array(one_hot_array) 

        elif isinstance(one_hot_array, np.ndarray) or isinstance(one_hot_array, torch.Tensor):
            pass

        else:
            raise AssertionError(f'Input type {type(one_hot_array)} is not accepted')

        one_hot_array = one_hot_array.squeeze()
        assert one_hot_array.ndim == 1, f'The dimension of input array cannot be greater or less than one.'
        assert one_hot_array.shape[0] == self.len, f'Index {one_hot_array.shape[0]} is out of range'
        idx = int(one_hot_array.argmax())
        return self.idx_to_category[idx]
    
