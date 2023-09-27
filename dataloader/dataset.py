import os
import numpy as np
from typing import *
import h5py
import sys
import torch

class InstructSG_Dataset():
    file_mode_keys = ['torch', 'hdf5']

    def __init__(self, config, data_path: str) -> None:
        '''
        config: a class contains all varaibles in the .cfg file to set up the processor
        data_path: the path for data loader to load raw data
        '''
        self.config = config
        self.data_path = data_path
        self.file_mode = config.get('save_type', 'torch')

        # Check valid file mode, force to default value if the mode is invalid
        if self.file_mode not in self.file_mode_keys:
            print(f'File mode "{self.file_mode}" is invalid. Set the file mode to the default "{self.file_mode_keys[0]}"')
            self.file_mode = self.file_mode_keys[0]

        # Get pointers to data saved on disk 
        if self.file_mode == self.file_mode_keys[1]:
            # Get file handler
            self.data_pointer = h5py.File(self.data_path, 'r')
            # Get length of data
            self.len_data = len(self.data_pointer) if config.dataset_size<=0 else min(len(self.data_pointer), config.dataset_size)
        else:
            # Get the file name
            data_file_list = os.listdir(self.data_path)
            # Sort file names regardless of extension
            data_file_list = sorted(data_file_list, key=lambda x: os.path.splitext(x)[:-1])
            # Store the sorted file directories
            self.data_pointer = []
            for i in range(len(data_file_list)):
                if data_file_list[i].endswith('.pt'):
                    self.data_pointer.append(os.path.join(self.data_path, data_file_list[i]))
            # Get length of dataset
            self.len_data = len(self.data_pointer) if config.dataset_size<=0 else min(len(self.data_pointer), config.dataset_size)
        
        assert self.len_data>0, f'The length of dataset {self.len_data} invalid'
        config.dataset_size = self.len_data

        return


    def __len__(self) -> int:
        """ get the length of dataset

        Returns:
            int: length of data
        """        
        return self.len_data

    def __getitem__(self, index: int) -> dict: 
        """ reads a sample of index from the preprocessed dataset

        Args:
            index (int): the sample index  

        Returns:
            dict: Nested dictionary of numpy arrays of current sample
        """            
        if self.file_mode == 'hdf5':
            sample_handler = self.data_pointer[f'sample_{index+1}']
            sample = self.hdf5_to_dict(sample_handler)
        else:
            sample = torch.load(self.data_pointer[index], map_location='cpu')

        return sample 
        
    
    def hdf5_to_dict(self, group:Any) -> Any:
        """
        Recursively converts an HDF5 group to a nested dictionary.
        """
        if isinstance(group, h5py.File):
            d = {}
            for key in group.keys():
                d[key] = self.hdf5_to_dict(group[key])
            return d
        elif isinstance(group, h5py.Group):
            d = {}
            for key in group.keys():
                d[key] = self.hdf5_to_dict(group[key])
            return d
        elif isinstance(group, h5py.Dataset):
            return torch.from_numpy(group[:])
            # return np.array(group)
