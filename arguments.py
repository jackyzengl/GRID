import argparse
import configparser
import ast
import os
import shutil
from typing import *
from utils.logging import get_logger
import yaml

logger = get_logger(__name__)

class Config:
    def __init__(self, path):
        # Check path validity
        assert os.path.isfile(path), f"Cannot find configuration file of checkpoint at config_path: {path}" 
        self.configuration_path = path
        if path.endswith('.yaml'):
            self.load_yaml()
        else:
            self.load_config()

    def load_yaml(self):
        def setObjectFromDict(input:dict):
            for key, value in input.items():
                if key != 'configuration_path':
                        setattr(self, key, value)
                # The following sets dictionary as sub class
                # if isinstance(value, dict):
                #     setattr(self, key, ObjectFromDict(value))
                # else:
                #     if key != 'configuration_path':
                        # setattr(self, key, value)

        with open(self.configuration_path, 'r') as file:
            setObjectFromDict(yaml.safe_load(file))
        

    def load_config(self):
        # Create config parser
        config = configparser.ConfigParser()
        # Read config from path
        config.read(self.configuration_path)
        # Iterate over each section in the configuration file
        for section in config.sections():            
            for option in config.options(section):
                value = config.get(section, option)
                try:
                    # cast value to appropriate types
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
                # Write configuration file variable to class variables
                if option != 'configuration_path':
                    setattr(self, option, value)
        return
    
    def get(self, attribute_name:str, default_value:Any=None) -> Any:
        """Savely get the attribute

        Args:
            attribute_name (str): _description_
            default_value (Any, optional): The default value to get the attribute if attribute does not exist. Defaults to None.
        Raise:
            when the attribute does not exist in configuration and default is not given
        Returns:
            Any : the value of the attribute
        """             
        if hasattr(self, attribute_name):
            return getattr(self, attribute_name)
        else:
            if default_value:
                setattr(self, attribute_name, default_value)
                return default_value
            else:
                raise NameError(f'{attribute_name} does not exist in configuration')

    def get_save_path(self, log_dir):
        file_name = os.path.basename(self.configuration_path)
        save_path = os.path.join(log_dir, file_name)
        return save_path

    def write_config(self, save_path):
        '''
            path: the folder path for the configuration file to be saved to
        '''
        if os.path.isdir(save_path):
            shutil.copy(self.configuration_path, save_path)
        else:
            logger.error(f'Cannot save file to {save_path} since the path is invalid.')
        return save_path

class ObjectFromDict:
    def __init__(self, input:dict):
        for key, value in input.items():
            if isinstance(value, dict):
                setattr(self, key, ObjectFromDict(value))
            else:
                setattr(self, key, value)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', "True"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_gpu_devices(input):
    if input == 'auto':
        return input
    numbers = input.split(',')
    parsed_numbers = []
    for num in numbers:
        try:
            parsed_numbers.append(int(num))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid number: {num}")
    return parsed_numbers

def preprocessor_parser(parser):
    # ADD ARGS IF YOU WISH
    parser.add_argument('--preprocessed_data_path', 
                    type=str, 
                    default='preprocess_data',
                    help='The output path of the pre-processed dataset')
    # Parse arguments
    args = parser.parse_args()

    # Parse configuration
    config = Config(args.config_path)
    
    return args, config

def trainer_parser(parser):
    # ADD ARGS IF YOU WISH
    parser.add_argument('--preprocessed_data_path', 
                        type=str, 
                        default='preprocess_data',
                        help='The path of the pre-processed dataset')
    parser.add_argument('--fit_flag',
                        default=True,
                        type=str2bool,
                        help='True: fit the network to dataset; False: run predict')
    parser.add_argument('--from_ckpt_flag',
                        default=False,
                        type=str2bool,
                        help='True: run prediction only from checkpoint; False: Start new model')
    parser.add_argument('--ckpt_path',
                        type=str,
                        help='The path to the checkpoint file which ends with ".ckpt", the directory requires a configuration file at ../../hparam.cfg')
    
    # Parse arguments
    args = parser.parse_args()
    
    # overwrite parameter file directory when running from checkpoint
    if args.from_ckpt_flag:
        assert args.ckpt_path, "Check point path must be specified" 
        assert os.path.exists(args.ckpt_path), f"Check point path is not valid. ckpt_path: {args.ckpt_path}" 
        # overwrite configuration file to f'../../{ckpt_path}'
        # config_path = os.path.dirname(os.path.dirname(args.ckpt_path))+'/hparams.yaml'
        config_path = os.path.dirname(os.path.dirname(args.ckpt_path))+'/hparams.cfg'
        if not os.path.exists(config_path):
            config_path = os.path.dirname(os.path.dirname(args.ckpt_path))+'/hparams.yaml'
        args.config_path = config_path
    logger.debug(args.config_path)
    
    # check path validity
    assert os.path.exists(args.config_path), f"Cannot find configuration file of checkpoint at config_path: {args.config_path}" 
    
    # Parse by config parser
    config = Config(args.config_path)
        
    return args, config

def print_args(args):
    # Print the values
    logger.debug('##############Parsing arguments:##############')
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        logger.debug(f'{arg_name}: {arg_value}')
    logger.debug('##########################################')


def create_parser(type='trainer', print_arg_flag=False, print_config_flag=False)->Tuple:
    """create argument parser, parse arguments and configurations

    Args:
        type (str, optional): the parser type. Defaults to 'trainer'. 
        Set to 'preprocessor' when parser is used by data preprocessor

    Returns:
        args: a class that holds the arguments, 
        config: a class that holds the configuration 
    """    
    # Create an ArgumentParser object with values
    parser = argparse.ArgumentParser(description='My Command Line Program')
    
    # ADD SHARED ARGS
    parser.add_argument('--accelerator',
                        default='gpu',
                        type=str)
    # python arg.py --gpu_devices 0,1
    parser.add_argument('--gpu_devices', 
                        metavar='N',
                        help='<Optional> Set gpu devices, default is "auto". Example: --gpu_devices 0,1', 
                        type=parse_gpu_devices,
                        default='auto'
                        )
    parser.add_argument('--data_path', 
                        type=str, 
                        default='dataset',
                        help='The path of the raw dataset')
    parser.add_argument('--config_path',
                        default='hparams.cfg',
                        type=str,
                        help='The path to the configuration file path')
    # Parse arguments and configurations 
    if type == 'trainer':
        args, configs = trainer_parser(parser)
    elif type == 'preprocessor':
        args, configs = preprocessor_parser(parser)
    else:
        raise AssertionError(f'Parser type {type} not recognized')
    
    # Print variables
    if print_arg_flag:
        print_args(args)
    if print_config_flag:
        print_args(configs)

    return args, configs
