import logging
import colorlog
import torch
import threading
from lightning.pytorch.utilities.rank_zero import rank_zero_only

# Use lightning util to check if we are on global rank zero
@rank_zero_only
def check_rank():
    return True

def get_logger(logger_name=None, logging_level=logging.DEBUG, save_dir=None):
    # Create a logger
    logger = colorlog.getLogger(logger_name)

    # Set the logging level to DEBUG
    logger.setLevel(logging_level)

    # Disable logging matplotlib
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Create a stream handler with a colored formatter
    console_handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(formatter)

    # Add the console_handler to the logger
    logger.addHandler(console_handler)

    # Create a debug logger on disk
    if save_dir:
        write_handler = logging.FileHandler(save_dir)
        logger.addHandler(write_handler)

    # Disable printing by removing handlers if we are not on global rank 0
    if not check_rank():
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    return logger
